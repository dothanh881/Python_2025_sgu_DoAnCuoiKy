"""
model_v2_encoder_cnn.py

Phiên bản V2 cho Image Captioning:
- EncoderV2: dùng backbone ViT từ timm, trả về đặc trưng dạng (B, num_pixels, encoder_dim).
- DecoderV2_LSTM: tái sử dụng từ model V1 (LSTM + Attention) thông qua import.
- DecoderV2_Transformer: decoder kiểu Transformer, dùng PyTorch nn.TransformerDecoder.

Mục tiêu: cho phép dễ dàng chuyển đổi giữa LSTM decoder và Transformer decoder
mà không phải thay đổi pipeline dataset/dataloader hiện có.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import timm  # backbone ViT


# ------------------------------------------------------------
# 1. Import Decoder LSTM V1 dưới alias DecoderV2_LSTM
# ------------------------------------------------------------

try:
    # Tutorial gốc thường đặt DecoderWithAttention trong models.py hoặc model.py
    from .model import Decoder as DecoderV2_LSTM  # type: ignore
except Exception:
    try:
        from .models import DecoderWithAttention as DecoderV2_LSTM  # type: ignore
    except Exception:
        DecoderV2_LSTM = None  # type: ignore


# ------------------------------------------------------------
# 2. Cấu hình cho model V2 (encoder + decoder)
# ------------------------------------------------------------

BackboneType = Literal["vit_small_patch16_224"]
DecoderType = Literal["lstm", "transformer"]


@dataclass
class ModelV2Config:
    """
    Cấu hình tổng hợp cho build_model_v2.

    Attributes
    ----------
    encoder_backbone: str
        Tên backbone ViT trong timm (ví dụ: 'vit_small_patch16_224').
    encoded_image_size: int
        Kích thước không gian "ảo" HxW sau khi flatten (thông tin mang tính gợi ý).
        Với ViT patch16_224, số patch = 14x14 = 196.
    encoder_dim: int
        Kích thước đặc trưng encoder sau khi map từ d_model của ViT.

    decoder_type: str
        'lstm' hoặc 'transformer' để chọn decoder.

    # Cho LSTM decoder (tái sử dụng từ V1)
    attention_dim: int
    embed_dim: int
    decoder_dim: int
    dropout: float

    # Cho Transformer decoder
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    max_len: int
    pad_idx: int
    """

    # Encoder
    encoder_backbone: BackboneType = "vit_small_patch16_224"
    encoded_image_size: int = 14
    encoder_dim: int = 512

    # Kiểu decoder
    decoder_type: DecoderType = "transformer"

    # Cho LSTM decoder
    attention_dim: int = 512
    embed_dim: int = 512
    decoder_dim: int = 512
    dropout: float = 0.5

    # Cho Transformer decoder
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 512
    max_len: int = 50
    pad_idx: int = 0


# ------------------------------------------------------------
# 3. EncoderV2 dùng ViT backbone
# ------------------------------------------------------------


class EncoderV2(nn.Module):
    """
    Encoder V2 dùng backbone ViT (timm), trả về (B, num_pixels, encoder_dim).

    - Backbone: vit_small_patch16_224 (mặc định).
    - Lấy patch embeddings (bỏ CLS), sau đó map qua Linear: d_model -> encoder_dim.
    """

    def __init__(
        self,
        backbone: BackboneType = "vit_small_patch16_224",
        encoded_image_size: int = 14,
        encoder_dim: int = 512,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.encoded_image_size = encoded_image_size
        self.encoder_dim = encoder_dim

        # Tạo backbone ViT
        self.backbone = timm.create_model(backbone, pretrained=pretrained)

        # Reset classifier (head) nếu có
        if hasattr(self.backbone, "reset_classifier"):
            try:
                self.backbone.reset_classifier(0)
            except Exception:
                pass

        # Lấy kích thước embed (d_model) của ViT
        if hasattr(self.backbone, "embed_dim"):
            d_model = self.backbone.embed_dim
        else:
            d_model = getattr(self.backbone, "num_features", 384)

        self.d_model = d_model

        # Linear map d_model -> encoder_dim
        self.proj = nn.Linear(d_model, encoder_dim)
        self.ln = nn.LayerNorm(encoder_dim)

        # Số token "prefix" (thường = 1 khi có CLS token)
        self.num_prefix_tokens = getattr(self.backbone, "num_prefix_tokens", 1)

        # Kích thước ảnh mà patch_embed mong đợi
        try:
            self.img_size = tuple(self.backbone.patch_embed.img_size)
        except Exception:
            self.img_size = (224, 224)

        # Bật/tắt auto-resize input trước khi vào ViT
        self.auto_resize: bool = True

    def forward(self, images: Tensor) -> Tensor:
        """
        Parameters
        ----------
        images: Tensor
            Tensor ảnh đầu vào dạng (B, 3, H, W).

        Returns
        -------
        encoder_out: Tensor
            Đặc trưng encoder dạng (B, num_pixels, encoder_dim).
        """
        if images.dim() == 4:
            _, _, H, W = images.shape
            if (H, W) != self.img_size:
                if self.auto_resize:
                    images = F.interpolate(
                        images, size=self.img_size, mode="bilinear", align_corners=False
                    )
                else:
                    # Để timm ViT báo lỗi nếu size không khớp
                    pass

        # forward_features thường trả (B, num_tokens, d_model) hoặc dict
        feats = self.backbone.forward_features(images)

        if isinstance(feats, dict):
            # Một số model trả dict với key 'x'
            if "x" in feats:
                tokens = feats["x"]
            else:
                tokens = next(iter(feats.values()))
        else:
            tokens = feats  # (B, num_tokens, d_model)

        # Loại bỏ CLS token nếu có
        if self.num_prefix_tokens > 0 and tokens.size(1) > self.num_prefix_tokens:
            patch_tokens = tokens[:, self.num_prefix_tokens :, :]
        else:
            patch_tokens = tokens  # (B, num_patches, d_model)

        # Map d_model -> encoder_dim
        enc = self.proj(patch_tokens)  # (B, num_patches, encoder_dim)
        enc = self.ln(enc)

        # num_patches = H_p * W_p (ví dụ 14*14=196); ta không reshape thêm,
        # chỉ đảm bảo shape (B, num_pixels, encoder_dim) thống nhất.
        return enc


# ------------------------------------------------------------
# 4. Positional Encoding cho Transformer Decoder
# ------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """
    Positional Encoding sinusoidal chuẩn (theo paper Attention is All You Need).

    Đầu vào: (B, S, d_model)  ->  (B, S, d_model) sau khi cộng vị trí.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)  # không học, lưu trong state_dict

    def _extend_pe(self, new_max_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Mở rộng buffer `pe` nếu seq_len > hiện tại.

        Tạo positional encoding mới với kích thước tối đa mới (ví dụ gấp đôi hoặc bằng new_max_len).
        """
        cur_max = self.pe.size(1)
        if new_max_len <= cur_max:
            return
        # tăng kích thước (gấp đôi cho dự phòng) hoặc ít nhất bằng new_max_len
        target_len = max(new_max_len, cur_max * 2)

        # Tính pe mới giống như __init__
        d_model = self.d_model
        pe = torch.zeros(target_len, d_model, dtype=dtype, device=device)
        position = torch.arange(0, target_len, dtype=dtype, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=dtype, device=device)
            * (-torch.log(torch.tensor(10000.0, dtype=dtype, device=device)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, target_len, d_model)

        # Ghi đè buffer
        # register_buffer không cần gọi lại; gán sẽ cập nhật tensor buffer
        self.pe = pe

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, S, d_model)
        """
        seq_len = x.size(1)
        # Nếu seq_len lớn hơn buffer hiện tại thì mở rộng buffer động
        if seq_len > self.pe.size(1):
            # mở rộng pe tới ít nhất seq_len
            self._extend_pe(seq_len, device=x.device, dtype=x.dtype)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return self.dropout(x)


# ------------------------------------------------------------
# 5. DecoderV2_Transformer
# ------------------------------------------------------------


class DecoderV2_Transformer(nn.Module):
    """
    Decoder Transformer cho Image Captioning V2.

    - Nhận encoder_out từ EncoderV2 (ViT) dạng (B, S_enc, encoder_dim).
    - Map sang d_model rồi chạy qua TransformerDecoder.
    - Hỗ trợ forward (training) + generate (greedy).
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 50,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_len = max_len

        # Ánh xạ encoder_dim -> d_model
        self.encoder_proj = nn.Linear(encoder_dim, d_model)

        # Embedding cho token
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        # TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Linear ra vocab
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Khởi tạo trọng số tương đối đơn giản
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device: torch.device) -> Tensor:
        """
        Tạo causal mask (S, S) cho TransformerDecoder để không nhìn thấy tương lai.
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        # True: positions to be masked
        return mask

    def forward(
        self,
        encoder_out: Tensor,          # (B, S_enc, encoder_dim)
        encoded_captions: Tensor,     # (B, S_tgt) caption đầu vào (đã shift-right)
        caption_lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward cho training.

        Parameters
        ----------
        encoder_out: Tensor
            (B, S_enc, encoder_dim)
        encoded_captions: Tensor
            (B, S_tgt) - caption input cho decoder (đã shift-right).
        caption_lengths: Tensor, optional
            (B,) hoặc (B,1), dùng để tạo padding mask (tùy ý).

        Returns
        -------
        logits: Tensor
            (B, S_tgt, vocab_size)
        """
        device = encoder_out.device
        B, S_enc, _ = encoder_out.size()
        B2, S_tgt = encoded_captions.size()
        assert B == B2, "Batch size encoder_out và encoded_captions phải khớp."

        # 1) Map encoder_out -> memory (B, S_enc, d_model)
        memory = self.encoder_proj(encoder_out)  # (B, S_enc, d_model)

        # 2) Embed target (B, S_tgt, d_model)
        tgt_embed = self.embedding(encoded_captions) * sqrt(self.d_model)
        tgt_embed = self.pos_encoder(tgt_embed)  # (B, S_tgt, d_model)
        tgt_embed = self.dropout(tgt_embed)

        # 3) Tạo tgt_mask (causal mask) dạng (S_tgt, S_tgt)
        tgt_mask = self._generate_square_subsequent_mask(S_tgt, device=device)

        # 4) Tạo tgt_key_padding_mask (B, S_tgt) mask cho <pad> (True = vị trí pad)
        if caption_lengths is not None:
            # caption_lengths có thể (B,) hoặc (B,1)
            if caption_lengths.dim() == 2:
                cap_lens = caption_lengths.squeeze(1)
            else:
                cap_lens = caption_lengths
            # positions >= length sẽ là pad (True)
            range_row = torch.arange(S_tgt, device=device).unsqueeze(0).expand(B, -1)
            # (B,S_tgt) True = pad
            tgt_key_padding_mask = range_row >= cap_lens.unsqueeze(1)
        else:
            tgt_key_padding_mask = encoded_captions.eq(self.pad_idx)  # (B,S_tgt)

        # 5) Gọi TransformerDecoder
        decoded = self.transformer_decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B, S_tgt, d_model)

        # 6) Qua Linear ra logits vocab
        logits = self.fc_out(decoded)  # (B, S_tgt, vocab_size)

        return logits

    @torch.no_grad()
    def generate(
        self,
        encoder_out: Tensor,   # (B, S_enc, encoder_dim)
        start_token_id: int,
        end_token_id: int,
        max_len: Optional[int] = None,
    ) -> Tensor:
        """
        Sinh caption (greedy decoding) cho mỗi ảnh.

        Parameters
        ----------
        encoder_out: Tensor
            (B, S_enc, encoder_dim)
        start_token_id: int
            ID token <start>
        end_token_id: int
            ID token <end>
        max_len: int, optional
            Độ dài tối đa muốn sinh; nếu None dùng self.max_len.

        Returns
        -------
        captions: Tensor
            (B, L) với L <= max_len, chứa các token ID generated.
        """
        device = encoder_out.device
        B, S_enc, _ = encoder_out.size()
        max_len = max_len or self.max_len

        # Map encoder_out -> memory (B, S_enc, d_model)
        memory = self.encoder_proj(encoder_out)

        # Khởi tạo input với <start>
        ys = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _t in range(1, max_len + 1):
            S_tgt = ys.size(1)

            # Embed + positional
            tgt_embed = self.embedding(ys) * sqrt(self.d_model)
            tgt_embed = self.pos_encoder(tgt_embed)

            tgt_mask = self._generate_square_subsequent_mask(S_tgt, device=device)

            tgt_key_padding_mask = ys.eq(self.pad_idx)

            out = self.transformer_decoder(
                tgt=tgt_embed,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )  # (B, S_tgt, d_model)

            logits = self.fc_out(out)  # (B, S_tgt, vocab_size)
            next_token_logits = logits[:, -1, :]  # (B, vocab_size)
            next_tokens = next_token_logits.argmax(dim=-1)  # (B,)

            # Gắn token mới
            ys = torch.cat([ys, next_tokens.unsqueeze(1)], dim=1)  # (B, S_tgt+1)

            # Cập nhật finished
            finished = finished | (next_tokens == end_token_id)
            if torch.all(finished):
                break

        return ys  # (B, <=max_len+1)


# ------------------------------------------------------------
# 6. Hàm factory: build_model_v2
# ------------------------------------------------------------


def build_model_v2(
    vocab_size: int,
    config: ModelV2Config,
) -> Tuple[nn.Module, nn.Module]:
    """
    Tạo encoder + decoder cho V2 dựa trên cấu hình.

    Parameters
    ----------
    vocab_size: int
        Kích thước vocab (số từ).
    config: ModelV2Config
        Cấu hình chứa encoder_backbone, encoder_dim, decoder_type, v.v.

    Returns
    -------
    encoder: nn.Module
        Instance của EncoderV2 (ViT).
    decoder: nn.Module
        Instance của DecoderV2_Transformer hoặc DecoderV2_LSTM (tùy config.decoder_type).
    """
    # 1) Encoder
    encoder = EncoderV2(
        backbone=config.encoder_backbone,
        encoded_image_size=config.encoded_image_size,
        encoder_dim=config.encoder_dim,
        pretrained=True,
    )

    # 2) Decoder
    if config.decoder_type == "transformer":
        decoder = DecoderV2_Transformer(
            vocab_size=vocab_size,
            encoder_dim=config.encoder_dim,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_len=config.max_len,
            pad_idx=config.pad_idx,
        )
    else:
        if DecoderV2_LSTM is None:
            raise RuntimeError(
                "DecoderV2_LSTM không import được từ model/models. "
                "Hãy kiểm tra lại tên file và class trong project gốc."
            )
        decoder = DecoderV2_LSTM(
            attention_dim=config.attention_dim,
            embed_dim=config.embed_dim,
            decoder_dim=config.decoder_dim,
            vocab_size=vocab_size,
            encoder_dim=config.encoder_dim,
            dropout=config.dropout,
        )

    return encoder, decoder