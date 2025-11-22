"""
train_v2.py

Huấn luyện Model V2 cho Image Captioning với kiến trúc:

- EncoderV2: dùng backbone ViT (từ timm), trả về feature (B, num_pixels, encoder_dim).
- Decoder:
    + DecoderV2_LSTM (LSTM + Attention), tái sử dụng từ model V1.
    + DecoderV2_Transformer (nn.TransformerDecoder).

Chọn decoder thông qua tham số: --decoder_type {lstm, transformer}.

Chỉ train decoder (encoder frozen).
Lưu checkpoint + BEST checkpoint theo BLEU-4 trên tập VAL.

Yêu cầu:
- Cấu trúc dataset, utils giống repo "a-PyTorch-Tutorial-to-Image-Captioning".
- model_v2_encoder_cnn.py trong cùng project với EncoderV2, DecoderV2_LSTM, DecoderV2_Transformer, ModelV2Config, build_model_v2.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence

from datasets import CaptionDataset
from utils import AverageMeter, clip_gradient

# Import model V2 (encoder + decoder + config + factory)
try:
    from model_v2_encoder_cnn import ModelV2Config, build_model_v2  # same dir
except Exception:  # nếu bạn đặt trong thư mục models/
    from models.model_v2_encoder_cnn import ModelV2Config, build_model_v2  # type: ignore


# ============================================================
# 1. Argparse
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V2 Image Captioning (LSTM/Transformer decoder)")

    # Encoder / Decoder config
    parser.add_argument("--encoder_backbone", type=str, default="vit_small_patch16_224",
                        help="backbone cho EncoderV2 (ví dụ: vit_small_patch16_224)")
    parser.add_argument("--decoder_type", type=str, default="transformer",
                        choices=["lstm", "transformer"],
                        help="kiểu decoder: lstm hoặc transformer")

    # Data
    parser.add_argument("--data_folder", type=str, required=True,
                        help="thư mục chứa dataset processed (HDF5 + json)")
    parser.add_argument("--data_name", type=str, required=True,
                        help="tên base cho processed files, ví dụ flickr8k_5_cap_per_img_5_min_word_freq")

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_v2",
                        help="thư mục lưu checkpoint")

    # Train hyperparams
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", type=str, default="",
                        help="đường dẫn tới checkpoint để resume (nếu có)")
    # Performance / training options
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="số bước để tích luỹ gradient trước khi update optimizer")
    parser.add_argument("--use_amp", action="store_true",
                        help="bật mixed-precision training (AMP) khi dùng CUDA")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="DataLoader prefetch_factor (số batch mỗi worker giữ trong bộ đệm)")

    # Encoder dims / image
    parser.add_argument("--encoded_image_size", type=int, default=14,
                        help="kích thước không gian cho encoder (ViT patch16_224 -> 14x14)")
    parser.add_argument("--encoder_dim", type=int, default=512,
                        help="encoder output dim dùng cho decoder")

    # Transformer-specific
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--pad_idx", type=int, default=0)

    # LSTM-specific (giống tutorial gốc)
    parser.add_argument("--attention_dim", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--decoder_dim", type=int, default=512)

    # Device
    parser.add_argument("--no_cuda", action="store_true", help="tắt CUDA")

    return parser.parse_args()


# ============================================================
# 2. Helper: word map + dataloaders
# ============================================================


def load_word_map(data_folder: str, data_name: str) -> dict:
    path = os.path.join(data_folder, f"WORDMAP_{data_name}.json")
    with open(path, "r") as f:
        return json.load(f)


def create_dataloaders(
    data_folder: str,
    data_name: str,
    batch_size: int,
    workers: int = 1,
    pin_memory: bool = False,
    backbone: str = "vit_small_patch16_224",
    encoded_image_size: int = 14,
    no_autoresize: bool = False,
    prefetch_factor: int = 2,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    num_workers = max(0, int(workers))

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, "TRAIN", transform=transforms.Compose([normalize])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, "VAL", transform=transforms.Compose([normalize])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    return train_loader, val_loader


# ============================================================
# 3. Train 1 epoch
# ============================================================


def train_one_epoch(
    train_loader,
    encoder: nn.Module,
    decoder: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
    decoder_type: str,
    pad_idx: int,
    print_freq: int = 100,
    accumulation_steps: int = 1,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    """
    Train cho 1 epoch.

    - encoder: frozen, chỉ forward (no grad).
    - decoder: LSTM hoặc Transformer.
    """
    encoder.eval()
    decoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - start)

        # TRAIN split: (img, caption, caplen)
        if len(batch) == 3:
            images, caps, caplens = batch
        else:
            images, caps, caplens, _ = batch  # đề phòng

        images = images.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward encoder (no grad). If AMP enabled use autocast for forward.
        from contextlib import nullcontext
        autocast = nullcontext
        if use_amp and scaler is not None and device.type == "cuda":
            try:
                from torch.cuda.amp import autocast as _autocast

                autocast = _autocast
            except Exception:
                autocast = nullcontext

        with torch.no_grad():
            with autocast():
                encoder_out = encoder(images)  # (B, S_enc, encoder_dim)

        # Compute decoder outputs and loss; do under autocast if AMP enabled
        if use_amp and scaler is not None and device.type == "cuda":
            with autocast():
                if decoder_type == "lstm":
                    scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                        encoder_out, caps, caplens
                    )
                    targets = caps_sorted[:, 1:]

                    scores_packed = pack_padded_sequence(
                        scores, decode_lengths, batch_first=True
                    )[0]
                    targets_packed = pack_padded_sequence(
                        targets, decode_lengths, batch_first=True
                    )[0]

                    loss = criterion(scores_packed, targets_packed)

                    # attention regularization
                    if alphas is not None:
                        loss = loss + ((1.0 - alphas.sum(dim=1)) ** 2).mean()
                else:
                    tgt_input = caps[:, :-1]
                    tgt_target = caps[:, 1:]

                    logits = decoder(encoder_out, tgt_input, caption_lengths=caplens)
                    B, S_minus1, V = logits.size()

                    logits_flat = logits.view(B * S_minus1, V)
                    targets_flat = tgt_target.reshape(-1)

                    loss = criterion(logits_flat, targets_flat)
        else:
            if decoder_type == "lstm":
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                    encoder_out, caps, caplens
                )
                targets = caps_sorted[:, 1:]

                scores_packed = pack_padded_sequence(
                    scores, decode_lengths, batch_first=True
                )[0]
                targets_packed = pack_padded_sequence(
                    targets, decode_lengths, batch_first=True
                )[0]

                loss = criterion(scores_packed, targets_packed)

                # attention regularization
                if alphas is not None:
                    loss += ((1.0 - alphas.sum(dim=1)) ** 2).mean()
            else:
                tgt_input = caps[:, :-1]
                tgt_target = caps[:, 1:]

                logits = decoder(encoder_out, tgt_input, caption_lengths=caplens)
                B, S_minus1, V = logits.size()

                logits_flat = logits.view(B * S_minus1, V)
                targets_flat = tgt_target.reshape(-1)

                loss = criterion(logits_flat, targets_flat)

        # scale loss for gradient accumulation
        loss = loss / float(accumulation_steps)

        if use_amp and scaler is not None:
            from torch.cuda.amp import autocast

            # perform backward with AMP
            with autocast():
                # loss already computed inside autocast branch above only if logits computed under autocast.
                pass
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # optimizer step every accumulation_steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            if use_amp and scaler is not None:
                # unscale before clipping
                scaler.unscale_(optimizer)
                clip_gradient(optimizer, 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                clip_gradient(optimizer, 5.0)
                optimizer.step()
            optimizer.zero_grad()

        # note: losses.avg is in terms of per-batch (not adjusted for accumulation)
        losses.update(loss.item() * float(accumulation_steps), images.size(0))
        batch_time.update(time.time() - start)
        start = time.time()

        if i % print_freq == 0:
            print(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                f"Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})"
            )

    return losses.avg


# ============================================================
# 4. Validation (BLEU-4 trên VAL)
# ============================================================


def validate(
    val_loader,
    encoder: nn.Module,
    decoder: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    decoder_type: str,
    pad_idx: int,
) -> float:
    """
    Validation + tính BLEU-4 (naive, dùng id token như word).
    """
    encoder.eval()
    decoder.eval()

    losses = AverageMeter()
    references = []
    hypotheses = []

    with torch.no_grad():
        for images, caps, caplens, allcaps in val_loader:
            images = images.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            encoder_out = encoder(images)

            if decoder_type == "lstm":
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                    encoder_out, caps, caplens
                )
                targets = caps_sorted[:, 1:]

                scores_packed = pack_padded_sequence(
                    scores, decode_lengths, batch_first=True
                )[0]
                targets_packed = pack_padded_sequence(
                    targets, decode_lengths, batch_first=True
                )[0]

                loss = criterion(scores_packed, targets_packed)

                # Predictions: argmax each timestep
                _, preds = scores.max(dim=2)
                preds_list = []
                for j, dec_len in enumerate(decode_lengths):
                    preds_list.append(preds[j, :dec_len].tolist())
            else:
                tgt_input = caps[:, :-1]
                tgt_target = caps[:, 1:]

                logits = decoder(encoder_out, tgt_input, caption_lengths=caplens)
                B, S_minus1, V = logits.size()
                logits_flat = logits.view(B * S_minus1, V)
                targets_flat = tgt_target.reshape(-1)
                loss = criterion(logits_flat, targets_flat)

                _, preds = logits.max(dim=2)
                preds_list = [p.tolist() for p in preds]

            losses.update(loss.item(), images.size(0))

            # references: list of list of refs per image
            allcaps_list = allcaps.tolist()
            for img_caps in allcaps_list:
                img_refs = []
                for ref in img_caps:
                    ref = [w for w in ref if w != pad_idx]
                    img_refs.append([str(w) for w in ref])
                references.append(img_refs)

            # hypotheses
            for pred in preds_list:
                hyp = [w for w in pred if w != pad_idx]
                hypotheses.append([str(w) for w in hyp])

    bleu4 = corpus_bleu(references, hypotheses)
    print(f"Validation Loss: {losses.avg:.4f}  BLEU-4: {bleu4:.4f}")
    return bleu4


# ============================================================
# 5. Main
# ============================================================


def main() -> None:
    args = parse_args()

    device = (
        torch.device("cpu")
        if args.no_cuda
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    cudnn.benchmark = True

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load vocab
    word_map = load_word_map(args.data_folder, args.data_name)
    vocab_size = len(word_map)
    pad_idx = args.pad_idx

    # Build ModelV2Config từ args
    cfg = ModelV2Config(
        encoder_backbone=args.encoder_backbone,
        encoded_image_size=args.encoded_image_size,
        encoder_dim=args.encoder_dim,
        decoder_type=args.decoder_type,
        attention_dim=args.attention_dim,
        embed_dim=args.embed_dim,
        decoder_dim=args.decoder_dim,
        dropout=args.dropout,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        max_len=args.max_len,
        pad_idx=args.pad_idx,
    )

    # Build model
    encoder, decoder = build_model_v2(vocab_size=vocab_size, config=cfg)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # Optimizer: chỉ decoder
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, decoder.parameters()), lr=args.lr
    )

    # Loss: CE ignore pad_idx
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)

    # DataLoader
    train_loader, val_loader = create_dataloaders(
        args.data_folder,
        args.data_name,
        args.batch_size,
        workers=args.workers,
        pin_memory=(device.type == "cuda"),
        backbone=args.encoder_backbone,
        encoded_image_size=args.encoded_image_size,
        no_autoresize=getattr(args, 'no_autoresize', False),
        prefetch_factor=args.prefetch_factor,
    )

    # Setup AMP GradScaler if requested and CUDA available (use new torch.amp API)
    scaler = None
    if args.use_amp and device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler(device_type="cuda")
        except Exception:
            # fallback for older torch versions
            scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    best_bleu4 = 0.0

    # Resume nếu cần
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        try:
            encoder.load_state_dict(ckpt["encoder"])
            decoder.load_state_dict(ckpt["decoder"])
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"Warning: failed to load state_dict from checkpoint: {e}")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_bleu4 = ckpt.get("best_bleu4", 0.0)
        print(f"Resumed from epoch {start_epoch}, best BLEU-4 so far: {best_bleu4:.4f}")

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(
            f"\n=== EPOCH {epoch}/{args.epochs - 1} "
            f"(decoder_type={args.decoder_type}, backbone={args.encoder_backbone}) ==="
        )

        train_loss = train_one_epoch(
            train_loader,
            encoder,
            decoder,
            criterion,
            optimizer,
            epoch,
            device,
            decoder_type=args.decoder_type,
            pad_idx=pad_idx,
            accumulation_steps=getattr(args, 'accumulation_steps', 1),
            use_amp=getattr(args, 'use_amp', False),
            scaler=scaler,
        )
        print(f"Epoch {epoch} train loss: {train_loss:.4f}")

        recent_bleu4 = validate(
            val_loader,
            encoder,
            decoder,
            criterion,
            device,
            decoder_type=args.decoder_type,
            pad_idx=pad_idx,
        )

        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(best_bleu4, recent_bleu4)

        # Save checkpoint
        state = {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_bleu4": best_bleu4,
            "decoder_type": args.decoder_type,
            "encoder_backbone": args.encoder_backbone,
            "encoder_dim": args.encoder_dim,
        }

        ckpt_path = os.path.join(
            args.checkpoint_dir,
            f"checkpoint_v2_{args.decoder_type}_epoch_{epoch}.pth",
        )
        torch.save(state, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

        if is_best:
            best_path = os.path.join(
                args.checkpoint_dir,
                f"BEST_checkpoint_v2_{args.decoder_type}.pth",
            )
            torch.save(state, best_path)
            print(f"Updated BEST checkpoint at {best_path}")

        scheduler.step()


if __name__ == "__main__":
    main()
