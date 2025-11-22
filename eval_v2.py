"""
eval_v2.py

Đánh giá Model V2 (EncoderV2 + DecoderV2_[LSTM|Transformer]) trên tập TEST.

- Load checkpoint V2 (state_dict + metadata: decoder_type, encoder_backbone, encoder_dim).
- Xây dựng lại model bằng ModelV2Config + build_model_v2 từ model_v2_encoder_cnn.py.
- Hỗ trợ:
    + Transformer decoder: dùng decoder.generate() (greedy).
    + LSTM decoder: greedy_decode_lstm hoặc beam_search_decode_lstm.
- Tính các metric:
    + BLEU-1/2/3/4 (nltk.corpus_bleu)
    + METEOR (nltk.meteor_score)
    + ROUGE-L (LCS-based)
    + CIDEr (nếu pycocoevalcap được cài)

Đồng thời lưu N mẫu ảnh + caption dự đoán + caption GT trong save_samples_dir.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from datasets import CaptionDataset

# Import model V2 (config + factory)
try:
    from model_v2_encoder_cnn import ModelV2Config, build_model_v2  # type: ignore
except Exception:
    from models.model_v2_encoder_cnn import ModelV2Config, build_model_v2  # type: ignore

# NLP metrics
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score

# Try import pycocoevalcap CIDEr
try:
    from pycocoevalcap.cider.cider import Cider
    _CIDEr_AVAILABLE = True
except Exception:
    _CIDEr_AVAILABLE = False


# ============================================================
# 1. Argparse
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Model V2 (EncoderV2 + DecoderV2)")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="đường dẫn tới BEST_checkpoint_v2_xxx.pth hoặc checkpoint_v2_xxx_epoch_k.pth")
    parser.add_argument("--data_folder", type=str, required=True,
                        help="thư mục processed dataset (HDF5 + json)")
    parser.add_argument("--data_name", type=str, required=True,
                        help="tên base files, ví dụ flickr8k_5_cap_per_img_5_min_word_freq")

    parser.add_argument("--decoder_type", type=str, default="transformer",
                        choices=["lstm", "transformer"],
                        help="kiểu decoder tương ứng với checkpoint")
    parser.add_argument("--encoder_backbone", type=str, default="vit_small_patch16_224",
                        help="backbone encoder; nếu checkpoint lưu, giá trị này sẽ bị override")
    parser.add_argument("--encoded_image_size", type=int, default=14)
    parser.add_argument("--encoder_dim", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--pad_idx", type=int, default=0)

    # Cho LSTM nếu cần
    parser.add_argument("--attention_dim", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--decoder_dim", type=int, default=512)

    parser.add_argument("--beam_size", type=int, default=0,
                        help="0 hoặc 1 = greedy; >1 = beam search (chỉ hỗ trợ LSTM).")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="nên để 1 khi beam search.")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda hoặc cpu; để None sẽ tự chọn")
    parser.add_argument("--save_samples_dir", type=str, default="./eval_v2_samples")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="số ảnh mẫu cần lưu (ảnh + caption)")

    return parser.parse_args()


# ============================================================
# 2. Decode helper cho LSTM
# ============================================================


def greedy_decode_lstm(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    image: torch.Tensor,
    word_map: dict,
    device: torch.device,
    max_len: int = 50,
) -> List[int]:
    """
    Greedy decode cho LSTM decoder của tutorial gốc.
    """
    with torch.no_grad():
        encoder_out = encoder(image.to(device))  # (1, S_enc, encoder_dim)
        if encoder_out.dim() == 4:
            b, h, w, c = encoder_out.size()
            encoder_out = encoder_out.view(b, h * w, c)

        h, c = decoder.init_hidden_state(encoder_out)
        k_prev_words = torch.LongTensor([[word_map["<start>"]]]).to(device)
        seq: List[int] = []

        for _ in range(max_len):
            embeddings = decoder.embedding(k_prev_words).squeeze(1)
            awe, _ = decoder.attention(encoder_out, h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
            scores = decoder.fc(h)
            scores = F.log_softmax(scores, dim=1)
            _, next_word = scores.max(dim=1)
            next_word_id = int(next_word.item())
            if next_word_id == word_map.get("<end>"):
                break
            seq.append(next_word_id)
            k_prev_words = next_word.unsqueeze(1)

        return seq


def beam_search_decode_lstm(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    image: torch.Tensor,
    word_map: dict,
    device: torch.device,
    beam_size: int = 3,
    max_len: int = 50,
) -> List[int]:
    """
    Beam search cho LSTM decoder (1 ảnh).
    """
    vocab_size = len(word_map)

    with torch.no_grad():
        encoder_out = encoder(image.to(device))
        if encoder_out.dim() == 4:
            b, h, w, c = encoder_out.size()
            encoder_out = encoder_out.view(b, h * w, c)

        encoder_dim = encoder_out.size(-1)
        num_pixels = encoder_out.size(1)

        k = beam_size
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

        k_prev_words = torch.LongTensor([[word_map["<start>"]]] * k).to(device)
        seqs = k_prev_words  # (k, 1)
        top_k_scores = torch.zeros(k, 1).to(device)

        complete_seqs: List[List[int]] = []
        complete_seqs_scores: List[torch.Tensor] = []

        h, c = decoder.init_hidden_state(encoder_out)
        step = 1

        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)
            awe, _ = decoder.attention(encoder_out, h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
            scores = decoder.fc(h)
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            prev_word_inds = (top_k_words // vocab_size).long()
            next_word_inds = (top_k_words % vocab_size).long()

            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

            incomplete_inds = [ind for ind, w in enumerate(next_word_inds) if w != word_map["<end>"]]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                for ci in complete_inds:
                    complete_seqs.append(seqs[ci].tolist())
                    complete_seqs_scores.append(top_k_scores[ci])
            k -= len(complete_inds)

            if k == 0:
                break

            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            if step > max_len:
                break
            step += 1

        if len(complete_seqs) == 0:
            seq = seqs[0].tolist()
        else:
            scores_np = [float(s.cpu().numpy()) for s in complete_seqs_scores]
            best_idx = int(np.argmax(scores_np))
            seq = complete_seqs[best_idx]

        if len(seq) > 0 and seq[0] == word_map.get("<start>"):
            seq = seq[1:]
        return seq


# ============================================================
# 3. Metric helpers
# ============================================================


def compute_rouge_l(reference: List[str], hypothesis: List[str]) -> float:
    """
    ROUGE-L F1 giữa 1 reference và hypothesis (danh sách token).
    """
    m = len(reference)
    n = len(hypothesis)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if reference[i] == hypothesis[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    lcs = dp[0][0]
    if lcs == 0:
        return 0.0
    prec = lcs / float(n)
    rec = lcs / float(m)
    if prec + rec == 0:
        return 0.0
    f1 = (2 * prec * rec) / (prec + rec)
    return f1


# ============================================================
# 4. Evaluate
# ============================================================


def evaluate_all(args: argparse.Namespace) -> None:
    device = (
        torch.device(args.device)
        if args.device
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )

    # word map
    word_map_path = os.path.join(args.data_folder, f"WORDMAP_{args.data_name}.json")
    with open(word_map_path, "r") as f:
        word_map = json.load(f)
    rev_word_map = {v: k for k, v in word_map.items()}

    # Load checkpoint
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder_backbone = ck.get("encoder_backbone", args.encoder_backbone)
    decoder_type = ck.get("decoder_type", args.decoder_type)
    encoder_dim = ck.get("encoder_dim", args.encoder_dim)

    # Build config đúng với model_v2_encoder_cnn.py
    cfg = ModelV2Config(
        encoder_backbone=encoder_backbone,
        encoded_image_size=args.encoded_image_size,
        encoder_dim=encoder_dim,
        decoder_type=decoder_type,
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
    encoder, decoder = build_model_v2(vocab_size=len(word_map), config=cfg)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load state_dict
    if "encoder" in ck and isinstance(ck["encoder"], dict):
        try:
            encoder.load_state_dict(ck["encoder"], strict=False)
        except Exception as e:
            print(f"Warning: failed to load encoder state_dict: {e}")

    if "decoder" in ck and isinstance(ck["decoder"], dict):
        try:
            decoder.load_state_dict(ck["decoder"], strict=False)
        except Exception as e:
            print(f"Warning: failed to load decoder state_dict: {e}")

    encoder.eval()
    decoder.eval()

    # DataLoader TEST
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, "TEST",
                       transform=transforms.Compose([normalize])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=(device.type == "cuda"),
    )

    refs_all: List[List[List[str]]] = []
    hyps_all: List[List[str]] = []

    os.makedirs(args.save_samples_dir, exist_ok=True)
    saved_samples = 0
    idx_global = 0

    # CIDEr storage
    cider_scorer = Cider() if _CIDEr_AVAILABLE else None
    cider_refs = {}
    cider_hyps = {}

    total = len(test_loader.dataset)
    print(f"Evaluating {total} examples on device={device} (decoder_type={decoder_type})")

    for images, caps, caplens, allcaps in tqdm(test_loader, desc="Evaluating"):
        bsz = images.size(0)
        for i in range(bsz):
            img = images[i].unsqueeze(0)  # (1,3,H,W)
            img_caps = allcaps[i].tolist()

            # References dạng từ
            img_refs_tok: List[List[str]] = []
            for ref in img_caps:
                ref_ids = [w for w in ref if w != args.pad_idx]
                tokens = [rev_word_map.get(int(w), "<unk>") for w in ref_ids]
                img_refs_tok.append(tokens)

            # Hypothesis
            if decoder_type == "transformer" and hasattr(decoder, "generate"):
                with torch.no_grad():
                    enc_out = encoder(img.to(device))
                    start_id = word_map.get("<start>")
                    end_id = word_map.get("<end>")
                    seq = decoder.generate(
                        enc_out,
                        start_token_id=start_id,
                        end_token_id=end_id,
                        max_len=args.max_len,
                    )[0].tolist()
                    if len(seq) > 0 and seq[0] == start_id:
                        seq = seq[1:]
                hyp_ids = [w for w in seq if w != args.pad_idx]
            else:
                # LSTM decode
                if args.beam_size and args.beam_size > 1:
                    hyp_ids = beam_search_decode_lstm(
                        encoder, decoder, img, word_map, device,
                        beam_size=args.beam_size, max_len=args.max_len
                    )
                else:
                    hyp_ids = greedy_decode_lstm(
                        encoder, decoder, img, word_map, device,
                        max_len=args.max_len
                    )

            hyp_tokens = [rev_word_map.get(int(w), "<unk>") for w in hyp_ids]

            refs_all.append(img_refs_tok)
            hyps_all.append(hyp_tokens)

            if cider_scorer is not None:
                cider_refs[idx_global] = [" ".join(r) for r in img_refs_tok]
                cider_hyps[idx_global] = [" ".join(hyp_tokens)]

            # Lưu sample
            if saved_samples < args.num_samples:
                # Cố gắng lấy ảnh từ HDF5 gốc
                h5_path = os.path.join(
                    args.data_folder, f"TEST_IMAGES_{args.data_name}.hdf5"
                )
                sample_img_path = None
                try:
                    with h5py.File(h5_path, "r") as h5f:
                        raw = h5f["images"][idx_global]
                        arr = np.transpose(raw, (1, 2, 0)).astype(np.uint8)
                        pil = Image.fromarray(arr)
                        sample_img_path = os.path.join(
                            args.save_samples_dir, f"sample_{saved_samples}.jpg"
                        )
                        pil.save(sample_img_path)
                except Exception:
                    # fallback: reverse normalize từ tensor
                    inv_norm = transforms.Normalize(
                        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                    )
                    try:
                        pil = transforms.ToPILImage()(
                            inv_norm(images[i]).cpu()
                        ).convert("RGB")
                        sample_img_path = os.path.join(
                            args.save_samples_dir, f"sample_{saved_samples}.jpg"
                        )
                        pil.save(sample_img_path)
                    except Exception:
                        sample_img_path = None

                # text
                with open(
                    os.path.join(args.save_samples_dir, f"sample_{saved_samples}.txt"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write("PREDICTED:\n")
                    f.write(" ".join(hyp_tokens) + "\n\n")
                    f.write("REFERENCES:\n")
                    for r in img_refs_tok:
                        f.write(" ".join(r) + "\n")

                saved_samples += 1

            idx_global += 1

    # BLEU-1..4
    weights = {
        "Bleu-1": (1.0, 0, 0, 0),
        "Bleu-2": (0.5, 0.5, 0, 0),
        "Bleu-3": (1 / 3, 1 / 3, 1 / 3, 0),
        "Bleu-4": (0.25, 0.25, 0.25, 0.25),
    }
    bleu_scores = {}
    for name, w in weights.items():
        try:
            bleu_scores[name] = corpus_bleu(refs_all, hyps_all, weights=w)
        except Exception:
            bleu_scores[name] = None

    # METEOR
    meteor_vals = []
    for refs, hyp in zip(refs_all, hyps_all):
        try:
            score = meteor_score([" ".join(r) for r in refs], " ".join(hyp))
        except Exception:
            score = 0.0
        meteor_vals.append(score)
    meteor_avg = float(np.mean(meteor_vals)) if meteor_vals else 0.0

    # ROUGE-L
    rouge_vals = []
    for refs, hyp in zip(refs_all, hyps_all):
        best = 0.0
        for r in refs:
            val = compute_rouge_l(r, hyp)
            if val > best:
                best = val
        rouge_vals.append(best)
    rouge_l_avg = float(np.mean(rouge_vals)) if rouge_vals else 0.0

    # CIDEr
    cider_score = None
    if cider_scorer is not None and cider_refs and cider_hyps:
        cider_score, _ = cider_scorer.compute_score(cider_refs, cider_hyps)
        cider_score = float(cider_score)

    # In kết quả
    print("\n=== Evaluation Results (V2) ===")
    for name in ["Bleu-1", "Bleu-2", "Bleu-3", "Bleu-4"]:
        v = bleu_scores.get(name)
        print(f"{name}: {v:.4f}" if v is not None else f"{name}: n/a")
    print(f"METEOR: {meteor_avg:.4f}")
    print(f"ROUGE-L: {rouge_l_avg:.4f}")
    if cider_score is not None:
        print(f"CIDEr: {cider_score:.4f}")
    else:
        print("CIDEr: n/a (pycocoevalcap not installed)")

    # Lưu summary.json
    summary = {
        "bleu": bleu_scores,
        "meteor": meteor_avg,
        "rouge_l": rouge_l_avg,
        "cider": cider_score,
        "decoder_type": decoder_type,
        "num_samples_saved": saved_samples,
    }
    os.makedirs(args.save_samples_dir, exist_ok=True)
    with open(
        os.path.join(args.save_samples_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {saved_samples} sample(s) and summary to {args.save_samples_dir}")


if __name__ == "__main__":
    args = parse_args()
    evaluate_all(args)
