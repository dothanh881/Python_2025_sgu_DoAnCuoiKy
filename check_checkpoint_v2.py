#!/usr/bin/env python3
"""
check_checkpoint_v2.py

Kiểm tra các checkpoint dạng V2 được tạo bởi `train_v2.py`.
In ra metadata cơ bản để bạn dễ so sánh "latest" và "best".

Usage:
    python check_checkpoint_v2.py --checkpoint_dir /path/to/checkpoints

Outputs:
 - Danh sách các file checkpoint_v2_* và BEST_checkpoint_v2_*.pth
 - Thông tin (nếu có): epoch, best_bleu4, decoder_type, encoder_backbone
 - Kiểm tra presence: encoder, decoder, optimizer

File này chỉ đọc checkpoint bằng torch.load(..., weights_only=False) để
hỗ trợ cả dạng state_dict và (kém khuyến nghị) module object.
"""

from __future__ import annotations

import argparse
import json
import os
import glob
import torch
from typing import Dict, Any


def safe_load_checkpoint(path: str) -> Dict[str, Any] | None:
    """Thử load checkpoint, trả về dict hoặc None nếu lỗi."""
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        return ck
    except Exception as e:
        print(f"  ! Lỗi khi load '{path}': {e}")
        return None


def inspect_ck(ck: Dict[str, Any]) -> Dict[str, Any]:
    """Rút trích metadata hữu ích từ checkpoint (không raise lỗi).
    Trả về dict đơn giản chứa các trường: epoch, best_bleu4, decoder_type, encoder_backbone, keys_present
    """
    meta = {}
    # epoch
    meta['epoch'] = ck.get('epoch', ck.get('iter', None))

    # best bleu
    meta['best_bleu4'] = ck.get('best_bleu4', ck.get('bleu-4', ck.get('best_bleu', None)))

    # decoder_type, encoder_backbone
    meta['decoder_type'] = ck.get('decoder_type', None)
    meta['encoder_backbone'] = ck.get('encoder_backbone', None)

    # presence of main keys
    keys = {}
    for k in ['encoder', 'decoder', 'optimizer']:
        keys[k] = k in ck
    meta['keys_present'] = keys

    return meta


def format_meta(path: str, meta: Dict[str, Any]) -> str:
    s = []
    s.append(f" Đang kiểm tra checkpoint: {path}")
    if meta.get('epoch') is not None:
        s.append(f"  - Epoch: {meta['epoch']}")
    if meta.get('best_bleu4') is not None:
        try:
            val = float(meta['best_bleu4'])
            s.append(f"  - Best BLEU-4: {val:.6f}")
        except Exception:
            s.append(f"  - Best BLEU-4: {meta['best_bleu4']}")
    if meta.get('decoder_type') is not None:
        s.append(f"  - Decoder type: {meta['decoder_type']}")
    if meta.get('encoder_backbone') is not None:
        s.append(f"  - Encoder backbone: {meta['encoder_backbone']}")

    for k, present in meta.get('keys_present', {}).items():
        s.append(f"   {k}: {'OK' if present else 'MISSING'}")

    return '\n'.join(s)


def main():
    parser = argparse.ArgumentParser(description='Kiểm tra checkpoint V2 (train_v2.py format)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Thư mục chứa checkpoint V2')
    parser.add_argument('--pattern', type=str, default='checkpoint_v2_*_epoch_*.pth', help='Pattern để tìm checkpoint epoch')
    args = parser.parse_args()

    ck_dir = os.path.abspath(args.checkpoint_dir)
    print('CHECKPOINT_DIR =', ck_dir)
    if not os.path.isdir(ck_dir):
        print(' Thư mục checkpoint không tồn tại:', ck_dir)
        return

    # tìm BEST và các epoch
    best_glob = glob.glob(os.path.join(ck_dir, 'BEST_checkpoint_v2_*.pth'))
    epoch_glob = glob.glob(os.path.join(ck_dir, args.pattern))

    # sort epoch files by epoch number if possible
    def epoch_key(p: str):
        bn = os.path.basename(p)
        # cố parse số epoch trong tên
        import re
        m = re.search(r'epoch_(\d+)', bn)
        if m:
            return int(m.group(1))
        return 0

    epoch_glob = sorted(epoch_glob, key=epoch_key)

    if best_glob:
        print('Found BEST checkpoint(s):')
        for p in best_glob:
            print(' -', p)
    else:
        print('BEST checkpoint not found in', ck_dir)

    if epoch_glob:
        print('\nFound epoch checkpoints:')
        for p in epoch_glob:
            print(' -', p)
    else:
        print('\nNo epoch checkpoints found (pattern {})'.format(args.pattern))

    print('\nCheckpoint metadata:')
    # show BEST first then epochs
    to_check = best_glob + epoch_glob
    if not to_check:
        print(' (Không có file để kiểm tra)')
        return

    for p in to_check:
        print('\n->', p)
        ck = safe_load_checkpoint(p)
        if ck is None:
            continue
        meta = inspect_ck(ck)
        print(format_meta(p, meta))


if __name__ == '__main__':
    main()

