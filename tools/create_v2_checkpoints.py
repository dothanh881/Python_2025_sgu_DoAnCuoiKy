"""
Script tạo checkpoint "v2" (state_dict format) từ các checkpoint gốc (.pth.tar hoặc .pth).
Nó tìm các file nguồn (hoặc một file BEST và một file checkpoint), đọc các khóa
(chẳng hạn 'epoch','encoder','decoder','optimizer','bleu-4'...) và ghi lại dưới
định dạng tương thích với `train_v2.py` / notebook: keys: epoch, encoder, decoder, optimizer, best_bleu4.

Usage:
    python tools/create_v2_checkpoints.py \
        --src_files ../BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar \
        --out_dir ../checkpoints --decoder_type transformer

Nếu không chỉ định --src_files, script sẽ cố gắng tìm các file
BEST_checkpoint_*.pth.tar và checkpoint_*.pth.tar trong project root.

"""
from __future__ import annotations
import argparse
import os
import glob
import torch
import shutil


def normalize_ck(ck):
    """Chuẩn hóa checkpoint loaded: trả về dict có các khóa mong muốn."""
    out = {}
    # epoch
    out['epoch'] = int(ck.get('epoch', ck.get('iter', ck.get('epochs', 0))))

    # best bleu
    best_bleu = ck.get('best_bleu4', ck.get('bleu-4', ck.get('best_bleu', None)))
    if best_bleu is None:
        # try nested
        best_bleu = ck.get('metrics', {}).get('bleu-4') if isinstance(ck.get('metrics', None), dict) else None
    out['best_bleu4'] = float(best_bleu) if best_bleu is not None else 0.0

    # encoder
    if 'encoder' in ck:
        out['encoder'] = ck['encoder']
    else:
        # some checkpoints store full module under key 'model' or top-level 'encoder' missing
        for k in ['model', 'state_dict']:
            if k in ck:
                out['encoder'] = ck[k]
                break
    # decoder
    if 'decoder' in ck:
        out['decoder'] = ck['decoder']

    # optimizer
    if 'optimizer' in ck:
        out['optimizer'] = ck['optimizer']

    # fallback: try to include any tensors that look like state_dicts
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_files', nargs='*', default=[], help='paths to source checkpoint files')
    parser.add_argument('--out_dir', default='checkpoints', help='output directory')
    parser.add_argument('--decoder_type', default='transformer', choices=['lstm','transformer'])
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if not args.src_files:
        # try find common names
        cand = glob.glob(os.path.join(project_root, 'BEST_checkpoint_*.pth*')) + glob.glob(os.path.join(project_root, 'checkpoint_*.pth*'))
        if not cand:
            print('No candidate source checkpoints found in project root. Provide --src_files')
            return
        args.src_files = cand

    os.makedirs(args.out_dir, exist_ok=True)

    created = []
    for src in args.src_files:
        try:
            print('Loading', src)
            ck = torch.load(src, map_location='cpu')
        except Exception as e:
            print('Failed to load', src, '->', e)
            continue

        meta = normalize_ck(ck)
        epoch = meta.get('epoch', 0)
        best = meta.get('best_bleu4', 0.0)

        # build state
        state = {
            'epoch': epoch,
            'best_bleu4': best,
        }

        if 'encoder' in meta and meta['encoder'] is not None:
            state['encoder'] = meta['encoder']
        elif 'encoder' in ck and ck['encoder'] is not None:
            state['encoder'] = ck['encoder']

        if 'decoder' in meta and meta['decoder'] is not None:
            state['decoder'] = meta['decoder']
        elif 'decoder' in ck and ck['decoder'] is not None:
            state['decoder'] = ck['decoder']

        if 'optimizer' in meta and meta['optimizer'] is not None:
            state['optimizer'] = meta['optimizer']
        elif 'optimizer' in ck and ck['optimizer'] is not None:
            state['optimizer'] = ck['optimizer']

        # if the loaded ck is a state_dict for a single module, try to guess
        # but we prefer saving whatever available

        # choose output name
        base_name = os.path.basename(src)
        is_best_src = base_name.lower().startswith('best') or 'best' in base_name.lower()

        if is_best_src:
            out_name = f'BEST_checkpoint_v2_{args.decoder_type}.pth'
        else:
            out_name = f'checkpoint_v2_{args.decoder_type}_epoch_{epoch}.pth'

        dst = os.path.join(args.out_dir, out_name)
        print('Saving v2 checkpoint to', dst)

        try:
            torch.save(state, dst)
            created.append(dst)
        except Exception as e:
            print('Failed to save', dst, '->', e)

    print('\nCreated files:')
    for p in created:
        print(' -', p)


if __name__ == '__main__':
    main()

