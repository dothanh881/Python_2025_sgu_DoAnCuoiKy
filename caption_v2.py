"""caption_v2.py

Compatibility caption utilities for Model V2 (ViT encoder + LSTM/Transformer decoders).
- caption_image_beam_search_v2: beam search that works with encoder outputs either
  (B, H, W, C) or (B, num_patches, C).
- greedy_decode_v2 / beam_search_decode_v2: lightweight wrappers for transformer or LSTM decoders.
- visualize_att_v2: display image + attention maps (works when attention maps are available and
  when number of patches is square -> reshape to HxW).

This file intentionally follows the original `caption.py` API but is robust for V2 models.
"""

from typing import Tuple, List, Optional
import math
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform

# device helper
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_token_ids(word_map: dict) -> Tuple[int, int]:
    """Return (start_id, end_id) given a word_map; tries common token names."""
    # word_map: word->idx
    candidates_start = ['<start>', '<START>', 'startseq']
    candidates_end = ['<end>', '<END>', 'endseq']
    start_id = None
    end_id = None
    for k in candidates_start:
        if k in word_map:
            start_id = int(word_map[k])
            break
    for k in candidates_end:
        if k in word_map:
            end_id = int(word_map[k])
            break
    if start_id is None or end_id is None:
        # try to infer common tokens
        for k, v in word_map.items():
            if str(k).lower().startswith('<start') and start_id is None:
                start_id = int(v)
            if str(k).lower().startswith('<end') and end_id is None:
                end_id = int(v)
    return start_id, end_id


def _preprocess_image(image_path: str, target_size: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)  # (1,3,H,W)


def _ensure_encoder_out_flattened(encoder_out: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    Normalize encoder output to (B, num_pixels, encoder_dim).
    Return (encoder_out_flat, spatial_size)
    spatial_size is H if num_pixels == H*H else -1
    """
    if encoder_out is None:
        raise ValueError('encoder_out is None')
    if encoder_out.dim() == 4:
        # assume (B, H, W, C) or (B, C, H, W) - try detect
        B, a, b, c = encoder_out.size()
        # If channels last (H,W,C), we want to reorder
        # Heuristic: if b is small (<50) and c reasonable, treat as H,W,C
        if c < 64 and a > 1 and b > 1:
            # (B, H, W, C) -> flatten
            B, H, W, C = encoder_out.size()
            flat = encoder_out.view(B, H * W, C)
            return flat, H
        else:
            # treat as (B, C, H, W)
            B, C, H, W = encoder_out.size()
            flat = encoder_out.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
            return flat, H
    elif encoder_out.dim() == 3:
        # (B, num_patches, C)
        B, num_patches, C = encoder_out.size()
        # try to determine spatial size
        sq = int(round(math.sqrt(num_patches)))
        if sq * sq == num_patches:
            return encoder_out, sq
        else:
            return encoder_out, -1
    else:
        raise ValueError('Unsupported encoder_out ndim: %d' % encoder_out.dim())


def beam_search_decode_v2(
    encoder, decoder, image_path: str, word_map: dict, beam_size: int = 3, max_cap_len: int = 50
) -> Tuple[List[int], Optional[np.ndarray]]:
    """
    Beam search that supports ViT encoder output shape.
    Returns (seq, alphas) where alphas shape is (T, H, W) if spatial size known, else None.
    Works when decoder is LSTM + attention (has methods: init_hidden_state, attention, embedding, decode_step, f_beta, sigmoid, fc).
    For transformer decoders, prefer using `decoder.generate` externally.
    """
    k = beam_size
    vocab_size = len(word_map)

    # preprocess image and run encoder
    # determine target size for encoder: prefer encoder.img_size if available
    target_size = (224, 224)
    if hasattr(encoder, 'img_size'):
        try:
            target_size = tuple(getattr(encoder, 'img_size'))
        except Exception:
            pass
    # load and preprocess
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_out = encoder(img_t)
    # normalize encoder_out to (B, num_pixels, encoder_dim)
    encoder_out, spatial = _ensure_encoder_out_flattened(encoder_out)
    B, num_pixels, encoder_dim = encoder_out.size()

    # flatten & expand
    encoder_out_flat = encoder_out.view(1, -1, encoder_dim)
    enc_image_size = spatial if spatial > 0 else int(round(math.sqrt(num_pixels))) if num_pixels > 0 else -1

    encoder_out = encoder_out_flat  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # expand to k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # init k prev words
    start_id, end_id = _find_token_ids(word_map)
    if start_id is None or end_id is None:
        raise RuntimeError('Start or end token not found in word_map')

    k_prev_words = torch.LongTensor([[start_id]] * k).to(device)  # (k,1)
    seqs = k_prev_words  # (k,1)
    top_k_scores = torch.zeros(k, 1).to(device)

    # store alphas
    if enc_image_size > 0:
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)
    else:
        seqs_alpha = torch.ones(k, 1, 1, num_pixels).to(device)

    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    step = 1

    # init hidden state from decoder
    h, c = decoder.init_hidden_state(encoder_out)

    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # awe: (s, encoder_dim), alpha: (s, num_pixels)

        if enc_image_size > 0:
            alpha_map = alpha.view(-1, enc_image_size, enc_image_size)
        else:
            # reshape to (s, 1, num_pixels)
            alpha_map = alpha.view(-1, 1, num_pixels)

        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))

        scores = decoder.fc(h)  # (s, vocab)
        scores = F.log_softmax(scores, dim=1)

        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab)

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        prev_word_inds = (top_k_words // vocab_size)
        next_word_inds = (top_k_words % vocab_size)

        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        # update alphas
        if enc_image_size > 0:
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha_map[prev_word_inds].unsqueeze(1)], dim=1)
        else:
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha_map[prev_word_inds].unsqueeze(1)], dim=1)

        # check complete
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != end_id]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            for ci in complete_inds:
                complete_seqs.append(seqs[ci].tolist())
                # convert alpha storage to numpy (T, H, W) if possible
                alpha_np = seqs_alpha[ci].cpu().numpy()
                complete_seqs_alpha.append(alpha_np)
                complete_seqs_scores.append(top_k_scores[ci])

        k -= len(complete_inds)

        if k == 0:
            break

        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        if step > max_cap_len:
            break
        step += 1

    if len(complete_seqs_scores) == 0:
        # no completed sequences -> take best partial
        seq = seqs[0].tolist()
        alphas = seqs_alpha[0].cpu().numpy()
    else:
        i = int(torch.argmax(torch.stack([s for s in complete_seqs_scores])))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]

    return seq, alphas


def greedy_decode_v2(encoder, decoder, image_path: str, word_map: dict, max_len: int = 50) -> Tuple[List[int], Optional[np.ndarray]]:
    """
    Greedy decode for LSTM+attention decoder (returns seq ids and alphas) or for transformer (if decoder has generate())
    """
    start_id, end_id = _find_token_ids(word_map)
    if start_id is None or end_id is None:
        raise RuntimeError('Start/end tokens not found in word_map')

    # preprocess and run encoder
    target_size = (224, 224)
    if hasattr(encoder, 'img_size'):
        try:
            target_size = tuple(getattr(encoder, 'img_size'))
        except Exception:
            pass
    img_t = _preprocess_image(image_path, target_size).to(device)

    with torch.no_grad():
        encoder_out = encoder(img_t)

    # transformer path
    if hasattr(decoder, 'generate'):
        # use transformer's generate
        enc_flat, spatial = _ensure_encoder_out_flattened(encoder_out)
        with torch.no_grad():
            gen = decoder.generate(enc_flat, start_token_id=start_id, end_token_id=end_id, max_len=max_len)
        seq = gen[0].tolist()
        return seq, None

    # LSTM path
    enc_flat, spatial = _ensure_encoder_out_flattened(encoder_out)
    B, num_pixels, encoder_dim = enc_flat.size()

    # init
    enc = enc_flat
    h, c = decoder.init_hidden_state(enc)
    word = torch.tensor([start_id], dtype=torch.long, device=device).unsqueeze(0)  # (1,1)
    seq = [start_id]
    alphas = []

    for t in range(max_len):
        embeddings = decoder.embedding(word).squeeze(1)  # (1, embed_dim)
        awe, alpha = decoder.attention(enc, h)  # (1, encoder_dim), (1, num_pixels)
        if spatial > 0:
            alpha_map = alpha.view(1, spatial, spatial)
        else:
            alpha_map = alpha.view(1, 1, num_pixels)
        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = decoder.fc(h)
        scores = F.log_softmax(scores, dim=1)
        next_word = scores.argmax(dim=1)
        next_id = int(next_word.item())
        seq.append(next_id)
        alphas.append(alpha_map[0].cpu().numpy())
        word = next_word.unsqueeze(1)
        if next_id == end_id:
            break
    alphas_np = np.stack(alphas, axis=0) if len(alphas) > 0 else None
    return seq, alphas_np


def visualize_att_v2(image_path: str, seq: List[int], alphas: Optional[np.ndarray], rev_word_map: dict, smooth: bool = True):
    """
    Visualize attention similar to original but handles alpha shape (T,H,W) or (T,1,num_pixels).
    """
    image = Image.open(image_path).convert('RGB')

    words = [rev_word_map.get(int(ind), '<unk>') for ind in seq]

    if alphas is None:
        print('No attention to display')
        return

    T = alphas.shape[0]
    # if alphas have shape (T,1,num_pixels) try to reshape to square
    if alphas.ndim == 3 and alphas.shape[1] == 1:
        num_pixels = alphas.shape[2]
        sq = int(round(math.sqrt(num_pixels)))
        if sq * sq == num_pixels:
            alphas = alphas.reshape((T, sq, sq))
        else:
            # fallback: expand each patch to a row
            alphas = alphas.reshape((T, 1, num_pixels))

    plt.figure(figsize=(15, 8))
    for t in range(min(len(words), T)):
        plt.subplot(int(np.ceil(min(len(words), T) / 5)), 5, t + 1)
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if current_alpha.ndim == 2:
            H, W = current_alpha.shape
            # upscale alpha to image size
            if smooth:
                alpha = skimage.transform.pyramid_expand(current_alpha, upscale=max(image.size)//max(H, W), sigma=8)
            else:
                alpha = skimage.transform.resize(current_alpha, [image.size[1], image.size[0]])
            plt.imshow(alpha, alpha=0.8)
        else:
            # 1D alpha - plot as overlay text
            plt.imshow(image)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Convenience wrapper that auto-chooses method
def caption_image_beam_search_v2(encoder, decoder, image_path: str, word_map: dict, beam_size: int = 3, max_cap_len: int = 50):
    # If decoder has generate (transformer), use it
    if hasattr(decoder, 'generate'):
        # preprocess and run encoder
        target_size = (224, 224)
        if hasattr(encoder, 'img_size'):
            try:
                target_size = tuple(getattr(encoder, 'img_size'))
            except Exception:
                pass
        img_t = _preprocess_image(image_path, target_size).to(device)
        with torch.no_grad():
            encoder_out = encoder(img_t)
        enc_flat, spatial = _ensure_encoder_out_flattened(encoder_out)
        start_id, end_id = _find_token_ids(word_map)
        if start_id is None or end_id is None:
            raise RuntimeError('Start/end token not found in word_map')
        with torch.no_grad():
            gen = decoder.generate(enc_flat.to(device), start_token_id=int(start_id), end_token_id=int(end_id), max_len=max_cap_len)
        seq = gen[0].tolist()
        return seq, None
    else:
        return beam_search_decode_v2(encoder, decoder, image_path, word_map, beam_size=beam_size, max_cap_len=max_cap_len)


# If run as script, simple CLI similar to original
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-i', required=True)
    parser.add_argument('--model', '-m', required=True)
    parser.add_argument('--word_map', '-wm', required=True)
    parser.add_argument('--beam_size', '-b', default=5, type=int)
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false')
    args = parser.parse_args()

    ck = torch.load(args.model, map_location=device, weights_only=False)
    decoder = ck.get('decoder', None)
    encoder = ck.get('encoder', None)
    if decoder is None or encoder is None:
        raise RuntimeError('Checkpoint must contain encoder and decoder modules or state_dicts.')

    # load word map
    with open(args.word_map, 'r', encoding='utf-8') as f:
        word_map = json.load(f)
    rev_word_map = {v: k for k, v in word_map.items()}

    seq, alphas = caption_image_beam_search_v2(encoder, decoder, args.img, word_map, beam_size=args.beam_size)
    if alphas is not None:
        alphas = torch.FloatTensor(alphas)
    print('Pred:', seq)
    if alphas is not None:
        visualize_att_v2(args.img, seq, alphas.numpy(), rev_word_map, smooth=args.smooth)

