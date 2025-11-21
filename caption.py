import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Đọc ảnh và sinh caption bằng Beam Search.

    :param encoder: mô hình encoder
    :param decoder: mô hình decoder
    :param image_path: đường dẫn ảnh
    :param word_map: ánh xạ từ -> chỉ mục
    :param beam_size: kích thước beam
    :return: câu mô tả (danh sách chỉ số), ma trận trọng số attention
    """

    k = beam_size
    vocab_size = len(word_map)

    # Đọc ảnh và tiền xử lý
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img = np.array(img)
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Làm phẳng không gian ảnh
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # Nhân bản để giả lập batch size = k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor lưu từ trước đó (khởi tạo với <start>)
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor lưu các sequence hiện tại (khởi tạo chỉ có <start>)
    seqs = k_prev_words  # (k, 1)

    # Điểm hiện tại của các sequence (khởi tạo = 0)
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor lưu alpha (attention) cho mỗi sequence
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Danh sách lưu sequence hoàn chỉnh, alpha và điểm
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Bắt đầu giải mã
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # Vòng lặp beam search (loại sequence khi gặp <end>)
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # cổng (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Cộng điểm trước đó vào
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # Lấy top k ứng viên
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (k)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (k)

        # Chuyển chỉ số unrolled về chỉ số seq và từ tiếp theo
        prev_word_inds = (top_k_words // vocab_size)
        next_word_inds = (top_k_words % vocab_size)

        # Cập nhật seqs và seqs_alpha
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Tách các sequence chưa kết thúc và đã kết thúc
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Lưu các sequence hoàn chỉnh
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # giảm beam size

        # Nếu không còn sequence đang tiến hành thì dừng
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        #  validate tránh quá dài
        if step > 50:
            break
        step += 1

    # Chọn sequence có điểm cao nhất
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    # Hiển thị ảnh và overlay attention cho từng từ
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    plt.figure(figsize=(15, 8))

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(int(np.ceil(len(words) / 5)), 5, t + 1)
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.savefig('attention_result.png')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hiển thị mô hình Show, Attend and Tell - sinh caption')

    parser.add_argument('--img', '-i', help='đường dẫn ảnh')
    parser.add_argument('--model', '-m', help='đường dẫn file checkpoint mô hình')
    parser.add_argument('--word_map', '-wm', help='đường dẫn file word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='kích thước beam cho beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='không làm mịn overlay alpha')

    args = parser.parse_args()

    # Tải mô hình
    checkpoint = torch.load(args.model, map_location=str(device), weights_only=False)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Tải word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode với attention và beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    # Hiển thị caption và attention cho sequence tốt nhất
    visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)
