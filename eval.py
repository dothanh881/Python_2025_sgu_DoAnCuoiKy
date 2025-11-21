import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

# Tham số (cập nhật theo đường dẫn của bạn)
data_folder = '/content/drive/MyDrive/Image_captioning_flickr8k/dataset/flickr8k_processed'
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'
checkpoint = '/content/drive/MyDrive/Image_captioning_flickr8k/a-PyTorch-Tutorial-to-Image-Captioning/BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'
word_map_file = '/content/drive/MyDrive/Image_captioning_flickr8k/dataset/flickr8k_processed/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # thiết bị cho mô hình và tensor
cudnn.benchmark = True  # bật nếu kích thước input cố định để tăng hiệu năng

# Tải mô hình đã lưu
checkpoint = torch.load(checkpoint, weights_only=False)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Tải từ điển từ (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Transform chuẩn hóa ảnh
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Hàm đánh giá mô hình bằng BLEU-4 sử dụng Beam Search.

    :param beam_size: kích thước beam khi sinh caption
    :return: BLEU-4 score
    """
    # DataLoader cho bộ TEST (batch_size=1 để beam search đơn ảnh)
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # Lưu references (caption thật) và hypotheses (dự đoán)
    references = list()
    hypotheses = list()

    # Duyệt từng ảnh
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="ĐÁNH GIÁ VỚI BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Chuyển ảnh lên thiết bị (GPU nếu có)
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode ảnh
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten không gian ảnh thành danh sách pixel
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Nhân bản encoding để coi như batch size = k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensors lưu từ trước đó (khởi tạo bằng <start>)
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensors lưu các sequence hiện tại (khởi tạo chỉ có <start>)
        seqs = k_prev_words  # (k, 1)

        # Scores hiện tại của k sequence (ban đầu = 0)
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Danh sách lưu các sequence hoàn chỉnh và điểm của chúng
        complete_seqs = list()
        complete_seqs_scores = list()

        # Bắt đầu giải mã
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # Vòng lặp beam search (k giảm dần khi có sequence hoàn chỉnh)
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # attention-weighted encoding (s, encoder_dim)

            gate = decoder.sigmoid(decoder.f_beta(h))  # cổng để điều chỉnh encoding
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Cộng điểm bước trước với điểm hiện tại
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # Lấy top k
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (k)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (k)

            # Chuyển chỉ số đã unroll về chỉ số sequence và từ tiếp theo
            prev_word_inds = (top_k_words // vocab_size).long()
            next_word_inds = (top_k_words % vocab_size).long()

            # Cập nhật seqs với từ mới
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Tìm sequence chưa kết thúc (chưa gặp <end>)
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Lưu các sequence hoàn chỉnh
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # giảm beam size

            # Nếu không còn sequence chưa hoàn chỉnh thì dừng
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Ngắt nếu quá nhiều bước (bảo vệ)
            if step > 50:
                break
            step += 1

        # Chọn sequence có điểm cao nhất
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References: lấy caption thật, loại bỏ tokens đặc biệt
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))
        references.append(img_captions)

        # Hypotheses: sequence dự đoán, loại bỏ tokens đặc biệt
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

    # Tính BLEU-4
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4


if __name__ == '__main__':
    beam_size = 3  # thử 3 hoặc 5
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))