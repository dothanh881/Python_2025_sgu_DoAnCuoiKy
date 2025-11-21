import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Bộ mã hóa (encoder) trích xuất đặc trưng ảnh bằng ResNet.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # ResNet-101 đã được huấn luyện trước trên ImageNet
        resnet = torchvision.models.resnet101(pretrained=True)

        # Bỏ đi các lớp phân loại cuối (chúng ta chỉ cần đặc trưng)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Adaptive pooling để chuẩn hóa kích thước không phụ thuộc hình ảnh đầu vào
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Lan truyền tiến của encoder.

        :param images: tensor ảnh kích thước (batch_size, 3, image_size, image_size)
        : tensor đặc trưng đã mã hóa
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, enc_image_size, enc_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, enc_image_size, enc_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Bật/tắt tính gradient cho một số block convolution của ResNet khi fine-tune.

        :param fine_tune: True để cho phép fine-tune, False để đóng
        """
        # Mặc định tắt gradient cho toàn bộ ResNet
        for p in self.resnet.parameters():
            p.requires_grad = False
        # Nếu fine-tune, chỉ mở gradient cho các block convolution cuối (blocks 2-4)
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Mạng attention tính trọng số attention trên các vùng ảnh.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: kích thước đặc trưng của encoder
        :param decoder_dim: kích thước ẩn của decoder
        :param attention_dim: kích thước mạng attention
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # dự đoán từ đặc trưng ảnh
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # dự đoán từ trạng thái decoder
        self.full_att = nn.Linear(attention_dim, 1)  # tính điểm attention
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax trên số pixel

    def forward(self, encoder_out, decoder_hidden):
        """
        :param encoder_out: (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: (batch_size, decoder_dim)
        :return: attention-weighted encoding (batch_size, encoder_dim), alpha (batch_size, num_pixels)
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder kết hợp attention để sinh caption từ đặc trưng ảnh.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        Khởi tạo decoder.
        :param attention_dim: kích thước mạng attention
        :param embed_dim: kích thước embedding từ
        :param decoder_dim: kích thước ẩn của RNN decoder
        :param vocab_size: kích thước từ vựng
        :param encoder_dim: kích thước đặc trưng encoder (mặc định 2048)
        :param dropout: tỉ lệ dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # mạng attention

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # lớp embedding từ
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # LSTMCell giải mã
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # khởi tạo hidden state từ đặc trưng ảnh
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # khởi tạo cell state
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # cổng sigmoid tinh chỉnh encoding
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # chuyển state -> điểm cho mỗi từ
        self.init_weights()  # khởi tạo trọng số

    def init_weights(self):
        """
        Khởi tạo một số tham số bằng phân phối đều để hội tụ tốt hơn.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Nạp embedding tiền huấn luyện vào lớp embedding.
        :param embeddings: tensor embedding đã có
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Cho phép/khóa việc fine-tune embeddings (nên tắt nếu dùng embedding cố định).
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Tạo trạng thái ẩn and cell ban đầu cho LSTM từ đặc trưng ảnh.
        :param encoder_out: (batch_size, num_pixels, encoder_dim)
        :return: h, c
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Lan truyền tiến của decoder.

        :param encoder_out: (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: (batch_size, max_caption_length)
        :param caption_lengths: (batch_size, 1)
        :return: predictions, encoded_captions_sorted, decode_lengths, alphas, sort_ind
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Làm phẳng không gian không gian ảnh để thành danh sách pixel
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sắp xếp theo độ dài caption giảm dần (cần cho pack_padded_sequence)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding các từ
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Khởi tạo trạng thái ẩn của LSTM
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # Không decode ở vị trí <end>, nên độ dài decode = độ dài caption thực - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Tạo tensor chứa kết quả dự đoán và attention weights
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # Từng bước time-step giải mã
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # cổng (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
