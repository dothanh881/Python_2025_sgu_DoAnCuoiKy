import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

# Tham số dữ liệu
data_folder = '/content/drive/MyDrive/Image_captioning_flickr8k/dataset/flickr8k_processed'
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq' # phải trùng base_filename tạo ra ở create_input_files
# Tham số mô hình
emb_dim = 512  # kích thước embedding từ
attention_dim = 512  # kích thước các lớp attention
decoder_dim = 512  # kích thước RNN của decoder
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # thiết bị cho mô hình và tensor (GPU nếu có)
cudnn.benchmark = True  # bật nếu kích thước input cố định để tối ưu hiệu năng

# Tham số huấn luyện
start_epoch = 0
# epochs là tổng số epoch mong muốn (mặc định)
epochs = 10  # số epoch để train (tăng nếu muốn train lâu hơn)
# Nếu muốn khi resume từ checkpoint train thêm N epoch so với epoch đã lưu, đặt thêm tham số này.
# Ví dụ: additional_epochs = 5 sẽ làm epochs = (checkpoint_epoch + 5) khi resume từ checkpoint.
additional_epochs = None  # đặt thành số nguyên (ví dụ 5) để train thêm N epoch khi resume
epochs_since_improvement = 0  # số epoch kể từ lần cải thiện BLEU gần nhất
batch_size = 32
workers = 1  # số worker cho data-loading; hiện chỉ 1 chạy tốt với h5py
encoder_lr = 1e-4  # learning rate cho encoder khi fine-tune
decoder_lr = 4e-4  # learning rate cho decoder
grad_clip = 5.  # giá trị để cắt gradient
alpha_c = 1.  # hệ số regularization cho 'doubly stochastic attention'
best_bleu4 = 0.  # BLEU-4 tốt nhất hiện tại
print_freq = 100  # in thông tin huấn luyện/validation mỗi __ batch
fine_tune_encoder = False  # có fine-tune encoder không?

# Thiết lập tham số resume/checkpoint bằng argparse để dễ dùng trên Colab.
import argparse
parser = argparse.ArgumentParser(description='Train model or resume from checkpoint')
parser.add_argument('--resume', choices=['best', 'latest', 'none', 'path'], default='best',
                    help="Chọn checkpoint để resume: 'best' (BEST_checkpoint), 'latest' (checkpoint_), 'none' (train from scratch), or 'path' (kết hợp với --checkpoint)")
parser.add_argument('--checkpoint', type=str, default=None, help='Đường dẫn tới file checkpoint (dùng khi --resume path)')
parser.add_argument('--additional_epochs', type=int, default=None, help='Số epoch muốn train thêm khi resume từ checkpoint')
args, unknown = parser.parse_known_args()

# Ánh xạ tham số từ args
if args.additional_epochs is not None:
    additional_epochs = args.additional_epochs

# Xác định đường dẫn checkpoint dựa theo lựa chọn resume
def _checkpoint_path_for(mode):
    # mode: 'best'|'latest'|'none'|'path'
    # Tạo danh sách vị trí có thể chứa checkpoint để kiểm tra: data_folder và project root
    project_root = os.path.dirname(os.path.abspath(__file__))

    best_filename = 'BEST_' + 'checkpoint_' + data_name + '.pth.tar'
    latest_filename = 'checkpoint_' + data_name + '.pth.tar'

    candidates = []
    # data_folder locations
    candidates.append(os.path.join(data_folder, latest_filename))
    candidates.append(os.path.join(data_folder, best_filename))
    # project root locations
    candidates.append(os.path.join(project_root, latest_filename))
    candidates.append(os.path.join(project_root, best_filename))
    # cwd locations
    candidates.append(os.path.join(os.getcwd(), latest_filename))
    candidates.append(os.path.join(os.getcwd(), best_filename))

    # For debugging: print candidates that will be checked
    print(f"[DEBUG] Looking for checkpoint (mode={mode}). Checking these candidate paths:")
    for p in candidates:
        exists = os.path.exists(p)
        print(f"  - {p} -> {'FOUND' if exists else 'missing'}")

    if mode == 'none':
        return None
    if mode == 'path':
        return args.checkpoint
    if mode == 'latest':
        # trả về first existing latest; nếu không có, fallback first existing best; else None
        for p in candidates:
            if os.path.basename(p) == latest_filename and os.path.exists(p):
                print(f"[DEBUG] Selected latest: {p}")
                return p
        for p in candidates:
            if os.path.basename(p) == best_filename and os.path.exists(p):
                print(f"[DEBUG] Fallback to best: {p}")
                return p
        print("[DEBUG] No latest or best found among candidates.")
        return None
    # default 'best' -> tìm best trước, sau đó latest
    for p in candidates:
        if os.path.basename(p) == best_filename and os.path.exists(p):
            print(f"[DEBUG] Selected best: {p}")
            return p
    for p in candidates:
        if os.path.basename(p) == latest_filename and os.path.exists(p):
            print(f"[DEBUG] Selected latest (as fallback): {p}")
            return p
    print("[DEBUG] No best or latest found among candidates.")
    return None

# Thiết lập checkpoint_path theo lựa chọn
checkpoint_path = _checkpoint_path_for(args.resume)
if checkpoint_path:
    if os.path.exists(checkpoint_path):
        checkpoint = checkpoint_path
        print(f"Found checkpoint: {checkpoint_path}")
    else:
        checkpoint = None
        print(f"Requested checkpoint not found: {checkpoint_path}. Starting from scratch or set --resume appropriately.")
else:
    checkpoint = None
    print("No checkpoint requested: starting fresh training...")


def main():
    """
    Huấn luyện và đánh giá mô hình.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, epochs

    # Đọc word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Khởi tạo hoặc tải checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        print(f"Loading checkpoint from: {checkpoint}")
        checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
        start_epoch = checkpoint_data['epoch'] + 1
        epochs_since_improvement = checkpoint_data['epochs_since_improvement']
        best_bleu4 = checkpoint_data['bleu-4']
        decoder = checkpoint_data['decoder']
        decoder_optimizer = checkpoint_data['decoder_optimizer']
        encoder = checkpoint_data['encoder']
        encoder_optimizer = checkpoint_data['encoder_optimizer']
        print(f"Loaded checkpoint from epoch {checkpoint_data['epoch']}")
        print(f"Resuming training from epoch {start_epoch}")
        print(f"Best BLEU-4 so far: {best_bleu4}")

        # Nếu muốn train thêm một số epoch so với checkpoint đã load, điều chỉnh biến `epochs`.
        # Ví dụ: nếu checkpoint_epoch = 14 và additional_epochs = 5 thì epochs sẽ được đặt thành 20
        if additional_epochs is not None:
            try:
                additional_epochs_int = int(additional_epochs)
                # epochs là giá trị *một quá* so với epoch cuối cùng (vì range(start_epoch, epochs) chạy tới epochs-1)
                # Để train thêm N epoch (ví dụ từ 15..19 nếu checkpoint_epoch=14 và N=5),
                # ta cần epochs = checkpoint_epoch + N + 1
                epochs = checkpoint_data['epoch'] + additional_epochs_int + 1
                print(f"Adjusted total epochs to {epochs} (will train epochs {start_epoch}..{epochs-1})")
            except Exception:
                print("Warning: additional_epochs không phải số nguyên, bỏ qua việc điều chỉnh epochs.")

        # Bảo vệ: nếu user không cung cấp additional_epochs và biến epochs mặc định nhỏ hơn hoặc bằng
        # start_epoch (ví dụ mặc định epochs=10 nhưng checkpoint ở epoch 14), ta cần điều chỉnh
        # để tránh range(start_epoch, epochs) rỗng (không train gì hết).
        if epochs <= start_epoch:
            if additional_epochs is not None:
                # đã cố gắng set epochs ở trên; nếu vẫn không hợp lệ thì đặt epochs = start_epoch + additional_epochs_int
                try:
                    epochs = start_epoch + max(1, int(additional_epochs))
                    print(f"Note: epochs adjusted to {epochs} based on additional_epochs to allow training to continue.")
                except Exception:
                    epochs = start_epoch + 1
                    print(f"Note: epochs adjusted to {epochs} to ensure at least one epoch of training.")
            else:
                # Không có additional_epochs, đặt ít nhất train 1 epoch tiếp
                epochs = start_epoch + 1
                print(f"Note: epochs was <= start_epoch, adjusted to {epochs} to ensure at least one epoch of training.")

        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Chuyển mô hình lên GPU nếu có
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Hàm mất mát
    criterion = nn.CrossEntropyLoss().to(device)

    # DataLoader tùy chỉnh
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Bắt đầu vòng epoch
    print(f"\n=== STARTING TRAINING ===")
    print(f"Training from epoch {start_epoch} to {epochs-1}")
    print(f"Total epochs to train: {epochs - start_epoch}")
    print("=" * 50)
    
    for epoch in range(start_epoch, epochs):
        print(f"\n>>> EPOCH {epoch}/{epochs-1} <<<")

        # Giảm lr nếu không cải thiện trong 8 epoch liên tiếp, và dừng sau 20 epoch không cải thiện
        if epochs_since_improvement == 20:
            print("Early stopping: No improvement for 20 epochs")
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # Huấn luyện 1 epoch
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # Validation 1 epoch
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # Kiểm tra xem có cải thiện không
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Lưu checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Thực hiện huấn luyện cho một epoch.

    :param train_loader: DataLoader cho dữ liệu huấn luyện
    :param encoder: mô hình encoder
    :param decoder: mô hình decoder
    :param criterion: hàm mất mát
    :param encoder_optimizer: optimizer cho encoder (nếu fine-tune)
    :param decoder_optimizer: optimizer cho decoder
    :param epoch: số epoch hiện tại
    """

    decoder.train()  # chế độ train (dropout và batchnorm hoạt động)
    encoder.train()

    batch_time = AverageMeter()  # thời gian forward + backward
    data_time = AverageMeter()  # thời gian load data
    losses = AverageMeter()  # mất mát (trên mỗi từ)
    top5accs = AverageMeter()  # độ chính xác top-5

    start = time.time()

    # Lặp qua các batch
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Chuyển lên thiết bị nếu có
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Vì ta bắt đầu decode từ <start>, nên target là các từ sau <start>
        targets = caps_sorted[:, 1:]

        # Loại bỏ timestep không decode hoặc là pad
        # Dùng pack_padded_sequence để xử lý nhanh
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Tính loss
        loss = criterion(scores, targets)

        # Thêm regularization cho attention
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Backward
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Cắt gradient
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Cập nhật tham số
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Cập nhật metric
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # In trạng thái
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    Thực hiện validation cho một epoch.

    :param val_loader: DataLoader cho validation
    :param encoder: mô hình encoder
    :param decoder: mô hình decoder
    :param criterion: hàm mất mát
    :return: BLEU-4 score
    """
    decoder.eval()  # chế độ eval (không dùng dropout/batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # danh sách references (caption thật) để tính BLEU-4
    hypotheses = list()  # danh sách dự đoán

    # Tắt gradient để tiết kiệm bộ nhớ GPU
    with torch.no_grad():
        # Lặp qua batch
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Chuyển lên thiết bị
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Targets là các từ sau <start>
            targets = caps_sorted[:, 1:]

            # Loại bỏ timestep không decode hoặc là pad
            scores_copy = scores.clone()
            packed = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            scores = packed.data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            # Tính loss
            loss = criterion(scores, targets)

            # Thêm regularization cho attention
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Cập nhật metric
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Lưu references (caption thật) và hypotheses (dự đoán)
            # Cấu trúc: references = [[ref1a, ref1b...], [ref2a, ...], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind.cpu()]  # do images được sắp xếp trong decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # loại bỏ <start> và pad
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # bỏ pad
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Tính BLEU-4
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()