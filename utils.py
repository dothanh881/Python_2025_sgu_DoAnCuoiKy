import os
import numpy as np
import h5py
import json
import torch
from PIL import Image
from tqdm import tqdm
from collections import Counter, deque
from random import seed, choice, sample

def imread(path):
    """
    Đọc ảnh từ đường dẫn và trả về mảng numpy H x W x C (uint8), ở định dạng RGB.
    """
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    return arr

def imresize(img, size):
    """
    Thay đổi kích thước ảnh về kích thước cho trước và trả về mảng numpy (uint8).
    - `img` có thể là numpy array HxWxC hoặc một PIL Image.
    - `size` được truyền theo (height, width) để tương thích với code gốc.
    """
    if isinstance(img, np.ndarray):
        pil_img = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        pil_img = img
    else:
        pil_img = Image.open(img).convert('RGB')

    # PIL mong đợi kích thước theo (width, height); code gốc truyền (height, width)
    target_size = (int(size[1]), int(size[0]))
    pil_resized = pil_img.resize(target_size, Image.LANCZOS)
    return np.array(pil_resized, dtype=np.uint8)

def _count_jpgs_in_dir(path):
    """Đếm số file .jpg trực tiếp trong thư mục (không đệ quy)."""
    try:
        return sum(1 for e in os.listdir(path) if e.lower().endswith('.jpg'))
    except Exception:
        return 0

def resolve_image_path(image_folder, img_entry, dataset, max_search_depth=2):
    """
    Trình tìm đường dẫn ảnh cẩn thận nhưng giới hạn độ sâu tìm kiếm.

    Chiến lược (nhanh, tránh quét toàn bộ ổ đĩa):
    1) Kiểm tra các đường dẫn ứng viên trực tiếp (image_folder/filename, image_folder/filepath/filename, ...)
    2) Nếu không tìm thấy, thực hiện tìm kiếm theo chiều rộng với độ sâu tối đa `max_search_depth` (mặc định 2).
       Điều này chỉ duyệt thư mục gốc, các thư mục con và các thư mục con cấp 2 (tùy cấu hình), nhanh hơn nhiều so với os.walk toàn bộ.
    3) Thử tìm tên file không phân biệt chữ hoa/chữ thường trong các thư mục đã quét.
    Trả về đường dẫn đầy đủ nếu tìm thấy, ngược lại trả về None.
    """
    filename = img_entry.get('filename', '').strip()
    filepath = (img_entry.get('filepath') or '').strip()

    # Bảo vệ nhanh: nếu `image_folder` chứa nhiều file jpg, kiểm tra các khớp trực tiếp sẽ nhanh
    candidates = []

    # Hành vi chuẩn: bộ dữ liệu Flickr thường chỉ dùng `filename` trực tiếp dưới `image_folder`
    candidates.append(os.path.join(image_folder, filename))

    # Nếu JSON có `filepath` (ví dụ COCO), thử đường dẫn đó
    if dataset == 'coco' and filepath:
        candidates.append(os.path.join(image_folder, filepath, filename))

    # Nếu có `filepath`, thử nối thêm một vài cách phổ biến
    if filepath:
        candidates.append(os.path.join(image_folder, filepath, filename))
        candidates.append(os.path.join(image_folder, os.path.basename(filepath), filename))

    # Thử trường hợp `image_folder` có thư mục con cùng tên (thường xảy ra khi giải nén)
    basename_imgfolder = os.path.basename(image_folder.rstrip(os.sep))
    candidates.append(os.path.join(image_folder, basename_imgfolder, filename))

    # Kiểm tra các ứng viên trực tiếp trước (nhanh)
    for c in candidates:
        if c and os.path.exists(c):
            return c
        # thử loại bỏ khoảng trắng thừa
        if c and os.path.exists(c.strip()):
            return c.strip()

    # Tìm kiếm theo chiều rộng với độ sâu giới hạn - không quét cả ổ đĩa
    # Duyệt các thư mục tối đa `max_search_depth` mức.
    q = deque()
    q.append((image_folder, 0))
    checked_dirs = set()

    while q:
        dirpath, depth = q.popleft()
        if dirpath in checked_dirs:
            continue
        checked_dirs.add(dirpath)

        #  liệt kê các mục; bỏ qua lỗi nếu xảy ra
        try:
            with os.scandir(dirpath) as it:
                for entry in it:
                    try:
                        if entry.is_file():
                            # So sánh trực tiếp hoặc không phân biệt hoa thường
                            if entry.name == filename or entry.name.lower() == filename.lower():
                                return entry.path
                        elif entry.is_dir() and depth < max_search_depth:
                            q.append((entry.path, depth + 1))
                    except PermissionError:
                        continue
                    except FileNotFoundError:
                        continue
        except PermissionError:
            continue
        except FileNotFoundError:
            continue
        except NotADirectoryError:
            continue

    # Không tìm thấy trong phạm vi giới hạn
    return None


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Tạo các file input cho tập huấn luyện, validation và test.

    :param dataset: tên bộ dữ liệu, một trong 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: đường dẫn tới file JSON Karpathy có splits và captions
    :param image_folder: thư mục chứa ảnh đã tải xuống
    :param captions_per_image: số caption lấy mẫu cho mỗi ảnh
    :param min_word_freq: từ xuất hiện ít hơn ngưỡng này sẽ được gán <unk>
    :param output_folder: thư mục lưu các file đầu ra
    :param max_len: không lấy caption dài hơn độ dài này
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Đọc JSON Karpathy
    with open(karpathy_json_path, 'r', encoding='utf-8') as j:
        data = json.load(j)

    # Danh sách đường dẫn ảnh và caption cho từng split
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    missing_images = 0
    total_images = 0

    for img in data['images']:
        total_images += 1
        captions = []
        for c in img['sentences']:
            # Cập nhật tần suất từ
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        # Giải quyết đường dẫn ảnh một cách cẩn thận (tìm giới hạn độ sâu)
        path = resolve_image_path(image_folder, img, dataset, max_search_depth=2)

        if path is None:
            missing_images += 1
            # bỏ qua ảnh này nhưng in cảnh báo
            print("Warning: image file not found for JSON entry. filename='{}', filepath='{}'".format(
                img.get('filename'), img.get('filepath')))
            continue

        # Thêm vào danh sách tương ứng theo split
        if img.get('split') in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img.get('split') in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img.get('split') in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    print("Total images in JSON:", total_images)
    print("Images missing on disk (skipped):", missing_images)
    print("Train images:", len(train_image_paths), "Val images:", len(val_image_paths), "Test images:", len(test_image_paths))

    # Kiểm tra nhất quán (số lượng đường dẫn và captions phải khớp)
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Tạo bản đồ từ (word map)
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Tạo tên cơ sở cho các file đầu ra
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Lưu word map ra JSON
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w', encoding='utf-8') as j:
        json.dump(word_map, j)

    # Lấy mẫu captions cho mỗi ảnh, lưu ảnh vào file HDF5, và captions cùng độ dài vào file JSON
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:
        # Đường dẫn file HDF5
        file_path = os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5')

        # Mở file ở chế độ 'r+' nếu đã tồn tại (để cập nhật); tạo mới 'w' nếu chưa có
        mode = 'r+' if os.path.exists(file_path) else 'w'
        with h5py.File(file_path, mode) as h:
            # Ghi số captions lấy mẫu cho mỗi ảnh vào thuộc tính
            h.attrs['captions_per_image'] = captions_per_image

            # Nếu dataset 'images' đã tồn tại từ lần chạy trước, xóa đi để tạo lại với shape đúng
            if 'images' in h:
                del h['images']

            # Tạo dataset để lưu ảnh
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):
                # Lấy mẫu captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Kiểm tra độ dài danh sách captions
                assert len(captions) == captions_per_image

                # Đọc ảnh
                img = imread(path)
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Lưu ảnh vào HDF5
                images[i] = img

                for j, c in enumerate(captions):
                    # Mã hóa caption
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Tính độ dài caption (sau khi thêm <start> và <end>)
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Kiểm tra nhất quán
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Lưu captions đã mã hóa và độ dài của chúng ra file JSON
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w', encoding='utf-8') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w', encoding='utf-8') as j:
                json.dump(caplens, j)


def init_embedding(embeddings):
    """
    Khởi tạo tensor embedding với giá trị lấy từ phân phối đều.

    :param embeddings: tensor embedding
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Tạo tensor embedding cho word map đã cho, để tải vào mô hình.
    """
    with open(emb_file, 'r', encoding='utf-8') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    print("\nLoading embeddings...")
    for line in open(emb_file, 'r', encoding='utf-8'):
        line = line.split(' ')
        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
        if emb_word not in vocab:
            continue
        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Cắt gradient trong quá trình lan truyền ngược để tránh gradient bùng nổ.
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'BEST_' + filename)

class AverageMeter(object):
    """
         giá trị hiện tại, trung bình, tổng và số lượng của một metric.
    """
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Giảm learning rate theo hệ số chỉ định.
    """
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def accuracy(scores, targets, k):
    """
    Tính độ chính xác top-k từ các score dự đoán và nhãn thật.
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)