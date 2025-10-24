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
    Read image from path and return numpy array H x W x C (uint8), RGB.
    """
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    return arr

def imresize(img, size):
    """
    Resize image to given size and return numpy array (uint8).
    - img can be a numpy array HxWxC or a PIL Image.
    - size should be (height, width) to keep compatibility with original code.
    """
    if isinstance(img, np.ndarray):
        pil_img = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        pil_img = img
    else:
        pil_img = Image.open(img).convert('RGB')

    # PIL expects size as (width, height); original code passes (height, width)
    target_size = (int(size[1]), int(size[0]))
    pil_resized = pil_img.resize(target_size, Image.LANCZOS)
    return np.array(pil_resized, dtype=np.uint8)

def _count_jpgs_in_dir(path):
    """Count jpg files directly inside path (non-recursive)."""
    try:
        return sum(1 for e in os.listdir(path) if e.lower().endswith('.jpg'))
    except Exception:
        return 0

def resolve_image_path(image_folder, img_entry, dataset, max_search_depth=2):
    """
    Robust but limited-depth image path resolver.

    Strategy (fast, avoids full os.walk on Drive):
    1) Check expected direct candidate paths (image_folder/filename, image_folder/filepath/filename, etc.)
    2) If not found, do a limited breadth-first search up to max_search_depth (default 2).
       This explores image_folder, its subdirs, and sub-subdirs only (configurable), much faster than full walk.
    3) Try a case-insensitive direct-name search in the scanned dirs.
    Returns full path if found, otherwise None.
    """
    filename = img_entry.get('filename', '').strip()
    filepath = (img_entry.get('filepath') or '').strip()

    # Quick guard: if image_folder directly contains many jpgs, checking only direct matches is fast
    # Candidate direct paths:
    candidates = []

    # Standard behavior: Flickr datasets typically use just filename under image_folder
    candidates.append(os.path.join(image_folder, filename))

    # If JSON has filepath (COCO), try that too
    if dataset == 'coco' and filepath:
        candidates.append(os.path.join(image_folder, filepath, filename))

    # Try joining filepath if given
    if filepath:
        candidates.append(os.path.join(image_folder, filepath, filename))
        candidates.append(os.path.join(image_folder, os.path.basename(filepath), filename))

    # Try if image_folder contains nested folder with same basename (common when unzipped)
    basename_imgfolder = os.path.basename(image_folder.rstrip(os.sep))
    candidates.append(os.path.join(image_folder, basename_imgfolder, filename))

    # Check direct candidates first (fast)
    for c in candidates:
        if c and os.path.exists(c):
            return c
        # also try stripping possible extra whitespace/newlines in filename
        if c and os.path.exists(c.strip()):
            return c.strip()

    # Limited depth BFS search (breadth-first) - do not search entire Drive
    # We explore directories up to max_search_depth deep.
    # This is intentionally conservative to avoid long Drive scans.
    q = deque()
    q.append((image_folder, 0))
    checked_dirs = set()

    while q:
        dirpath, depth = q.popleft()
        if dirpath in checked_dirs:
            continue
        checked_dirs.add(dirpath)

        # Try to list entries; continue on errors
        try:
            with os.scandir(dirpath) as it:
                for entry in it:
                    try:
                        if entry.is_file():
                            # direct match or case-insensitive match
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

    # Not found in limited search
    return None

def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r', encoding='utf-8') as j:
        data = json.load(j)

    # Read image paths and captions for each image
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
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        # Resolve actual path robustly (limited search)
        path = resolve_image_path(image_folder, img, dataset, max_search_depth=2)

        if path is None:
            missing_images += 1
            # skip this image but warn
            print("Warning: image file not found for JSON entry. filename='{}', filepath='{}'".format(
                img.get('filename'), img.get('filepath')))
            continue

        # Append to the appropriate split lists
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

    # Sanity check (lengths should match)
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w', encoding='utf-8') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:
        # Prepare HDF5 file path
        file_path = os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5')

        # Open file in r+ if exists (so we can update); create with 'w' if it does not exist
        mode = 'r+' if os.path.exists(file_path) else 'w'
        with h5py.File(file_path, mode) as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # If dataset 'images' already exists (from a previous run), delete it to recreate with correct shape
            if 'images' in h:
                del h['images']

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):
                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = imread(path)
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w', encoding='utf-8') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w', encoding='utf-8') as j:
                json.dump(caplens, j)

def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)

def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.
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
    Clips gradients computed during backpropagation to avoid explosion of gradients.
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
    Keeps track of most recent, average, sum, and count of a metric.
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
    Shrinks learning rate by a specified factor.
    """
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)