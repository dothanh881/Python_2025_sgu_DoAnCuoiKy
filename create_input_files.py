#!/usr/bin/env python3
"""
Run create_input_files for Flickr8k - edited to use Flickr8k JSON and Drive paths.
Chỉnh DATA_DIR cho đúng đường dẫn của bạn trước khi chạy trên Colab.
"""
import os

# ĐIỀU CHỈNH: đặt DATA_DIR tới thư mục trên Drive nơi bạn lưu dataset
DATA_DIR = '/content/drive/MyDrive/Image_captioning_flickr8k/dataset'  # <- sửa đúng đường dẫn của bạn

# Tên file/folder bên trong DATA_DIR (chỉnh nếu khác)
KARPATHY_JSON = os.path.join(DATA_DIR, 'dataset_flickr8k.json')      # file JSON bạn đã tạo
IMAGE_FOLDER = os.path.join(DATA_DIR, 'Flickr8k_Dataset')           # folder chứa ảnh (flat)
OUTPUT_FOLDER = os.path.join(DATA_DIR, 'flickr8k_processed')       # nơi lưu HDF5 + WORDMAP + captions

# Tham số create_input_files
CAPTIONS_PER_IMAGE = 5
MIN_WORD_FREQ = 5
MAX_LEN = 50

# Không cần sửa phần dưới nếu bạn đã mount Drive và đã pip install dependencies
from utils import create_input_files

if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print("Karpathy JSON:", KARPATHY_JSON)
    print("Image folder:", IMAGE_FOLDER)
    print("Output folder:", OUTPUT_FOLDER)

    create_input_files(dataset='flickr8k',
                       karpathy_json_path=KARPATHY_JSON,
                       image_folder=IMAGE_FOLDER,
                       captions_per_image=CAPTIONS_PER_IMAGE,
                       min_word_freq=MIN_WORD_FREQ,
                       output_folder=OUTPUT_FOLDER,
                       max_len=MAX_LEN)

    print("create_input_files finished. Check:", OUTPUT_FOLDER)
