import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CaptionDataset(Dataset):
    """
    Lớp Dataset của PyTorch để tạo batch cho DataLoader.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: thư mục chứa các file dữ liệu đã xử lý
        :param data_name: tên cơ sở của các file đầu vào
        :param split: một trong 'TRAIN', 'VAL', 'TEST'
        :param transform: pipeline transform cho ảnh
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Mở file hdf5 chứa ảnh
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Số captions mỗi ảnh
        self.cpi = self.h.attrs['captions_per_image']

        # Nạp captions đã mã hóa vào bộ nhớ
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Nạp độ dài mỗi caption
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Transform cho ảnh (chẳng hạn normalize)
        self.transform = transform

        # Tổng số datapoints (mỗi caption là một datapoint)
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Lưu ý: caption thứ N tương ứng với ảnh số (N // captions_per_image)
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        # Nếu đang ở split TRAIN, trả về (img, caption, caplen)
        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # Với VAL/TEST, trả thêm tất cả captions của ảnh để tính BLEU
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
