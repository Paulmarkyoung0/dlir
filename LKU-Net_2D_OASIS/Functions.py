"""
Helper functions from https://github.com/zhangjun001/ICNet.

Some functions has been modified.
"""

import numpy as np
import torch.utils.data as Data
from pathlib import Path
from skimage import io
from torchvision.transforms.functional import to_tensor

def load_images(image_dir, supervision):
    hist_path = image_dir / 'hist.png'
    hist_img = to_tensor(io.imread(hist_path))
    oct_path = image_dir / 'oct.png'
    oct_img = to_tensor(io.imread(oct_path, as_gray=True)).unsqueeze(0)
    if supervision == 'dice':
        hist_seg_path = image_dir / 'hist_seg.png'
        hist_seg = to_tensor(io.imread(hist_seg_path, as_gray=True)).unsqueeze(0)
        oct_seg_path = image_dir / 'oct_seg.png'
        oct_seg = to_tensor(io.imread(oct_seg_path, as_gray=True)).unsqueeze(0)
        return hist_img, oct_img, hist_seg, oct_seg
    elif supervision == 'truth':
        truth_path = image_dir / 'truth.png'
        truth_img = to_tensor(io.imread(truth_path))
        return hist_img, oct_img, truth_img
    else:
        return hist_img, oct_img

class TrainDataset(Data.Dataset):
    def __init__(self, data_path, img_file=None, supervision='dice'):
        super().__init__()
        self.data_path = Path(data_path)
        self.names = np.loadtxt(self.data_path / img_file, dtype='str')
        self.supervision = supervision
        
    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        image_dir = self.data_path / self.names[index]
        return load_images(image_dir, self.supervision)


class ValidationDataset(Data.Dataset):
    def __init__(self, data_path, img_file=None, supervision=None):
        super().__init__()
        self.data_path = Path(data_path)
        self.names = np.loadtxt(self.data_path / img_file, dtype='str')
        self.supervision = supervision
    def __len__(self):
        return len(self.names)
    def __getitem__(self, index):
        image_dir = self.data_path / self.names[index]
        return load_images(image_dir, self.supervision)
    