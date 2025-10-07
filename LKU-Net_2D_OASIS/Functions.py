"""
Helper functions from https://github.com/zhangjun001/ICNet.

Some functions has been modified.
"""

import numpy as np
import torch.utils.data as Data
from pathlib import Path
from skimage import io
from torchvision.transforms.functional import to_tensor
from torch.nn.functional import pad

def load_images(image_dir, supervision):
    """
    Load and process histology and OCT images with optional segmentation masks.
    
    Args:
        image_dir: Path to directory containing image files
        supervision: Type of supervision ('dice', 'truth', or None)
    
    Returns:
        Tuple of tensors depending on supervision type
    """
    hist_path = image_dir / 'hist.tiff'
    hist_img = to_tensor(io.imread(hist_path))
    oct_path = image_dir / 'oct.tiff'
    oct_img = to_tensor(io.imread(oct_path, as_gray=True))
    oct_dims = oct_img.shape[1:]
    
    hist_dims = hist_img.shape[1:]
    pad_list = []
    for i in range(1, -1, -1):
        diff = oct_dims[i] - hist_dims[i]
        pad_left = diff // 2
        pad_right = diff - pad_left
        pad_list.extend([pad_left, pad_right])
    hist_img = pad(hist_img, pad_list, mode='constant', value=0)
    if supervision == 'dice':
        hist_seg_path = image_dir / 'hist_seg.tiff'
        hist_seg = to_tensor(io.imread(hist_seg_path, as_gray=True))
        oct_seg_path = image_dir / 'oct_seg.tiff'
        oct_seg = to_tensor(io.imread(oct_seg_path, as_gray=True))
        return hist_img, oct_img, hist_seg, oct_seg
    elif supervision == 'truth':
        truth_path = image_dir / 'truth.tiff'
        truth_img = to_tensor(io.imread(truth_path))
        return hist_img, oct_img, truth_img
    else:
        return hist_img, oct_img

class TrainDataset(Data.Dataset):
    def __init__(self, data_path, img_file=None, supervision='dice'):
        super().__init__()
        self.data_path = Path(data_path)
        self.names = np.loadtxt(img_file, dtype='str')
        self.supervision = supervision
        
    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        image_dir = self.data_path / 'train' / self.names[index]
        return load_images(image_dir, self.supervision)


class ValidationDataset(Data.Dataset):
    def __init__(self, data_path, img_file=None, supervision=None):
        super().__init__()
        self.data_path = Path(data_path)
        self.names = np.loadtxt(img_file, dtype='str')
        self.supervision = supervision
    def __len__(self):
        return len(self.names)
    def __getitem__(self, index):
        image_dir = self.data_path / 'test' / self.names[index]
        return load_images(image_dir, self.supervision)
    
