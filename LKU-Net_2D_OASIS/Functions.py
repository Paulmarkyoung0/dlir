"""
Helper functions from https://github.com/zhangjun001/ICNet.

Some functions has been modified.
"""

import numpy as np
import torch.utils.data as Data
from pathlib import Path
from skimage import io
from torchvision.transforms.functional import to_tensor

def crop_and_pad(img,sizex,sizey,sizez):
    img_new = np.zeros((sizex,sizey,sizez))
    h = np.amin([sizex,img.shape[0]])
    w = np.amin([sizey,img.shape[1]])
    d = np.amin([sizez,img.shape[2]])

    img_new[sizex//2-h//2:sizex//2+h//2,sizey//2-w//2:sizey//2+w//2,sizez//2-d//2:sizez//2+d//2]=img[img.shape[0]//2-h//2:img.shape[0]//2+h//2,img.shape[1]//2-w//2:img.shape[1]//2+w//2,img.shape[2]//2-d//2:img.shape[2]//2+d//2]
    return img_new
def rescale_intensity(image, thres=(0.0, 100.0)):
    """ Rescale the image intensity to the range of [0, 1] """
    image = image.astype(np.float32)
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2
def load_images(data_path, image_dir):
    # Load images and labels
    oct_path = data_path / image_dir / 'oct.png'
    oct_img = to_tensor(io.imread(oct_path, as_gray=True)).unsqueeze(0)
    hist_path = data_path / image_dir / 'hist.png'
    hist_img = to_tensor(io.imread(hist_path))
    oct_seg_path = data_path / image_dir / 'oct_seg.png'
    oct_seg = to_tensor(io.imread(oct_seg_path, as_gray=True)).unsqueeze(0)
    hist_seg_path = data_path / image_dir / 'hist_seg.png'
    hist_seg = to_tensor(io.imread(hist_seg_path, as_gray=True)).unsqueeze(0)
    return hist_img, oct_img, hist_seg, oct_seg

class TrainDataset(Data.Dataset):
    def __init__(self, data_path, img_file=None, trainingset = 1):
        'Initialization'
        super().__init__()
        self.data_path = Path(data_path)
        self.names = np.loadtxt(self.data_path / img_file, dtype='str')
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.names)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        mov_img, fix_img, mov_label, fix_label = load_images(self.data_path, self.names[index])
        return  mov_img, fix_img, mov_label, fix_label

class ValidationDataset(Data.Dataset):
    def __init__(self, data_path, img_file=None):
        super().__init__()
        self.data_path = Path(data_path)
        self.names = np.loadtxt(self.data_path / img_file, dtype='str')
    def __len__(self):
        return len(self.names)
    def __getitem__(self, index):
        mov_img, fix_img, mov_label, fix_label = load_images(self.data_path, self.names[index])
        return mov_img, fix_img, mov_label, fix_label
    