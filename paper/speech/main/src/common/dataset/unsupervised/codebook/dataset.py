import os
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import read_video
from glob import glob
from torch import nn
import sys
sys.path.append("../../../../../src")
from common.utils import Sobel, GaussianBlur

class PCL_Dataset(Dataset):
    def __init__(self, dir_path:str, size:int)->None:
        self.video_path = glob(os.path.join(dir_path, "*", "*.mp4"))
        self.flip = transforms.RandomHorizontalFlip(p=1.)
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.data_transforms = transforms.Compose([  
            transforms.Resize((size, size)),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.8),
            transforms.RandomApply([Sobel()], p=0.6),
            transforms.ToTensor()])
        self.normal_data_transform = transforms.Compose([transforms.Resize((size,size)),
                                                         transforms.ToTensor(),
                                                         ])
        self.tensor_to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.video_path)

    def __getitem__(self, idx):
        filename = self.video_path[idx]
        frames, audio, metadata = read_video(filename, output_format='THWC', pts_unit='sec')
        
        img = frames[random.choice(range(frames.shape[0]))] #TODO add face crop
        img = self.tensor_to_pil(img.permute((2, 0, 1)))
        img_flip = self.flip(img)
        img_normal = self.normal_data_transform(img)
        img_flip_normal = self.normal_data_transform(img_flip)

        neg_list = []
        exp_images = [self.data_transforms(img) for i in range(2)]
        for i in range(100):
            neg_list.append(self.data_transforms(img_flip))

        return {
            'img_normal':img_normal,
            'img_flip':img_flip_normal,
            'exp_images':exp_images,
            'neg_images':neg_list
        }

if __name__ == "__main__":
    dataset = PCL_Dataset("/app/lrs3", 64)
    print(dataset[0])