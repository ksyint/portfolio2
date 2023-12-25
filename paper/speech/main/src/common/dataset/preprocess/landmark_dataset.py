import os
import torchvision
from torch.utils.data import Dataset
from glob import glob

import mediapipe as mp

import sys
sys.path.append('/home/jml20/Talking_Head_Generation/src/common/detection')

import cv2
import numpy as np

class LandmarkDataset(Dataset): # video에서 audio만 추출
    def __init__(self, video_path, transform=None):
        self.video_path = glob(os.path.join(video_path, "*", "*.mp4"))
        self.transform = transform
        
    def __getitem__(self, idx):
        filename = self.video_path[idx]
        frames, audio, metadata = torchvision.io.read_video(filename, output_format='THWC', pts_unit='sec')
        
        if self.transform:
            frames = self.transform
            
        return frames, filename.split("/")[-2], os.path.basename(filename)[:-4], metadata['video_fps']
    
    def __len__(self):
        return len(self.video_path)

if __name__ == "__main__":
    dataset = LandmarkDataset("/home/jml20/lrs3test")
    print(dataset.__len__())
    
    frames, filename, meta = dataset[0]
    print(frames.shape)
