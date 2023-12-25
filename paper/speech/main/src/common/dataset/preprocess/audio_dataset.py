import os
import torchvision
from torch.utils.data import Dataset
from glob import glob

class AudioDataset(Dataset): # video에서 audio만 추출
    def __init__(self, video_path, transform=None):
        self.video_path = glob(os.path.join(video_path, "*.mp4"))
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.video_path[idx]
        frames, audio, metadata = torchvision.io.read_video(filename, output_format='THWC', pts_unit='sec')
        
        if self.transform:
            audio = self.transform
        
        return audio.squeeze(0), os.path.basename(filename)[:-4], metadata['audio_fps']
    
    def __len__(self):
        return len(self.video_path)

if __name__ == "__main__":
    dataset = AudioDataset("/app/Talking_Head_Generation/dataset/LRS3")
    print(dataset.__len__())
    print(dataset[0][0].shape)