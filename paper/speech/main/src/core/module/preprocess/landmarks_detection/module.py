import torch
import lightning as pl
from torch.utils.data import DataLoader, Dataset
import os
import json
import sys
import numpy as np
from natsort import natsorted
sys.path.append(os.path.realpath("./src"))
from common.utils import make_dir

class LandmarksDetection(pl.LightningModule):
    def __init__(self, backbone: torch.nn.Module,
                 dataset: dict,
                 detection: torch.nn.Module = None,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 save_path: str = './landmark')-> None:
        super().__init__()
        self.model = backbone
        self.dataloader = DataLoader(dataset["preprocess"], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.detection = detection
        self.save_path = save_path
    
    def test_dataloader(self) -> DataLoader:
        return self.dataloader

    def step(self, frames:torch.Tensor) -> dict:
        if self.detection:
            bboxes = self.detection(frames)
        else:
            bboxes = [{"xmin":0, "ymin":0, "w":255, "h":255} for _ in range(frames.shape[1])]
        results = self.model(frames, bboxes)
        return results

    def predict_step(self, batch, batch_idx) -> None:
        video, dirname, filename, info = batch
        folderpath = make_dir(self.save_path, dirname[0])
        json_save_path = os.path.join(folderpath, filename[0]+".json")
        results = self.step(video)
        print(json_save_path)
        with open(json_save_path, "w") as json_file:
            json.dump(results, json_file, indent='\t')
        
        