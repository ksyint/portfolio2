import torch
import lightning as pl
from torch.utils.data import DataLoader, Dataset
import os

class AudioFeatureExtra(pl.LightningModule):
    def __init__(self, backbone: torch.nn.Module,
                    dataset: dict, #(Dataset),
                    batch_size: int = 1,
                    num_workers: int = 1,
                    save_path: str = "./audio_feature"
                    )-> None:
        super().__init__()
        self.model = backbone
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.dataloader = DataLoader(dataset["preprocess"], batch_size=batch_size, shuffle=False)#, num_workers=num_workers)

    def test_dataloader(self):
        return self.dataloader

    def step(self, input_ct):
        output = self.model(input_ct)
        return output

    def predict_step(self, batch, batch_idx) -> None:
        audio, filename, info = batch
        audio_feature_ = self.step(audio)
        save_fpath = os.path.join(self.save_path, filename[0]+".pt")
        torch.save(audio_feature_, save_fpath)