import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import sys
import random
import numpy as np
import os
sys.path.append(os.path.realpath("./src"))
import core
from core.utils import seed_everything
from torch.utils.data import DataLoader, Dataset
    
seed_everything()

@hydra.main(version_base=None, config_path="../../../config", config_name="config_audio")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)
    
    cfg["module"]["backbone"] = instantiate(cfg["module"]["backbone"])
    cfg['module']['dataset']['preprocess'] = instantiate(cfg['module']['dataset']['preprocess'])

    cfg["module"] = instantiate(cfg["module"])
    cfg["Trainer"] = instantiate(cfg["Trainer"])

    cfg["Trainer"].predict(cfg["module"], cfg["module"].dataloader)

if __name__ == "__main__":
    main()