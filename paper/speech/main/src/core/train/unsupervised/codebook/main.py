import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import sys
import random
import numpy as np
import os
sys.path.append(os.path.realpath("./src"))
import core
from common.utils import seed_everything, instantiate_dict
from common.models.unsupervised import FaceCycleBackbone
seed_everything()

@hydra.main(version_base=None, config_path="../../../../config/unsupervised", config_name="config_PCL")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)
    
    cfg = instantiate_dict(cfg)  

    cfg["Trainer"].fit(cfg["module"])
    # print('all things done! It worked!')

if __name__ == "__main__":
    main()