import torch
from torch.utils.data import DataLoader
import lightning as pl
import sys
import torchvision

class PCL(pl.LightningModule):
    def __init__(self,
                 model: torch.nn.Module, 
                 criterion: torch.nn.Module, 
                 optimizer: torch.nn.Module,
                 scheduler: torch.nn.Module,
                 dataset: dict,
                 batch_size: int, num_workers: int, save_path: str=None, print_img: bool=True):

        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
        self.print_img = print_img

    def train_dataloader(self):
        return self.dataloader
    
    def test_dataloader(self):
        return self.dataloader
    
    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def step(self, input_:dict) -> torch.Tensor:
        img_normal = input_['img_normal'] #(input_['img_normal'] + torch.randn_like(input_['img_normal'], device=input_['img_normal'].device)*4e-1
        img_flip = input_['img_flip'] #(input_['img_flip'] + torch.randn_like(input_['img_normal'], device=input_['img_normal'].device)*4e-1
        exp_images = torch.cat(input_['exp_images'],dim=0).to(img_normal.device)
        neg_list = torch.cat(input_['neg_images'],dim=0).to(img_normal.device)
        normal_exp_fea, normal_pose_fea, flip_exp_fea, flip_pose_fea, recon_normal_exp_flip_pose_img, recon_flip_exp_normal_pose_img, recon_normal_exp_normal_pose_img, recon_flip_exp_flip_psoe_img = self.model(normal_img=img_normal,
                                                                                                                                                                                                                flip_img=img_flip,
                                                                                                                                                                                                                recon_only=True, state='pfe')
        pose_fea_fc,pose_neg_fea_fc = self.model(exp_img=exp_images,
                                                exp_image_pose_neg=neg_list,
                                                recon_only=False,state='pose')

        return normal_exp_fea, normal_pose_fea, flip_exp_fea, flip_pose_fea, recon_normal_exp_flip_pose_img, recon_flip_exp_normal_pose_img, recon_normal_exp_normal_pose_img, recon_flip_exp_flip_psoe_img, pose_fea_fc,pose_neg_fea_fc

    def training_step(self, batch, batch_idx: int) -> dict:
        input_ = batch
        normal_exp_fea, normal_pose_fea, flip_exp_fea, flip_pose_fea, recon_normal_exp_flip_pose_img, recon_flip_exp_normal_pose_img, recon_normal_exp_normal_pose_img, recon_flip_exp_flip_psoe_img, pose_fea_fc,pose_neg_fea_fc = self.step(input_)
        loss, log_dict = self.criterion(
                                input_['img_normal'],
                                input_['img_flip'],
                                normal_exp_fea, 
                                normal_pose_fea, 
                                flip_exp_fea, 
                                flip_pose_fea, 
                                recon_normal_exp_flip_pose_img, 
                                recon_flip_exp_normal_pose_img, 
                                recon_normal_exp_normal_pose_img, 
                                recon_flip_exp_flip_psoe_img, 
                                pose_fea_fc,pose_neg_fea_fc)
        if self.print_img:
            print_img = torch.cat([input_['img_normal'][:2], input_['img_flip'][:2], recon_flip_exp_normal_pose_img[:2],
                        recon_normal_exp_flip_pose_img[:2], recon_normal_exp_normal_pose_img[:2]],
                        dim=3)
            grid = torchvision.utils.make_grid(print_img,nrow=1)
            #self.log.experiment.add_image('train_img', grid)
            self.logger.experiment.add_image('train_img', grid) #, global_step=total_num * step + i)
        return log_dict
    
    def on_training_epoch_end(self, outputs: list) -> None:
        for key in outputs[0].keys():
            loss = [output[key] for output in outputs]
            self.log("train/{}".format(key), np.mean(loss), on_epoch=True, sync_dist=True)