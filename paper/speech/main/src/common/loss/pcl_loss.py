import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("../../../src")
from common.utils import topk_accuracy as accuracy


class PCL_Loss(torch.nn.Module):
    def __init__(self, recon_loss: torch.nn.Module, diff_loss: torch.nn.Module, pose_loss: torch.nn.Module, recon_weight: int = 1, pose_weight: int = 1, neg_alpha: float = 1.6, temperature: float = 0.07)->None:
        super().__init__()
        self.recon_loss = recon_loss
        self.diff_loss = diff_loss
        self.pose_loss = pose_loss
        self.recon_weight = recon_weight
        self.pose_weight = pose_weight
        self.neg_alpha = neg_alpha
        self.temperature = temperature

    def neg_inter_info_nce_loss_pose(self, features: torch.Tensor, neg_pose_fea: torch.Tensor)->torch.Tensor:
        b,dim=features.size()

        neg_pose_fea_li = list(neg_pose_fea.chunk(b//2,dim=0))
        neg_pose_fea_tensor = torch.stack(neg_pose_fea_li,dim=0)

        neg_mask = np.random.beta(self.neg_alpha, self.neg_alpha,
                                  size=(neg_pose_fea_tensor.shape[0], neg_pose_fea_tensor.shape[1]))
        if isinstance(neg_mask,np.ndarray):
            neg_mask = torch.from_numpy(neg_mask).float().to(features.device)
            neg_mask = neg_mask.unsqueeze(dim=2)
        indices = torch.randperm(neg_pose_fea_tensor.shape[1])
        neg_pose_fea_tensor = neg_pose_fea_tensor * neg_mask + (1 - neg_mask) * neg_pose_fea_tensor[:,indices]

        features = F.normalize(features,dim=1)
        q,k = features.chunk(2,dim=0)
        neg_pose_fea_tensor = F.normalize(neg_pose_fea_tensor,dim=2)
        neg_pose_fea_tensor = neg_pose_fea_tensor.repeat(2,1,1)

        pos = torch.cat(
            [torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1), torch.einsum('nc,nc->n', [k, q]).unsqueeze(-1)], dim=0)

        neg_pose_fea_tensor = neg_pose_fea_tensor.transpose(2,1)
        neg = torch.bmm(features.view(b,1,-1),neg_pose_fea_tensor).view(b,-1)

        logits = torch.cat([pos,neg],dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        logits = logits / self.temperature

        return logits,labels

    def forward(self, img_normal: torch.Tensor, img_flip: torch.Tensor,
                normal_exp_fea: torch.Tensor, 
                normal_pose_fea: torch.Tensor, 
                flip_exp_fea: torch.Tensor, 
                flip_pose_fea: torch.Tensor, 
                recon_normal_exp_flip_pose_img: torch.Tensor, 
                recon_flip_exp_normal_pose_img: torch.Tensor, 
                recon_normal_exp_normal_pose_img: torch.Tensor, 
                recon_flip_exp_flip_psoe_img: torch.Tensor,
                pose_fea_fc: torch.Tensor,
                pose_neg_fea_fc: torch.Tensor):

        recon_normal_loss = self.recon_loss(recon_flip_exp_normal_pose_img, img_normal)  # || s-s'||
        recon_flip_loss = self.recon_loss(recon_normal_exp_flip_pose_img, img_flip)  # ||f-f'||
        recon_orin_loss = self.recon_loss(recon_normal_exp_normal_pose_img, img_normal)  # ||s-s''||
        recon_flip_ori_loss = self.recon_loss(recon_flip_exp_flip_psoe_img, img_flip)
        loss_pfe = recon_normal_loss + recon_flip_loss + recon_orin_loss + recon_flip_ori_loss

        diff_loss = self.diff_loss(normal_exp_fea, normal_pose_fea) + self.diff_loss(flip_exp_fea, flip_pose_fea)

        pose_logits,pose_labels = self.neg_inter_info_nce_loss_pose(pose_fea_fc,pose_neg_fea_fc)
        pose_loss = self.pose_loss(pose_logits,pose_labels)
        pose_acc1,pose_acc5=accuracy(pose_logits,pose_labels,(1,5))
        loss = diff_loss + self.recon_weight * loss_pfe +  pose_loss * self.pose_weight
        log_dic = {
            'diff_loss': diff_loss.item(),
            'recon_normal_loss':recon_normal_loss.item(),
            'recon_flip_loss':recon_flip_loss.item(),
            'recon_orin_loss':recon_orin_loss.item(),
            'recon_flip_orin_loss':recon_flip_ori_loss.item(),
            'pose_acc1':pose_acc1,
            'pose_acc5':pose_acc5,
            'pose_loss':pose_loss.item(),
            "Total_loss": loss.item(),
            "loss": loss
        }
        return loss, log_dic

class DiffLoss(torch.nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        #diff_loss = torch.mean((input1_l2 * input2_l2).sum(dim=1).pow(2))

        return diff_loss

if __name__=="__main__": 
    l1 = torch.nn.L1Loss()
    diff = DiffLoss()
    ce = torch.nn.CrossEntropyLoss()
    loss = PCL_Loss(l1, diff, ce)
    print(loss(img_normal=torch.rand(10, 3, 64, 64), img_flip=torch.rand(10, 3, 64, 64),
                normal_exp_fea=torch.rand(10, 2048), 
                normal_pose_fea=torch.rand(10, 2048), 
                flip_exp_fea=torch.rand(10, 2048), 
                flip_pose_fea=torch.rand(10, 2048), 
                recon_normal_exp_flip_pose_img=torch.rand(10, 3, 64, 64), 
                recon_flip_exp_normal_pose_img=torch.rand(10, 3, 64, 64), 
                recon_normal_exp_normal_pose_img=torch.rand(10, 3, 64, 64), 
                recon_flip_exp_flip_psoe_img=torch.rand(10, 3, 64, 64),
                pose_fea_fc=torch.rand(10, 512),
                pose_neg_fea_fc=torch.rand(10, 512)))