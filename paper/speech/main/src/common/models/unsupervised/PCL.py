import torch
import torch.nn.functional as F
from torch import nn

class ExpPoseModel(nn.Module):
    def __init__(self, encoder:torch.nn.Module, decoder:torch.nn.Module):
        super(ExpPoseModel, self).__init__()
        self.encoder = encoder #FaceCycleBackbone()

        self.pose_fc = nn.Sequential(nn.Linear(2048, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048,512),
                                         nn.BatchNorm1d(512))

        self.decoder = decoder #Generator(d_model=2048)

    def forward(self,exp_img=None,normal_img=None,flip_img=None,exp_image_pose_neg=None,recon_only=False,state='pfe'):
        if not recon_only:
            if state == 'pfe':
                _,normal_exp_fea,normal_pose_fea = self.encoder(normal_img)
                _,flip_exp_fea,flip_pose_fea = self.encoder(flip_img)

                recon_normal_exp_flip_pose_fea = F.normalize(normal_exp_fea+flip_pose_fea, dim=1)
                recon_flip_exp_normal_pose_fea = F.normalize(flip_exp_fea+normal_pose_fea, dim=1)
                recon_normal_exp_normal_pose_fea = F.normalize(normal_exp_fea+normal_pose_fea, dim=1)
                recon_flip_exp_flip_pose_fea = F.normalize(flip_exp_fea+flip_pose_fea,dim=1)

                recon_normal_exp_flip_pose_img = self.decoder(recon_normal_exp_flip_pose_fea)
                recon_flip_exp_normal_pose_img = self.decoder(recon_flip_exp_normal_pose_fea)
                recon_normal_exp_normal_pose_img = self.decoder(recon_normal_exp_normal_pose_fea)
                recon_flip_exp_flip_psoe_img = self.decoder(recon_flip_exp_flip_pose_fea)

                return normal_exp_fea,normal_pose_fea,flip_exp_fea,flip_pose_fea, recon_normal_exp_flip_pose_img,recon_flip_exp_normal_pose_img,recon_normal_exp_normal_pose_img,recon_flip_exp_flip_psoe_img

            elif state == 'pose':
                _,_,pose_fea = self.encoder(exp_img)
                pose_fea_fc = self.pose_fc(pose_fea)
                # todo check
                _,_,pose_neg_fea = self.encoder(exp_image_pose_neg)
                pose_neg_fea_fc = self.pose_fc(pose_neg_fea)
                return pose_fea_fc,pose_neg_fea_fc
        else:
            _,normal_exp_fea, normal_pose_fea = self.encoder(normal_img)
            _,flip_exp_fea, flip_pose_fea = self.encoder(flip_img)

            recon_normal_exp_flip_pose_fea = F.normalize(normal_exp_fea + flip_pose_fea, dim=1)
            recon_flip_exp_normal_pose_fea = F.normalize(flip_exp_fea + normal_pose_fea, dim=1)
            recon_normal_exp_normal_pose_fea = F.normalize(normal_exp_fea + normal_pose_fea, dim=1)
            recon_flip_exp_flip_pose_fea = F.normalize(flip_exp_fea + flip_pose_fea, dim=1)

            recon_normal_exp_flip_pose_img = self.decoder(recon_normal_exp_flip_pose_fea)
            recon_flip_exp_normal_pose_img = self.decoder(recon_flip_exp_normal_pose_fea)
            recon_normal_exp_normal_pose_img = self.decoder(recon_normal_exp_normal_pose_fea)
            recon_flip_exp_flip_psoe_img = self.decoder(recon_flip_exp_flip_pose_fea)

            return normal_exp_fea,normal_pose_fea,flip_exp_fea,flip_pose_fea, recon_normal_exp_flip_pose_img,recon_flip_exp_normal_pose_img,recon_normal_exp_normal_pose_img,recon_flip_exp_flip_psoe_img



if __name__=="__main__":
    from networks import FaceCycleBackbone, Generator
    encoder = FaceCycleBackbone()
    decoder = Generator(d_model=2048)
    model = ExpPoseModel(encoder, decoder)
    pose_fea_fc,pose_neg_fea_fc = model(exp_img=torch.rand((2,3,64, 64)),
                                        exp_image_pose_neg=torch.rand((2,3,64, 64)),
                                        recon_only=False, state='pose')
    print(pose_fea_fc.shape, pose_neg_fea_fc.shape)
