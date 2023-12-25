import torch
from torch import nn
import torch.nn.functional as F
from loss import BCE3
from modules import ImageEncoder, ProjectionHead

def scaling(value):
    return (value + 1) / 2

device=torch.device("cuda:0")
criterion=BCE3().to(device)


class Model(nn.Module):
    def __init__(
        self,
        image_embedding=1000,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.image_encoder2 = ImageEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.image_projection2 = ProjectionHead(embedding_dim=image_embedding)
       
    def forward(self, img1,img2,label):
        
        
        image_features = self.image_encoder(img1)
        image_features2 = self.image_encoder2(img2)

        image_embeddings = self.image_projection(image_features)
        image_embeddings2 = self.image_projection2(image_features2)
        
        logits=F.cosine_similarity(image_embeddings,image_embeddings2)
        
        logits=scaling(logits)
        logits=logits.to(torch.float)
        label=label.to(torch.float)
        loss=criterion(logits,label)
        
        return loss


