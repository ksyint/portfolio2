import torch
from PIL import Image
import numpy as np
from .esrgan.RealESRGAN import RealESRGAN
def sr(img):
   # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device=torch.device("cuda")
    model = RealESRGAN(device, scale=2)
    model.load_weights('weights/RealESRGAN_x2.pth', download=True)


    img=Image.fromarray(img)
    sr_image = model.predict(img)
    sr_image=np.array(sr_image)
    return sr_image