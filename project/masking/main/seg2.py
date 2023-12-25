import torch
from dino2.model import Segmentor
from PIL import Image 
from torchvision import transforms
import numpy as np 
import cv2



T2=transforms.ToPILImage()


weights="/home/ksyint/other1213/craft_ku/weights/dinov2.pt"
device=torch.device("cpu")
model = Segmentor(device,1,backbone = 'dinov2_b',head="conv")
model.load_state_dict(torch.load(weights,map_location="cpu"))
model = model.to(device)


img_transform = transforms.Compose([
    transforms.Resize((14*64,14*64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# 송장부분을 crop 하기 위한 코드 
# dinov2 모델을 segmentation 모델로 활용하였고 이를 train 하여서 pretrained weight를 weight 폴더에 저장하고 아래와 같이 load합니다.

def dino_seg(numpy_array):
    img0=Image.fromarray(numpy_array)
    original_size=img0.size
    img=img_transform(img0)
    a=img.unsqueeze(0)
    b=model(a)
    b=b.squeeze(0)
    b=b*255.0
    model_output=T2(b) #pil image 
    model_output=model_output.resize(original_size)
    
    model_output=np.array(model_output)
    model_output[model_output > 220] = 255.0
    model_output[model_output <= 220] = 0.0
    model_output2=model_output
    model_output3=model_output
    output = np.stack([model_output, model_output2, model_output3])
    output=np.transpose(output,(1,2,0))
    

    return output




# def find_connected_components(image):
    
    
#     _, binary_image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

#     _, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)
   

#     largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    
#     x, y, w, h,_=stats[largest_component_index, cv2.CC_STAT_LEFT:cv2.CC_STAT_TOP+cv2.CC_STAT_HEIGHT+1]
    
#     return x, y, w, h



