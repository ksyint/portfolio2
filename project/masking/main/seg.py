from unet.predict import predict_img,mask_to_image
import torch
from PIL import Image
import numpy as np
import cv2

# 아래는 unet이고 dinov2 전에 실험했던 모델입니다. 
def segmentation(img):
    device=torch.device("cuda")
    net=torch.load("weights/unet.pth")
    
    mask_values=[[0, 0, 0], [255, 255, 255]]

    mask=predict_img(net,img,device,scale_factor=1,out_threshold=0.5)
    result = mask_to_image(mask, mask_values)
    result=np.array(result)
    
    
    return result 


# 위 segmentation 을 통해서 crop 된 부분이 이미지내에서 몇프로 차지하는지 계산합니다.
# 아래 함수는 dinov2, unet모두에게 적용합니다
# 아래 코드는 하얀색 픽셀이 연속적으로 이어져서 만들어진 덩어리가 전체에서 몇프로 차지하는지 계산합니다. 
# 아래 코드는 덩어리(송장으로 추정) 들이 2개 이상이어도 적용할수 있습니다. 
def mask_percentage(mask_path):

    image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    ret, threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = image.shape[0] * image.shape[1]  
    contours_list=contours
    
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    

    percentages = [(area / total_area) * 100 for area in contour_areas]
    percentage_list=[]
    for i, percentage in enumerate(percentages):
        percentage_list.append(percentage)
    return contours_list,percentage_list



