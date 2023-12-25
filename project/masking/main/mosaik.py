import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import cv2 
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon



# 1프로에서 5프로사이의 crop 된부분만 전체 마스킹하는 코드입니다. 
def mosaik(img,bboxes):
    for box in bboxes:
        #[y_min,y_max,x_min,x_max]) #
        
        cropped=img[box[0]:box[1],box[2]:box[3],:]
        
    
        cropped=np.array(cropped)
        cropped = gaussian_filter(cropped, sigma=16)
        img[box[0]:box[1],box[2]:box[3],:]=cropped


    return img
        
        
        
            
            
    
    