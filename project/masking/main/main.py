from recognize import recongize
from ner import ner
import os
import time
import argparse
from sr.sr import sr
import torch
from scipy.ndimage import gaussian_filter
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from mosaik import mosaik
from PIL import Image
import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
from seg import mask_percentage

from seg2 import dino_seg

from craft import CRAFT
from collections import OrderedDict
import gradio as gr
from refinenet import RefineNet


# craft, refine 모델 불러오는 코드 
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='사전학습 craft 모델')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--refine', default=True, help='enable link refiner')
parser.add_argument('--image_path', default="input/2.png", help='input image')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()
# 아래는 option
def full_img_masking(full_image,net,refine_net):
    reference_image=sr(full_image)
    reference_boxes=text_detect(reference_image,net=net,refine_net=refine_net)
    boxes=get_box_from_refer(reference_boxes)
    for index2,box in enumerate(boxes):
        xmin,xmax,ymin,ymax=get_min_max(box)
        
        text_area=full_image[int(ymin):int(ymax),int(xmin):int(xmax),:]
        
        text=recongize(text_area)
        label=ner(text)
        
        if label==1:
            A=full_image[int(ymin):int(ymax),int(xmin):int(xmax),:]
            full_image[int(ymin):int(ymax),int(xmin):int(xmax),:] = gaussian_filter(A, sigma=16)
    return full_image

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    
    x = Variable(x.unsqueeze(0))               
    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y, feature = net(x)

    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)


    return boxes, polys, ret_score_text

def text_detect(image,net,refine_net): 
    
        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        
        return bboxes


def get_box_from_refer(reference_boxes):
    
    real_boxes=[]
    for box in reference_boxes:
       
        real_boxes.append(box//2)
    
    return real_boxes
def get_min_max(box):
    xlist=[]
    ylist=[]
    for coor in box:
        xlist.append(coor[0])
        ylist.append(coor[1])
    return min(xlist),max(xlist),min(ylist),max(ylist)
        
def main(image_path0):
# 1단계
    
    # ==> craft 모델과 refinenet 모델을 불러오고 cuda device 에 얹힙니다. 
    
    net = CRAFT()     
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
   
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = False

    net.eval()

    refine_net = None
    if args.refine:
        refine_net = RefineNet()
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
        

        refine_net.eval()
        args.poly = True

# 2단계 

    # gradio 빈칸에 이미지를 넣고 A 에 입력됩니다. 
   
    A=image_path0
    image_list=[]
    image_list.append(A)
    for k, image_path in enumerate(image_list):
        


        image = imgproc.loadImage(image_path)
        if image.shape[2]>3:
            image=image[:,:,0:3]
        
        original_image=image 
        # 이미지에서 송장부분만 dinov2 모델로 segmentation 을 합니다.

        output=dino_seg(image)
        image3=Image.fromarray(output)
        image3.save("temporal_mask/mask.png")
        
        # 마스크이미지(white pixel, black background)를 만듭니다.
        # 위 마스크 이미지에서 각 덩어리들(송장으로 추정)이 전체 이미지내에서 몇프로차지하는지 계산합니다.
        contours_list,percentage_list=mask_percentage("temporal_mask/mask.png")
        
        normal_image_list=[]
   
        small_coordinate_list=[]
        original_coordinate_list=[]
        
        
#3단계 
    
        
        
        sorted_list = sorted(percentage_list, reverse=True)
        top_5 = sorted_list[:5]
        print("상위 5개 값:", top_5)
        # percentage list의 경우 송장으로 추정되는 뭉치들의 퍼센트를 모아놓은것이고
        # contours list는 이미지내에서 송장으로 추정되는 뭉치들이 크롭되어서 정렬된 리스트입니다. 
        # 예 : percentatge list 의 첫번째 요소는 contours list의 첫번째 요소의 percentage 
        
        for index,percentage in enumerate(percentage_list):
            
            if 5<percentage:
                
            # percentage 가 아미지내에서 5프로 넘는 것들은 normal list로 포함됩니다.
            # normal list안에는 이미지내에서 충분히 큰 뭉치들(송장으로 추정) 을 모아놓았습니다.
            # 1-5프로 인것들은 small coordinate list에 포함되고 매우 작은 뭉치로 간주합니다.
            # 매우작은 뭉치의 경우 zoom in을 했을때 뭉치(송장으로 추정)내 글자가 거의 보이지않아서 따라서 뭉치 전체를 mosaik합니다. 
            # 1프로미만 뭉치들은 소멸직전일정도로 작아 생략합니다. 
    
                contour=contours_list[index]
                
                x_list=[]
                y_list=[]
                contour2=list(contour)
                
                for r in contour2:
                    r2=r[0]
                    x_list.append(r2[0])
                    y_list.append(r2[1])
                x_min=min(x_list)
                y_min=min(y_list)
                x_max=max(x_list)
                y_max=max(y_list)
                original_coordinate_list.append([y_min,y_max,x_min,x_max])
                image2=original_image[y_min:y_max,x_min:x_max,:]
                normal_image_list.append(image2)
                
                
            #
            elif 1<percentage<5:
                contour=contours_list[index]
                
                x_list=[]
                y_list=[]
                contour2=list(contour)
                
                for r in contour2:
                    r2=r[0]
                    x_list.append(r2[0])
                    y_list.append(r2[1])
                x_min=min(x_list)
                y_min=min(y_list)
                x_max=max(x_list)
                y_max=max(y_list)
                small_coordinate_list.append([y_min,y_max,x_min,x_max]) #송장 5프로미만의 좌표
            else:
                continue
        
        
        
        
        # 4단계 (매우작은 송장)
        
        # small coordinate list안에 매우작은 송장들이 모여져있지만 list안에 요소가 없으면 5단계로 바로갑니다. 
        # 바로 가지않을경우(list 안요소 최소하나) mosaik 를 통해서 전체이미지에서 작은 뭉치에 해당하는 좌표들을 모두 모자이크합니다. 
                
        if len(small_coordinate_list)>0:
            original_image=mosaik(original_image,small_coordinate_list)
        else:
            pass
        
        # 5단계 (어느정도 사이즈 있는 송장) ==> normal list
        
        # normal image list안에 적절한 크기의 송장(줌 하면 글자 보이는) 들이 있습니다.
        # craft 입장에서 text 위치를 return 할수 있게끔 크롭된 송장을 esrgan 으로 화질개선합니다.
        # 화질개선된 송장을 craft에 넣어서 정확하게 text 좌표들을 모두 구합니다. 
        # 좌표를 구할때 화질 좋은 송장이미지의 좌표를 그대로 return 하지 않고 원본 송장이미지에 맞추어서 scale(//2) 하고 최종좌표를 구합니다. 
        
        for index,normal_image in enumerate(normal_image_list):
            reference_image=sr(normal_image)
            reference_boxes=text_detect(reference_image,net=net,refine_net=refine_net)
            boxes=get_box_from_refer(reference_boxes)
            for index2,box in enumerate(boxes):
                xmin,xmax,ymin,ymax=get_min_max(box)
               
                text_area=normal_image[int(ymin):int(ymax),int(xmin):int(xmax),:]
                text_area=Image.fromarray(text_area)
                os.makedirs("text_area",exist_ok=True)
                text_area.save(f"text_area/new_{index2+1}.png")
                
                
        # 6단계 (text recognize, ner)
        
        # 위 좌표들을 통해서 송장 내에서 박스들을 크롭합니다.
        # 크롭된 송장내 부분(크롭된 박스 , 즉 text 있는곳으로 추정되는곳) 을 trocr 에넣습니다.
        # trocr은 상자내에 추정되는 text를 보여줍니다. 
        # text를 ko electra 에넣어서 해당 상자에있는 text가 개인정보인지아닌지 판별합니다.
        # 송장내 해당 상자가 개인정보로(레이블 :1) 추정될경우 모자이크를합니다. 
        # 모자이크라고 판별할경우 해당상자의 좌표를 송장이미지에 맞는 좌표로 변환하고 그 좌표에 해당하는 부분을 모자이크합니다.
        # 부분적으로 모자이크된 송장이미지를 전체이미지(송장을 포함하는 이미지)에 붙입니다. 
        
                text=recongize(text_area)
                label=ner(text)
                with open("output/text_recongnize.txt","a") as recognized:
                    recognized.writelines(str(index2+1))
                    recognized.writelines(" ")
                    recognized.writelines(str(text))
                    recognized.writelines(" ")
                    recognized.writelines(str(label))
                    recognized.writelines("\n")
                    recognized.close()
                print("done")
                if label==1:
                   A=normal_image[int(ymin):int(ymax),int(xmin):int(xmax),:]
                   normal_image[int(ymin):int(ymax),int(xmin):int(xmax),:] = gaussian_filter(A, sigma=16)
                   
                else:
                    pass
            a,b,c,d=original_coordinate_list[index]
            original_image[a:b,c:d,:]=normal_image 
            
        # 더 정확도 높이기위해서 이미지 전체(송장과 배경 둘다) craft에 통째로 넣기
        # 단 optional (단점 : infer speed )
        
        # print("full mask start")
        # original_image=full_img_masking(original_image,net=net,refine_net=refine_net)
        # print("full mask done")
        
        
        
        original_image=Image.fromarray(original_image)
        original_image.save("output/mosaiked.png")
        print("masked complete")
        return original_image

        
if __name__ == '__main__':

    import subprocess

    bash_script_path = 'reset.sh'


    subprocess.run(['bash', bash_script_path], check=True)
    

    iface = gr.Interface(
        fn=main,
        inputs=gr.Image(type="filepath", label="Invoice Image"),
        outputs=gr.Image(type="pil", label="Masked Invoice Image"),
        live=True
    )

    iface.launch()
    
