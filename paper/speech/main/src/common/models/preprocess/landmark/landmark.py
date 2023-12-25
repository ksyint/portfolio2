import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import sys
import os
sys.path.append(os.path.realpath("./src/external/"))
import cv2

from dense_face_alignment.predictor import FaceMeshPredictor

# from .dense_face_alignment.detector import mediapipe_face_detection
# from .dense_face_alignment.landmarks import find_68_lmks, find_custom_lmks, find_dense_lmks

import random
import numpy as np
from .pose_estimation import PoseEstimator

class Landmark(nn.Module):
    def __init__(self, debug=False) -> None:
        super(Landmark, self).__init__()
        self.predictor =  FaceMeshPredictor.dad_3dnet()
        self.pose_estimator = PoseEstimator(255, 255)
        self.debug = debug
    
    def _make_result(self, pose:tuple, landmark:list) -> dict:
        return {
            "pose": {
                    "rotation_vector": [int(p) for p in pose[0][:, 0].tolist()],
                    "translation_vector": [int(p) for p in pose[1][:, 0].tolist()]
                },
                "landmark": landmark}

    def _numpy2tensor(self, array: torch.Tensor)-> np.array:
        array = array.cpu().numpy()
        array = array.astype(np.uint8)
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        return array

    def forward(self, frames: torch.Tensor, bboxes: list) -> (list, list):           
        returns = []
        for i in range(frames.shape[1]):
            image = self._numpy2tensor(frames[0, i, ::])

            cropped_face = image[bboxes[i]['ymin']:bboxes[i]['ymin']+bboxes[i]['h'], bboxes[i]['xmin']:bboxes[i]['xmin']+bboxes[i]['w']]
            coordinates = self.predictor(cropped_face)['points']
            pred_x, pred_y = coordinates[:,0], coordinates[:,1]
            
            xmin = int(bboxes[i]['xmin'])
            ymin = int(bboxes[i]['ymin'])
            
            pred_x_new, pred_y_new = xmin + pred_x, ymin + pred_y #the coordiantes of landmarks on the original image
            lmks = []
            for idx,point in enumerate(zip(pred_x_new, pred_y_new)):
                lmks.append([int(p) for p in point])
                if self.debug:
                    image = cv2.circle(image, point, radius=2, color=(0, 0, 255), thickness=-1) #save the original image with landmark on it

            pose = self.pose_estimator.solve(np.array(lmks, dtype=np.float32))
            returns.append(self._make_result(pose, lmks))
            if self.debug:
                self.pose_estimator.visualize(image, pose, color=(0, 255, 0))
                cv2.imwrite('test/test%d.png'%i, image)

        return returns
        
if __name__=='__main__':
    test = Landmark('any')
    frames, audio, metadata = torchvision.io.read_video('/home/jml20/lrs3test/test.mp4', output_format='THWC')
    
    print(frames.shape)
    lmks = test(frames)
    
    print(np.array(lmks).shape)