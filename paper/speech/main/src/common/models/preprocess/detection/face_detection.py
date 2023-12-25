import torch
import torch.nn as nn
import torch.nn.functional as F

import mediapipe  as mp
import numpy as np
import cv2

class FaceDetect(nn.Module):
    def __init__(self, backbone=None):
        super(FaceDetect, self).__init__()
        self.mp_face_detection = mp.solutions.face_detection

    def forward(self, frames):
        bboxes_per_frame = []

        with self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
            
            ### TODO : batch 처리
            if frames.ndim==5: ### batch 잘라내기
                frames = frames[0,::]
                
            for i in range(frames.shape[0]):
                image = frames[i, ::].cpu().numpy()
                image = image.astype(np.uint8)
                results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                if not results.detections:
                    bboxes_per_frame.append({
                    'xmin':0,
                    'ymin':0,
                    'w':image.shape[1],
                    'h':image.shape[0]
                    })
                    continue
                
                bbox = results.detections[0].location_data.relative_bounding_box
                
                xmin = int(bbox.xmin*image.shape[1])
                ymin = int(bbox.ymin*image.shape[0])
                w,h = int(bbox.width*image.shape[1]), int(bbox.height*image.shape[0])

                bboxes_per_frame.append({
                    'xmin':xmin,
                    'ymin':ymin,
                    'w':w,
                    'h':h
                })
                
        return bboxes_per_frame