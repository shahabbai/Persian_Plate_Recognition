from ultralytics import YOLO
import cv2
import math
import time
import os
import glob
import torch


"""

This tiny script can crop plate from a bunch of images

"""

#load YOLOv8 model
model_object = YOLO("../weights/best.pt")

cars_path = 'path to your cars images'
plate_path = 'path for saving crop plates as jpg files'

jpg_files = glob.glob(os.path.join(cars_path, '*.jpg'))
temp = 0
for i in jpg_files:
    temp += 1
    img = cv2.imread(i)
    output = model_object(i, show=False, conf=0.75)
    if (len(output[0].boxes.cls)) > 1 :
        index = torch.nonzero(output[0].boxes.cls == 1.0).squeeze()
        if (torch.numel(index) == 1) :
            x1, y1, x2, y2 = output[0].boxes.xyxy[index].numpy().astype(int)
            #crop plate from frame
            plate_img = img[ y1:y2, x1:x2]
            cv2.imwrite(plate_path + '/' +str(temp)+ '.jpg', plate_img)
                
        else:

            for j in range(len(index)):
                x1, y1, x2, y2 = output[0].boxes.xyxy[index[j]].numpy().astype(int)
                #crop plate from frame
                plate_img = img[ y1:y2, x1:x2]
                cv2.imwrite(plate_path + '/' +str(temp)+ '.jpg', plate_img)
                temp += 1  
    
    



