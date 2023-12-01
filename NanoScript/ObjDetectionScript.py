import torch
import torchvision
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2

# Loading our YOLOv5 model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp9/weights/last.pt', force_reload=True)
model.conf = 0.75


# Training model
#### python ./yolov5/train.py --img 320 --batch 16 --epochs 50 --data ./dataset.yaml --weights ./yolov5s.pt --workers 2

#img = 'https://miutrgv.github.io/images/resources/lab.jpg'
#results = model(img)
#results.print()


#%%matplotlib inline
# plt.imshow(np.squeeze(results.render()))
# plt.show()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    results = model(frame)

    cv2.imshow('YOLO', np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




