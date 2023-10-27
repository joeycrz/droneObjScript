import torch
import torchvision
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2

# Loading our YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

img = 'https://miutrgv.github.io/images/resources/lab.jpg'
results = model(img)
results.print()


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




