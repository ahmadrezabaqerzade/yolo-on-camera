import torch
import time
from ultralytics import models
import cv2
from draw import draw_box, draw_segment, draw_keypoints, plot_real_time
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

colors = ((0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
          (255, 0, 255), (0, 256, 255), (255, 255, 255),
          (100, 255, 50), (200, 30, 45), (45, 30, 0), (140, 255, 50))

device_id = 0
cap = cv2.VideoCapture(device_id)

model_id = 'yolo11m'

if torch.cuda.is_available():
    model = models.YOLO(f'{model_id}.pt').to(device)
    model.export(format='engine', int8=False, half=False, optimize=False)
    #model = models.YOLO(f'{model_id}.openvino')
    model = models.YOLO(f'yolo11m.engine')
    while True:
        size_img = (640, 640)
        size_win = (1000, 600)
        speed_win = (64, 600)
        start = time.time()
        image = cap.read()[1]
        image = cv2.resize(image, dsize=size_img)
        image = cv2.flip(image, 1)
        s_m = time.time()
        out = model(task = 'predict', model = 'predict', source = image, show = False, conf = 0.5)
        e_m = time.time()
        #image = np.zeros_like(image)
        names = out[0].names
        for result in out:
            image = draw_box(image, result, colors, names)
            try:
                image = draw_keypoints(image, result, colors)
            except:
                None

            try:
                image = draw_segment(image, result, colors, 0.9)
            except:
                None

        end = time.time()
        cv2.putText(img = image, text = f"speed: {int(1/(end - start))}fps", org = (2, 15),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0, 0, 200),
                    thickness=1, fontScale=0.5, lineType=cv2.LINE_AA)
        cv2.putText(img=image, text=f"model speed: {int(1 / (e_m - s_m))}fps", org=(140, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 200),
                    thickness=1, fontScale=0.5, lineType=cv2.LINE_AA)

        #cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
        #cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        image = cv2.resize(image, dsize=(size_win))
        #speed_img = plot_real_time([int(1 / (e_m - s_m))], [int(1 / (end - start))], speed_win)
        #image = np.concatenate([speed_img, image], axis = 1)
        cv2.imshow('yolo', image)
        cv2.resizeWindow("yolo", image.shape[1], image.shape[0])
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()