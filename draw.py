import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt


def draw_box(image, result, colors, class_names):
    for i, obj in enumerate(result.boxes):
        conf = obj.conf.item()

        if conf > 0.6:
            bbox = obj.xyxy[0].cpu().numpy().astype(int)
            cls = int(obj.cls.item())
            if cls > len(colors):
                l = len(colors) - 1
            else:
                l = cls

            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[l], 2)
            cv2.putText(image, class_names[cls], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[l], 1)
    return image

def draw_keypoints(image, result, colors):
    for j, obj in enumerate(result.keypoints.cpu()):
        keypoints = obj.xy[j].tolist()
        conf = obj.conf[j].tolist()
        for i, (key, c) in enumerate(zip(keypoints, conf)):
            if i > len(colors):
                l = len(colors) - 1
            else:
                l = i
            if c > 0.5:
                cv2.circle(image, (int(key[0]), int(key[1])), 4, colors[l], thickness=-1, lineType=cv2.FILLED)
                cv2.putText(image, f"{i}", (int(key[0]), int(key[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[l], 1,
                            lineType=cv2.LINE_AA)

    return image

def draw_segment(image, result, colors, alpha):
    for i, obj in enumerate(result.masks):
        if i > len(colors) - 1:
            l = len(colors) - 1
        else:
            l = i
        mask = obj.data.cpu()
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=colors[l])
        image_overlay = masked.filled()
        image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image


def plot_real_time(speed_model, speed_all, speed_win):
    plt.bar(['speed model'], speed_model)
    plt.text(0, 2, f'{speed_model[0]}', fontsize = 20)
    plt.bar(['speed'], speed_all)
    plt.text(1, 2, f'{speed_all[0]}', fontsize = 20)
    plt.yticks(range(0, 70))
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf)
    sp_im = cv2.resize(np.array(Image.open(buf).convert('RGB')), dsize=speed_win)
    plt.close()
    return sp_im
