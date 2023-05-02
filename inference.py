import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from models.experimental import attempt_load
import argparse

import os
import sys

def load_model(model_name = "run_19ehq8rw_model:v0"):
    root_path = os.path.abspath(os.getcwd())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = attempt_load(f'{root_path}/ckpt/{model_name}/best.pt', map_location=device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()  # to FP16
    model = model.to(device)
    return model
def main(model):
    root_path = os.path.abspath(os.getcwd())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    half = device.type != 'cpu'  # half precision only supported on CUDA
    _ = model.eval()

    source = cv2.imread(f'{root_path}/data/inference/source.png')
    image = source.copy()
    image_ = source.copy()
    image = letterbox(image, 1728, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)
    if half:    
        image = image.half()  # to FP16

    
    output = model(image)
    pred = non_max_suppression(output[0], 1e-3*3, 0.6, agnostic=False)
    det = pred[0]
    det[:, :4] = scale_coords(image.shape[2:], det[:, :4], source.shape).round()
    for i,(*xyxy, conf, cls) in enumerate(det):
        if cls == 0:
        # print(xyxy[0].detach().cpu().numpy(), conf)
            plot_one_box(xyxy, image_ , color=(0,(int(cls)*50)/255,(i*113)%255),line_thickness=5)
            # draw conf and cls on source using cv2
            cv2.putText(image_, f'{conf:.7f}', (int(xyxy[0].detach().cpu().numpy()), int(xyxy[1].detach().cpu().numpy())-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image_, f'{cls:.0f}', (int(xyxy[3].detach().cpu().numpy()), int(xyxy[1].detach().cpu().numpy())), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, bottomLeftOrigin=False)
            cv2.imwrite(f'{root_path}/data/inference/detect.png', image_)
            break
    x1, y1, x2, y2 = int(xyxy[0].detach().cpu().numpy()), int(xyxy[1].detach().cpu().numpy()), int(xyxy[2].detach().cpu().numpy()), int(xyxy[3].detach().cpu().numpy())
    pred_img=source[y1:y2, x1:x2, :]
    with open(f'{root_path}/data/inference/bbox.txt', 'w') as f:
        f.write(f'{x1} {y1} {x2} {y2}')
    cv2.imwrite(f'{root_path}/data/inference/infer.png', pred_img)

    if os.path.exists(f'{root_path}/data/inference/target.png'):
        target = cv2.imread(f'{root_path}/data/inference/target.png')
        target = target[y1:y2, x1:x2, :]
        cv2.imwrite(f'{root_path}/data/inference/target.png', target)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="run_19ehq8rw_model:v0", help='model name')
    args = parser.parse_args()
    model =load_model(args.model_name)
    main(model)
