# -*- coding: UTF-8 -*-
import argparse
from genericpath import isdir
import time
from pathlib import Path
import os
import glob

import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from numpy import random
import copy
import numpy as np
import math

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import hrnetlib.models as models
from hrnetlib.utils.transforms import crop
from hrnetlib.core.evaluation import decode_preds
from hrnetlib.config import config, update_config
from collections import OrderedDict


def load_yolo_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

def load_hrnet_model(opt, device):
    model = models.get_face_alignment_net(config)
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).to(device)

    state_dict = torch.load(opt.hrnet_weights)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.module.load_state_dict(new_state_dict)
    
    model.eval()
    return model

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def detect_face(opt, model, orgimg, image_name, device):
    # Load model
    # img_size = 800
    # conf_thres = 0.3
    # iou_thres = 0.5
    img_size = opt.img_size
    conf_thres = opt.conf_thres
    iou_thres = opt.iou_thres

    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img1 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
    else:
        img1 = img0

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img1, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, hwc to chw

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    
    # post-process
    bboxes = []
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:15].view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                img0 = plot_bboxes(img0, xyxy, conf, landmarks, class_num)
                
                bboxes.append(xyxy)
    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)
    base_name, _ = os.path.splitext(image_name)
    save_path = os.path.join(opt.save_folder, "{}_bboxes.jpg".format(base_name))
    cv2.imwrite(save_path, img0)
    
    return bboxes


def prepare_input(orgimg, bbox, image_size):
    """

    :param image: the laoded image using opencv, BRG format
    :param bbox:The bbox of target face
    :param image_size: refers to config file
    :return:
    """
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200
    center_w = (bbox[0] + bbox[2]) / 2
    center_h = (bbox[1] + bbox[3]) / 2
    center = torch.Tensor([center_w, center_h])
    scale *= 1.25
    
    # preprocess
    img = orgimg[:, :, ::-1].copy() # 
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = crop(img, center, scale, image_size, rot=0)
    img = img.astype(np.float32)
    img = (img / 255.0 - mean) / std
    img = img.transpose([2, 0, 1])
    img = torch.Tensor(img)
    img = img.unsqueeze(0)
    return img, center, scale


def detect_landmarks(config, model, orgimg, bboxes, device):
   
    landmarks = []
    
    for bbox in bboxes:
        # preprocess
        inp, center, scale = prepare_input(orgimg, bbox, config.MODEL.IMAGE_SIZE)
        inp = inp.to(device)
        output = model(inp)
        score_map = output.data.cpu()
        preds = decode_preds(score_map, center, scale, [64, 64])
        preds = preds.numpy()
        landmarks.append(preds)
        
    return landmarks


def plot_bboxes(img, xyxy, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def save_results(orgimg, landmarks, save_folder, image_name):
    for landmark in landmarks:
        for i in landmark[0, :, :]:
            cv2.circle(orgimg, tuple(list(int(p) for p in i.tolist())), 2, (255, 255, 0), 1)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    base_name, _ = os.path.splitext(image_name)
    save_path = os.path.join(save_folder, "{}_landmarks.jpg".format(base_name))
    cv2.imwrite(save_path, orgimg)


def main(opt, config, yolo_model, hrnet_model, device):
    img_paths = []
    if os.path.isdir(opt.input):
        img_paths = sorted(glob.glob(os.path.join(opt.input, "*.jpg")))
    elif os.path.isfile(opt.input):
        img_paths = [opt.input]
    else:
        print("input is not an file nor a folder")
    
    for img_path in img_paths:
        image_name = os.path.basename(img_path)
        print("processing {} ...".format(image_name))
        orgimg = cv2.imread(img_path)  # BGR
        bboxes = detect_face(opt, yolo_model, orgimg, image_name, device)
        landmarks = detect_landmarks(config, hrnet_model, orgimg, bboxes, device)
        save_results(orgimg, landmarks, opt.save_folder, image_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, help='path to yolo model.pt', 
                        default='runs/train/yolov5le_5x5_stem_coco_pre_p800b64/weights/best.pt')
    parser.add_argument('--cfg', type=str, help='experiment configuration filename', 
                       default='../HRNet-Facial-Landmark-Detection/experiments/300w/face_alignment_300w_hrnet_w18.yaml')
    parser.add_argument('--hrnet_weights', type=str, 
                        default='/mnt/cephfs/home/dengzeshuai/pretrained_models/Detection/HRnet/HR18-300W.pth', help='path to hrnet model.pt')
    parser.add_argument('--input', type=str, default='/mnt/cephfs/home/dengzeshuai/data/Detection/IBUG300W/ibug', help='source')  # file or folder
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default="0", help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save_folder', default='./runs/detect/yolov5le_hrnet/', type=str, help='Dir to save results')
    opt = parser.parse_args()
    update_config(config, opt)

    print(opt)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if opt.device else "cpu")
    yolo_model = load_yolo_model(opt.yolo_weights, device)
    hrnet_model = load_hrnet_model(opt, device)

    main(opt, config, yolo_model, hrnet_model, device)
