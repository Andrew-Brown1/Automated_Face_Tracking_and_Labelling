
import os
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
import torch
import cv2
import numpy as np 
from .model_datasets import detection_dataset_mobilenet, detection_downloaded_image_dir
from .retinaface import *
from tqdm import tqdm
import pdb

def detect_faces(args, episode, net, device, irregular_images=False):
    
    cfg = cfg_mnet


    if not irregular_images:
        dataset = detection_dataset_mobilenet(os.path.join(args.temp_dir, episode), args.down_res)
    else:
        dataset = detection_downloaded_image_dir(os.path.join(args.temp_dir, episode))

    
    dataloader = DataLoader(dataset, batch_size=args.det_batch_size, shuffle=False,
                            num_workers=int(args.num_workers), drop_last=False,
                            sampler=SequentialSampler(dataset))
    

    args.nms_threshold = 0.4

    detection_dict = {}

    all_dets = []

    for i, batch in enumerate(dataloader):
        

        scale = dataset.scale.to(device)
        img = batch.to(device)


        loc, conf, landms = net(img)

        priorbox = PriorBox(cfg, image_size=(dataset.im_height, dataset.im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        priors = torch.unsqueeze(priors, 0).repeat(batch.shape[0], 1, 1)

        prior_data = priors.data
        if len(loc.data.shape) > 3:
            loc_info = loc.data.squeeze(0)
        else:
            loc_info = loc.data
        boxes = decode(loc_info, prior_data, cfg[
            'variance'])  # torch.FloatTensor(np.repeat(np.expand_dims(np.asarray(cfg['variance']),0),[batch_size],axis=0)).cuda())
        boxes = boxes * scale / args.down_res
        boxes = boxes.cpu().numpy()
        
        scores = conf.data.cpu().numpy()[:, :, 1]

        for score_ind, score in enumerate(scores):
            inds = np.where(score > args.face_conf_thresh)[0]
            new_boxes = boxes[score_ind, inds]
            new_scores = score[inds]

            order = new_scores.argsort()[::-1]
            new_boxes = new_boxes[order]
            new_scores = new_scores[order]

            # do NMS
            dets = np.hstack((new_boxes, new_scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, args.nms_threshold)

            dets = dets[keep, :]

            all_dets.append(dets)
            
            
        # image_path = dataset.test_dataset[0]
        # img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # for det in dets:
        #     rect = [int(float(det[0])),int(float(det[1])),int(float(det[2])),int(float(det[3]))]
        #     img_raw = cv2.rectangle(img_raw, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (0,0,255), 7)
        # cv2.imwrite('temp.jpg', img_raw)


    # assemble the same dictionary for output

    assert(len(dataset.test_dataset) == len(all_dets))
    for ind, image in enumerate(dataset.test_dataset):
        entry = [[],[],[],[]]
        for det in all_dets[ind]:
            entry[2].append(det[:4].tolist())
            entry[3].append(det[-1])

        detection_dict[image] = entry

    return detection_dict




# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
