
import os
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
import torch
import numpy as np 
from .model_datasets import detection_dataset_mobilenet
from .retinaface import *

def detect_faces(args, episode, net, device):

    cfg = cfg_mnet
    dataset = detection_dataset_mobilenet(os.path.join(args.temp_dir, episode), args.down_res)

    scale = dataset.scale.to(device)
    args.nms_threshold = 0.4

    dataloader = DataLoader(dataset, batch_size=args.det_batch_size, shuffle=False,
                            num_workers=int(args.num_workers), drop_last=False,
                            sampler=SequentialSampler(dataset))

    detection_dict = {}

    all_dets = []

    for i, batch in enumerate(dataloader):

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
