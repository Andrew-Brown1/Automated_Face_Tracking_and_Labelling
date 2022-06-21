import argparse
import os
import torch
import time
import cv2
import pdb
import torch.nn as nn
from pathlib import Path
import pickle
import copy

import Tracker
import models
import utils
device = torch.device("cuda")
os.system('module load apps/ffmpeg-4.2.1')


class VideoFaceTracker:
    def __init__(self,
                 save_path='',
                 path_to_vids='',
                 temp_dir = '',
                 make_video=False,
                 down_res=0.5,
                 verbose=True,
                 gpu='0',
                 num_workers=6,
                 det_batch_size=100,
                 face_conf_thresh=0.75,
                 recog_batch_size=50,
                 recog_weights=''):
        
        utils.auto_init_args(self)
        
        self.OG_temp_dir = copy.deepcopy(args.temp_dir)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        
        print("Using GPUs: ", os.environ['CUDA_VISIBLE_DEVICES'])

        print('USING SMOOTHAP FEATURES - CHECK IF THIS IS WHAT YOU WANT')
        # smooth-ap feats = '/scratch/shared/beegfs/abrown/AP_CVPR/AP-Project/Average_Precision_Face/weights/DSFD_Weights/weights_adam_apLoss_1e-06_b320_cpb80_an0.01_freeze_False_HNM1_HNM_choice4/2020-04-11_11-46-04/6.pth.tar SET HACK TO 1
        # regular VGF2 feats = '/scratch/shared/beegfs/abrown/Full_Tracker_Pipeline/weights/senet50_256.pth'
        
        # ================================================================================================
        #  load the detection model
        # ================================================================================================
        self.net = models.RetinaFace()
        self.net = models.retina_face_load_model(net, '/work/abrown/Face_Detectors/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth', device)
        self.net.eval()
        self.net = self.net.to(device)
        print('Finished loading detection model!')    
        # ================================================================================================
        #  load the ID discriminator model
        # ================================================================================================
        self.model = self.models.ID_discriminator_model_loader(self.recog_weights, hack=False)
        self.model.eval()
        self.model.to(device)
        print('Finished loading recognition model!')

        # ================================================================================================
        # read the video files
        # ================================================================================================

        self.file_paths = utils.getListOfFiles(args.path_to_vids)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # paths
    parser.add_argument('--save_path', type=str,default='/scratch/shared/beegfs/abrown/Full_Tracker_Pipeline/out', help='path to where all outputs are saved')
    parser.add_argument('--path_to_vids', default='/scratch/shared/beegfs/abrown/Full_Tracker_Pipeline/data/DFD', help='path to directory containing videos to process (mp4)', type=str)
    parser.add_argument('--temp_dir', default='/scratch/shared/beegfs/abrown/Full_Tracker_Pipeline/temp', help='path to where temporary directory can be created and then deleted at end of process',
                        type=str)
    # options
    parser.add_argument('--make_video', default=False, help='output the video of face tracks ', type=bool)
    parser.add_argument('--down_res', default=0.5, help='lower the resolution of the frames for the detection process to speed everything up', type=float)
    parser.add_argument('--verbose', default=True, help='print timings throughout processing', type=bool)
    # system
    parser.add_argument('--gpu', default='0', help='specify the gpu number', type=str)
    parser.add_argument('--num_workers', help='choose number of workers', default=6, type=int)
    # detecter parameters
    parser.add_argument('--det_batch_size', default=100, help='the batchsize', type=int)
    parser.add_argument('--face_conf_thresh', type=float,default=0.75, help='threshold for face detections being considered') 
    # identity discriminator parameters
    parser.add_argument('--recog_batch_size', default=50, help='the batchsize', type=int)
    parser.add_argument('--recog_weights', default='/scratch/shared/beegfs/abrown/Full_Tracker_Pipeline/weights/senet50_256.pth', type=str,
                        help='Trained state_dict file path to open for recognition model')
    args = parser.parse_args()
    
    
    
    videofacetracker = VideoFaceTracker(save_path=args.save_path,
                                    path_to_vids=args.path_to_vids,
                                    temp_dir = args.temp_dir,
                                    make_video=args.make_video,
                                    down_res=args.down_res,
                                    verbose=args.gpu,
                                    gpu=args.gpu,
                                    num_workers=args.num_workers,
                                    det_batch_size=args.det_batch_size,
                                    face_conf_thresh=args.face_conf_thresh,
                                    recog_batch_size=args.recog_batch_size,
                                    recog_weights=args.recog_weights)
    
    videofacetracker.run()