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

############################################## TO DO ########################################################

# 1) change the aspect ratio constraint - currently all vidoes are exctracted at the same aspect ratio - this 
# is first used when extracting the frames and then I think it is propogated to how the detections are scaled.
# This should be changed such that each individual video is extracted at its actual aspect ratio. 

############################################################################################################

os.system('module load apps/ffmpeg-4.2.1')
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

    args.OG_temp_dir = copy.deepcopy(args.temp_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Using GPUs: ", os.environ['CUDA_VISIBLE_DEVICES'])

    print('USING SMOOTHAP FEATURES - CHECK IF THIS IS WHAT YOU WANT')
    # smooth-ap feats = '/scratch/shared/beegfs/abrown/AP_CVPR/AP-Project/Average_Precision_Face/weights/DSFD_Weights/weights_adam_apLoss_1e-06_b320_cpb80_an0.01_freeze_False_HNM1_HNM_choice4/2020-04-11_11-46-04/6.pth.tar SET HACK TO 1
    # regular VGF2 feats = '/scratch/shared/beegfs/abrown/Full_Tracker_Pipeline/weights/senet50_256.pth'

    if args.recog_weights == '/scratch/shared/beegfs/abrown/Full_Tracker_Pipeline/weights/senet50_256.pth':
        args.hack = False
    else:
        args.hack=True
    
    # ================================================================================================
    #  load the detection model
    # ================================================================================================
    net = models.RetinaFace()
    net = models.retina_face_load_model(net, '/work/abrown/Face_Detectors/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth', device)
    net.eval()
    net = net.to(device)
    print('Finished loading detection model!')    
    # ================================================================================================
    #  load the ID discriminator model
    # ================================================================================================
    model = models.ID_discriminator_model_loader(args.recog_weights, hack=args.hack)
    model.eval()
    model.to(device)
    print('Finished loading recognition model!')

    
    # ================================================================================================
    # read the video files
    # ================================================================================================

    file_paths = utils.getListOfFiles(args.path_to_vids)

    # args.original_path = args.path_to_vids

    all_times = []
    
    # ================================================================================================
    # start the video processing
    # ================================================================================================
    
    timer = utils.Timer()
        
    with torch.no_grad():
        for ind, full_episode in enumerate(file_paths):
            print('video ' + str(ind) + ' of ' + str(len(file_paths)))

            # ----------------------------------------------------------
            # create local paths and variables for this video
            # ----------------------------------------------------------
            
            episode = full_episode.split('/')[-1]

            save_path = os.path.join(args.save_path,episode[:-4])
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            
            temp_file_name = ''.join(full_episode[(len(args.path_to_vids)+1):-(len(episode))].split('/'))+episode

            args.temp_dir = os.path.join(args.OG_temp_dir,episode[:-4])
            if not os.path.isdir(args.temp_dir):
                os.mkdir(args.temp_dir)
                
            # make the full save path if it doesn't exist
            if not os.path.isdir(save_path):
                save_folder = Path(save_path)
                save_folder.mkdir(exist_ok=True, parents=True)

            # do not continue if:
            proceed = True
            if os.path.isfile(os.path.join(save_path, episode + '.pickle')):
                # this video has already been processed
                proceed = False

            if proceed:
                
                # ----------------------------------------------------------
                # (1) extract frames for this video to a temporary directory
                # ----------------------------------------------------------
                timer._start('frame extraction',args.verbose)

                # (a) find the resolution and fps of the videos

                vid = cv2.VideoCapture(full_episode)
                vid_resolution = [int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))]
                vid_fps = vid.get(cv2.CAP_PROP_FPS)
                if args.verbose:
                    start1 = time.time()
                start1 = time.time()

                # (b) extract the frames (if not done already)
                
                if not os.path.isdir(os.path.join(args.temp_dir, temp_file_name)):
                    os.mkdir(os.path.join(args.temp_dir, temp_file_name))

                    Command = "ffmpeg -i " + full_episode + " -threads 1 -deinterlace -q:v 1 -s "+str(vid_resolution[0])+":"+str(vid_resolution[1])+" -vf fps="+str(vid_fps) + " " + args.temp_dir + "/" + temp_file_name + "/%06d.jpg"

                    os.system(Command)
                
                timer._log_end('frame extraction', args.verbose)
                
                # ----------------------------------------------------------
                # (2) detect the faces in the frames
                # ----------------------------------------------------------
                timer._start('detecting faces',args.verbose)
                
                detection_dict = models.detect_faces(args, temp_file_name, net, device)
                        
                timer._log_end('detecting faces', args.verbose)
                
                # ----------------------------------------------------------
                # (3) extract ID discriminating features
                # ----------------------------------------------------------
                timer._start('extracting features',args.verbose)

                TrackInfo = models.Extract_Features(args, temp_file_name, detection_dict, model)
                
                timer._log_end('extracting features', args.verbose)
                
                # ----------------------------------------------------------
                # (4) create face-tracks using the detections and features
                # face-tracking is done with a simple tracker that combines
                # detection IOU and feature similarity
                # ----------------------------------------------------------
                timer._start('tracking faces',args.verbose)

                Tracker.Track(TrackInfo, os.path.join(save_path, episode))
                
                timer._log_end('tracking faces', args.verbose)
                # ----------------------------------------------------------
                # (5) optionally save a video of the face-tracks 
                # ----------------------------------------------------------
                                
                if args.make_video:
                    utils.MakeVideo(episode, args.temp_dir, save_path, full_episode, fps=vid_fps)
                
                # ----------------------------------------------------------
                # (6) delete temporary written files
                # ----------------------------------------------------------
                os.system('rm -R '+ args.temp_dir)

