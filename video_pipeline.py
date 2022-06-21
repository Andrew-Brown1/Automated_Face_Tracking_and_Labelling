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


from MakeVideo import MakeVideos


# from processes import detect_faces, Extract_Features

# from Track import Track


# from utils import getListOfFiles
# from retinaface import cfg_mnet, load_model, RetinaFace
device = torch.device("cuda")

############################################## TO DO ########################################################

# 1) change the aspect ratio constraint - currently all vidoes are exctracted at the same aspect ratio - this 
# is first used when extracting the frames and then I think it is propogated to how the detections are scaled.
# This should be changed such that each individual video is extracted at its actual aspect ratio. 

# Furthermore - the "down_res" argment is too specific - this should depend on the original resoltion of the video
# A fix to this would be to extract the original resolution and aspect ratio and then choose the down_res from there.
# it might be the case that if the video is a really low resolution anyway then you shouldn't down res it even furhter.

# there is the problem now that if I am looking at the face detections of past tracks, I don't know what res they were 
# extracted at and they are not normalised.

# 2) add syncnet capability - this should take the tracked faces and then return a wav file for each (where possible) 
# this wav file only exists if the syncnet provided ASD confirms that they are talking - and only for the proportion 
# of the track that they are talking in.

# 3) read the whole video into RAM in each iteration, then have every subsequent part of the process read the frames 
# from RAM - this will speed everything up massively. 

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
    parser.add_argument('--save_face_dets', default='', help='save the face dets or not (advisable for long vids)', type=str)
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

    # OG_down_res = copy.deepcopy(args.down_res)

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

            save_path = os.path.join(args.save_path,full_episode[(len(args.path_to_vids)+1):-(len(episode))])
            
            temp_file_name = ''.join(full_episode[(len(args.path_to_vids)+1):-(len(episode))].split('/'))+episode

            # make the full save path if it doesn't exist
            if not os.path.isdir(save_path):
                save_folder = Path(save_path)
                save_folder.mkdir(exist_ok=True, parents=True)

            # do not continue if:
            proceed = True
            if not os.path.isfile(os.path.join(save_path, episode + '.pickle')):
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

                if args.save_face_dets:
                    # save the face dets
                    with open(os.path.join(args.save_face_dets, temp_file_name+'fdets.pk'),'wb') as f:
                        pickle.dump(detection_dict, f)
                        
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

                output_tracks, track_features = Tracker.Track(TrackInfo)
                
                
                # save the face-tracks
                with open(os.path.join(save_path,episode+'.pickle'),'wb') as f:
                    pickle.dump(output_tracks, f)
                with open(os.path.join(save_path,episode+'_TrackFeats.pk'),'wb') as f:
                    pickle.dump(track_features, f)
                
                
                # ----------------------------------------------------------
                # (4) optionally save a video of the face-tracks 
                # TODO: (1) add audio to the video 
                # TODO: (2) make sure that this works with down-res
                # ----------------------------------------------------------
                MakeVideos([episode], args.temp_dir, save_path, syncnet=False)
                pdb.set_trace()
                if args.make_video:
                    MakeVideos([episode], args.temp_dir, save_path, syncnet=False)





    very_end = time.time()
    pdb.set_trace()
