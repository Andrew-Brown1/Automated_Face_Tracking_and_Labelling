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

import ImageProcessor
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
                 recog_weights='',
                 detector_weights=''):
               
        utils.auto_init_args(self)
        
        self.OG_temp_dir = copy.deepcopy(self.temp_dir)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)
        
        print("Using GPUs: ", os.environ['CUDA_VISIBLE_DEVICES'])

        # ================================================================================================
        #  load the detection model
        # ================================================================================================
        self.net = models.RetinaFace()
        self.net = models.retina_face_load_model(self.net,self.detector_weights, device)
        self.net.eval()
        self.net = self.net.to(device)
        print('Finished loading detection model!')    
        # ================================================================================================
        #  load the ID discriminator model
        # ================================================================================================
        self.model = models.ID_discriminator_model_loader(self.recog_weights, hack=False)
        self.model.eval()
        self.model.to(device)
        print('Finished loading recognition model!')

        # ================================================================================================
        # read the video files
        # ================================================================================================

        self.file_paths = utils.getListOfFiles(self.path_to_vids)
        
        self.timer = utils.Timer()
    
    
    def run(self):
        """
        track the faces in the videos in self.file_paths 
        """

        with torch.no_grad():
            for ind, full_episode in enumerate(self.file_paths):
                print('video ' + str(ind) + ' of ' + str(len(self.file_paths)))

                # ----------------------------------------------------------
                # create local paths and variables for this video
                # ----------------------------------------------------------
                
                episode = full_episode.split('/')[-1]

                save_path = os.path.join(self.save_path,episode[:-4])
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                
                temp_file_name = episode

                self.temp_dir = os.path.join(self.OG_temp_dir,episode[:-4])
                if not os.path.isdir(self.temp_dir):
                    os.mkdir(self.temp_dir)
                    
                # make the full save path if it doesn't exist
                if not os.path.isdir(save_path):
                    save_folder = Path(save_path)
                    save_folder.mkdir(exist_ok=True, parents=True)

                # do not continue if:

                proceed = True
                if os.path.isfile(os.path.join(save_path, episode[:-4] + '.pickle')):
                    # this video has already been processed
                    proceed = False

                if proceed:
                    
                    # ----------------------------------------------------------
                    # (1) extract frames for this video to a temporary directory
                    # ----------------------------------------------------------
                    self.timer._start('frame extraction',self.verbose)
                    
                    # extract the frames using ffmpeg
                    utils.extract_frames_from_video(full_episode, self.temp_dir, temp_file_name)
                    
                    self.timer._log_end('frame extraction', self.verbose)
                    
                    # ----------------------------------------------------------
                    # (2) detect the faces in the frames
                    # ----------------------------------------------------------
                    self.timer._start('detecting faces',self.verbose)
                    
                    detection_dict = models.detect_faces(self, temp_file_name, self.net, device)
                            
                    self.timer._log_end('detecting faces', self.verbose)
                    
                    # ----------------------------------------------------------
                    # (3) extract ID discriminating features
                    # ----------------------------------------------------------
                    self.timer._start('extracting features',self.verbose)

                    TrackInfo = models.Extract_Features(self, temp_file_name, detection_dict, self.model)
                    
                    self.timer._log_end('extracting features', self.verbose)
                    
                    # ----------------------------------------------------------
                    # (4) create face-tracks using the detections and features
                    # face-tracking is done with a simple tracker that combines
                    # detection IOU and feature similarity
                    # ----------------------------------------------------------
                    self.timer._start('tracking faces',self.verbose)

                    ImageProcessor.Track(TrackInfo, os.path.join(save_path, episode))
                    
                    self.timer._log_end('tracking faces', self.verbose)
                    
                    # ----------------------------------------------------------
                    # (5) optionally, create a video with the face-trakcs shown 
                    # ----------------------------------------------------------
                                    
                    if self.make_video:
                        utils.MakeVideo(episode, self.temp_dir, save_path, full_episode, fps=vid_fps)
                    
                    # ----------------------------------------------------------
                    # (6) delete temporary written files
                    # ----------------------------------------------------------
                    os.system('rm -R '+ self.temp_dir)
                    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # paths
    parser.add_argument('--save_path', type=str,default='../save_dir/', help='path to where all outputs are saved')
    parser.add_argument('--path_to_vids', default='../weights_and_data/videos/', help='path to directory containing videos to process (mp4)', type=str)
    parser.add_argument('--temp_dir', default='../temporary_dir', help='path to where temporary directory can be created and then deleted at end of process',
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
    parser.add_argument('--detector_weights', type=str,default='../weights_and_data/weights/mobilenet0.25_Final.pth', help='path to weights for detector')
    # identity discriminator parameters
    parser.add_argument('--recog_batch_size', default=50, help='the batchsize', type=int)
    parser.add_argument('--recog_weights', default='../weights_and_data/weights/senet50_256.pth', type=str,
                        help='Trained state_dict file path to open for recognition model')
    args = parser.parse_args()
    
    
    
    videofacetracker = VideoFaceTracker(save_path=args.save_path,
                                    path_to_vids=args.path_to_vids,
                                    temp_dir = args.temp_dir,
                                    make_video=args.make_video,
                                    down_res=args.down_res,
                                    verbose=args.verbose,
                                    gpu=args.gpu,
                                    num_workers=args.num_workers,
                                    det_batch_size=args.det_batch_size,
                                    face_conf_thresh=args.face_conf_thresh,
                                    recog_batch_size=args.recog_batch_size,
                                    recog_weights=args.recog_weights,
                                    detector_weights=args.detector_weights)
    
    videofacetracker.run()