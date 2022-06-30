"""

This script contains an object ImageFaceRecognise 

when used, this object will iterate over a set of directories of face-images, and ouput some meta data (detections, features, whether to use or not, as well as aggregated)

this also has a feature of "clean" which will do the famous / non-famous part 

"""

import os
import pdb 
import pickle
import torch
import models
import argparse
import numpy as np
import utils
import ImageProcessor

device = torch.device("cuda")
os.system('module load apps/ffmpeg-4.2.1')

class ImageFaceRecognise:
    
    def __init__(self,
                 save_path='',
                 path_to_input='',
                 gpu='0',
                 loaded_face_detector=None,
                 loaded_face_recogniser=None,
                 num_workers=6,
                 down_res=1,
                 face_conf_thresh=0.9,
                 recog_batch_size=64,
                 irregular_images=True,
                 verbose=True,
                 det_batch_size=1,
                 recog_weights=''):
        
        utils.auto_init_args(self)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)
        print("Using GPUs: ", os.environ['CUDA_VISIBLE_DEVICES'])
        print('USING SMOOTHAP FEATURES - CHECK IF THIS IS WHAT YOU WANT')
        # smooth-ap feats = '/scratch/shared/beegfs/abrown/AP_CVPR/AP-Project/Average_Precision_Face/weights/DSFD_Weights/weights_adam_apLoss_1e-06_b320_cpb80_an0.01_freeze_False_HNM1_HNM_choice4/2020-04-11_11-46-04/6.pth.tar SET HACK TO 1
        # regular VGF2 feats = '/scratch/shared/beegfs/abrown/Full_Tracker_Pipeline/weights/senet50_256.pth'
        
        # ================================================================================================
        #  load the detection model
        # ================================================================================================
        if self.loaded_face_detector is None:
            self.net = models.RetinaFace()
            self.net = models.retina_face_load_model(self.net, '/work/abrown/Face_Detectors/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth', device)
            self.net.eval()
            self.net = self.net.to(device)
            print('Finished loading detection model!')    
        else:
            self.net = loaded_face_detector
            self.net.eval()
            self.net = self.net.to(device)
            print('Finished loading detection model!')    
        # ================================================================================================
        #  load the ID discriminator model
        # ================================================================================================
        if self.loaded_face_recogniser is None:
            self.model = models.ID_discriminator_model_loader(self.recog_weights, hack=False)
            self.model.eval()
            self.model.to(device)
            print('Finished loading recognition model!')
        else:
            self.model = loaded_face_recogniser
            self.model.eval()
            self.model.to(device)
            print('Finished loading recognition model!')
        
        # ================================================================================================
        #  get the list of image directories
        # ================================================================================================
        self.image_dirs = [f for f in os.listdir(self.path_to_input) if os.path.isdir(os.path.join(self.path_to_input,f))]
        self.temp_dir = self.path_to_input # just for the detector 
        
        self.timer = utils.Timer()
        
        if self.irregular_images:
            self.det_batch_size = 1
        
        # ================================================================================================
        #  prepare the "cleaning" face processor (outlier removal stage)
        # ================================================================================================
        
        self.OutlierDetector = ImageProcessor.FaceOutlierDetection()
        
    def _prepare_outputs(self,outputs, dominant_class, outlier_labels):
        """
        prepare the outputs for the image face recognise
        
        """
        outputs['x'] = np.concatenate([np.array([f]) for f in outputs['x']])
        outputs['y'] = np.concatenate([np.array([f]) for f in outputs['y']])
        outputs['w'] = np.concatenate([np.array([f]) for f in outputs['w']])
        outputs['h'] = np.concatenate([np.array([f]) for f in outputs['h']])
        
        outputs['Features'] = np.concatenate([np.expand_dims(f,0) for f in outputs['Features']], axis=0)
        
        # aggregated features 
        # (a) for every computed feature (when ignoring outlier detection)
        outputs['aggregated_feature_all'] = ImageProcessor.tracker.AverageVectorfunc(outputs['Features'])
        #  (b) for the non-outlier features
        features_without_outliers = np.concatenate([np.expand_dims(f,0) for ind, f in enumerate(outputs['Features'] ) if outlier_labels[ind] == 1 ], axis=0)

        outputs['aggregated_feature_without_outliers'] = ImageProcessor.tracker.AverageVectorfunc(features_without_outliers)
        
        outputs['famous'] = dominant_class
        outputs['outlier_labels'] = outlier_labels
        
        return outputs

       
    def run(self):
        """
        detect and extract features from the faces in the directories in self.image_dirs        
        """

        with torch.no_grad():
            for ind, image_dir in enumerate(self.image_dirs):
                print('image dir ' + str(ind) + ' of ' + str(len(self.image_dirs)))
                
                # do not continue if:
                proceed = True
                if os.path.isfile(os.path.join(self.save_path, image_dir + '.pk')):
                    # this video has already been processed
                    proceed = False
                
                if proceed:
                    
                    # ----------------------------------------------------------
                    # (1) detect the faces in the image directory
                    # ----------------------------------------------------------

                    self.timer._start('detecting faces',self.verbose)
                    
                    detection_dict = models.detect_faces(self, image_dir, self.net, device, irregular_images=self.irregular_images)
                            
                    self.timer._log_end('detecting faces', self.verbose)
                    
                    # ----------------------------------------------------------
                    # (2) extract ID discriminating features
                    # ----------------------------------------------------------
                    
                    self.timer._start('extracting features',self.verbose)

                    Feature_Info = models.Extract_Features(self, image_dir, detection_dict, self.model)
                    
                    self.timer._log_end('extracting features', self.verbose)
                                        
                    # ----------------------------------------------------------
                    # (3) optionally - flag outlier faces. This process finds the 
                    # most commonly depicted face in the directory, and ignores
                    # the outlier faces from different identities when 
                    # computing an aggregated representation
                    # ----------------------------------------------------------
                    
                    dominant_class, outlier_labels = self.OutlierDetector.run(Feature_Info)
                    
                    # ----------------------------------------------------------
                    # (4) prepare the outputs
                    # ----------------------------------------------------------
                    
                    # x, y, w, h, features, image_names, labels, bool, aggregated_features (both)
                    
                    outputs = self._prepare_outputs(Feature_Info, dominant_class, outlier_labels)
                                                            
                    with open(os.path.join(self.save_path, image_dir + '.pk'),'wb') as f:
                        pickle.dump(outputs, f)


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    
    # paths
    parser.add_argument('--path_to_image_dirs', default='/scratch/shared/beegfs/abrown/Full_Tracker_Pipeline/temp_image_dirs', help='path to parent directory of image-directories', type=str)
    parser.add_argument('--save_path', default='/scratch/shared/beegfs/abrown/Full_Tracker_Pipeline/out',  help='path to where all outputs are saved')
    # options
    parser.add_argument('--irregular_images', default=True, help='the images in the directories are all different sizes. If set to true, the detector will not any image and use a batch size of 1. If False, the detector will batch the images and resize them according to the down_res argument', type=bool)
    parser.add_argument('--down_res', default=0.5, help='lower the resolution of the images for the detection process to speed everything up. This is only used if irregular_images is False', type=float)
    parser.add_argument('--verbose', default=True, help='print timings throughout processing', type=bool)   
    # system
    parser.add_argument('--gpu', default='0', help='specify the gpu number', type=str)
    parser.add_argument('--num_workers', help='choose number of workers', default=6, type=int)
    # detecter parameters
    parser.add_argument('--face_conf_thresh', type=float,default=0.99, help='threshold for face detections being considered') 
    parser.add_argument('--det_batch_size', default=100, help='the batchsize. This is only used if irregular_images is False. Otherwise is 1.', type=int)
    # identity discriminator parameters
    parser.add_argument('--recog_batch_size', default=50, help='the batchsize', type=int)
    parser.add_argument('--recog_weights', default='/scratch/shared/beegfs/abrown/Full_Tracker_Pipeline/weights/senet50_256.pth', type=str,
                        help='Trained state_dict file path to open for recognition model')
    args = parser.parse_args()
    
    
    imagefacerecogniser = ImageFaceRecognise(save_path=args.save_path,
                                    path_to_input=args.path_to_image_dirs,
                                    gpu=args.gpu,
                                    num_workers=args.num_workers,
                                    face_conf_thresh=args.face_conf_thresh,
                                    recog_batch_size=args.recog_batch_size,
                                    recog_weights=args.recog_weights,
                                    verbose=args.verbose,
                                    irregular_images=args.irregular_images,
                                    det_batch_size=args.det_batch_size)
    
    imagefacerecogniser.run()