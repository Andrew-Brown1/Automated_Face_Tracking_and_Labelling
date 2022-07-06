

"""
this is the pipeline to do everything - go from a directory of images to "cleaned" feature vectors 

(1) detect the faces - in Full_tracker_pipeline/src

(2) extract features - /scratch/shared/beegfs/abrown/BBC_work/ICMR_BL/Extract_Features

(3) do the cleaning + decide famous / non-famous - in the /scratch/shared/beegfs/abrown/BBC_work/ICMR_BL directory 

(4) save the dictionary 

(5) re-use everything possible from the video pipeline, and maybe objectify some stuff

(6) track the videos 

(6.5) get the annotations into some format for the annotating

(7) do the annotation - with the query expansion as well (ignore the voice and non-famous people for now)

(8) visualise the annotations

(9) think a lot about the output format 
"""
from VideoFaceTracker import VideoFaceTracker
from ImageFaceRecognise import ImageFaceRecognise

import argparse 
import ImageProcessor
import utils


class VideoFaceAnnotator:
    def __init__(self,
                 save_path='',
                 path_to_vids='',
                 temp_dir = '',
                 make_video=False,
                 down_res=0.5,
                 verbose=True,
                 gpu='0',
                 num_workers=6,
                 path_to_input='',
                 det_batch_size=100,
                 face_conf_thresh=0.75,
                 recog_batch_size=50,
                 irregular_images=True,
                 recog_weights='',
                 make_annotation_video=True,
                 face_annotation_threshold=0.5,
                face_annotation_QE_threshold=0.5,
                detector_weights='',
                only_use_non_outlier_faces=True):

        utils.auto_init_args(self)

        # ================================================================================================
        #  load the video face tracker
        # ================================================================================================

        self.videofacetracker = VideoFaceTracker(save_path=save_path,
                                    path_to_vids=path_to_vids,
                                    temp_dir = temp_dir,
                                    make_video=make_video,
                                    down_res=down_res,
                                    verbose=verbose,
                                    gpu=gpu,
                                    num_workers=num_workers,
                                    det_batch_size=det_batch_size,
                                    face_conf_thresh=face_conf_thresh,
                                    recog_batch_size=recog_batch_size,
                                    recog_weights=recog_weights,
                                    detector_weights=detector_weights)
        
        
        # ================================================================================================
        #  load the face image detection object
        # ================================================================================================
        
        self.imagefacerecogniser = ImageFaceRecognise(save_path=save_path,
                                    path_to_input=path_to_input,
                                    gpu=gpu,
                                    num_workers=num_workers,
                                    face_conf_thresh=face_conf_thresh,
                                    recog_batch_size=recog_batch_size,
                                    recog_weights=recog_weights,
                                    verbose=verbose,
                                    irregular_images=irregular_images,
                                    det_batch_size=det_batch_size,
                                    loaded_face_detector=self.videofacetracker.net,
                                    loaded_face_recogniser=self.videofacetracker.model)
        
        
        # ================================================================================================
        #  load video face track annotator
        # ================================================================================================
        
        self.track_annotator = ImageProcessor.TrackAnnotator(save_path=save_path,
                                                              path_to_vids=path_to_vids,
                                                              path_to_input=path_to_input,
                                                              temp_dir = temp_dir,
                                                              make_annotation_video=make_annotation_video,
                                                              face_verification_threshold=face_annotation_threshold,
                                                              query_expansion_threshold=face_annotation_QE_threshold,
                                                              only_use_non_outlier_faces=only_use_non_outlier_faces)
        
            
    def run(self):
        
        # ================================================================================================
        #  run the video face tracker
        # ================================================================================================
        self.videofacetracker.run()
        
        # ================================================================================================
        #  run the image face recogniser 
        # ================================================================================================
        self.imagefacerecogniser.run()
        
        # ================================================================================================
        #  annotate the tracks
        # ================================================================================================
        self.track_annotator.run()
        
            
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # paths
    parser.add_argument('--save_path', type=str,default='../save_dir/', help='path to where all outputs are saved')
    parser.add_argument('--path_to_vids', default='../weights_and_data/videos/', help='path to directory containing videos to process (mp4)', type=str)
    parser.add_argument('--temp_dir', default='../temporary_dir', help='path to where temporary directory can be created and then deleted at end of process',
                        type=str)
    parser.add_argument('--path_to_image_dirs', default='../weights_and_data/person_images/', help='path to parent directory of image-directories', type=str)
    # options
    parser.add_argument('--irregular_images', default=True, help='the images in the directories are all different sizes. If set to true, the detector will not any image and use a batch size of 1. If False, the detector will batch the images and resize them according to the down_res argument', type=bool)
    parser.add_argument('--make_video', default=False, help='output the video of face tracks ', type=bool)
    parser.add_argument('--make_annotation_video', default=False, help='output the video of face tracks annotations ', type=bool)
    parser.add_argument('--down_res', default=1, help='lower the resolution of the frames for the detection process to speed everything up', type=float)
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
    # face annotation parameters
    parser.add_argument('--face_annotation_threshold', default=0.5, help='the initial face annotating threshold', type=int)
    parser.add_argument('--face_annotation_QE_threshold', default=0.5, type=int,help='the face annotating threshold for query expanded bank')
    parser.add_argument('--only_use_non_outlier_faces', default=True, type=int,help='assuming that the face-images contain noise. When set to True, automatic outlier detection is used. If set to False, then all images in the face images directories are used')
    
    args = parser.parse_args()
        
    video_face_annotator = VideoFaceAnnotator(save_path=args.save_path,
                                    path_to_vids=args.path_to_vids,
                                    temp_dir = args.temp_dir,
                                    path_to_input=args.path_to_image_dirs,
                                    make_video=args.make_video,
                                    down_res=args.down_res,
                                    verbose=args.verbose,
                                    gpu=args.gpu,
                                    num_workers=args.num_workers,
                                    det_batch_size=args.det_batch_size,
                                    irregular_images=args.irregular_images,
                                    face_conf_thresh=args.face_conf_thresh,
                                    recog_batch_size=args.recog_batch_size,
                                    recog_weights=args.recog_weights,
                                    make_annotation_video=args.make_annotation_video,
                                    face_annotation_threshold=args.face_annotation_threshold,
                                    face_annotation_QE_threshold=args.face_annotation_QE_threshold,
                                    detector_weights=args.detector_weights)

    
    video_face_annotator.run()