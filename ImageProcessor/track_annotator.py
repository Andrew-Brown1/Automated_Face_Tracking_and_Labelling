import os 
import pdb
import pickle
import numpy as np
import utils
import ImageProcessor

class TrackAnnotator:
    
    def __init__(self, save_path, 
                 path_to_vids,
                 path_to_input,
                 temp_dir,
                 face_verification_threshold=0.7, 
                 query_expansion_threshold=0.7,
                 only_use_non_outlier_faces=True,
                 min_track_duration=5,
                 make_annotation_video=False):
        
        utils.auto_init_args(self)
        
    def _UTILS_read_image_face_recognition(self):
        # read the image face recognition data into features, depending on 
        # if outliers are to be considered or not
        
        # ------------------------------------------------
        # read the features
        # ------------------------------------------------
        
        people = [f + '.pk' for f in os.listdir(self.path_to_input) if os.path.isfile(os.path.join(self.save_path,f+'.pk'))]
        self.face_dictionary_names = []
        self.face_dictionary_feats = []
        for person in people:
            # read the data 
            with open(os.path.join(self.save_path,person),'rb') as f:
                face_data = pickle.load(f)

                if not self.only_use_non_outlier_faces:
                    # then use all features from the face images 
                    self.face_dictionary_names.append(person[:-3])
                    self.face_dictionary_feats.append(face_data['aggregated_feature_all'])
                elif face_data['famous']:
                    # then only use the non-outlier features for the non-outlier identities
                    self.face_dictionary_names.append(person[:-3])
                    self.face_dictionary_feats.append(face_data['aggregated_feature_without_outliers'])
        
        # ------------------------------------------------
        # aggregate the features to single matrix
        # ------------------------------------------------
                 
        if len(self.face_dictionary_names) == 0:
            raise Exception('No identities to annotate with')
        else:
            self.face_dictionary_names = np.array(self.face_dictionary_names)
            self.face_dictionary_feats = np.concatenate(self.face_dictionary_feats, axis=0)
        
    def _UTILS_read_processed_videos(self):
        # read the videos that have been processed
        
        self.videos = [f for f in os.listdir(self.path_to_vids) if os.path.isdir(os.path.join(self.save_path,f[:-4]))]
          
    def _UTILS_read_detection_meta_data(self, file_path):
        # read the average detection size and duration for each track
        track_meta_data = []
        with open(file_path) as f:
            lines = f.readlines()
            lines = [x.strip().split(',') for x in lines]
        
        if len(lines) > 0:
            track_ID = lines[0][1]
            areas = []
            duration = 0
            for i in range(len(lines)):
                line = lines[i]
                track_ID_new = line[1]
                if track_ID_new == track_ID:
                    duration += 1
                    areas.append(float(line[4])*float(line[5]))
                
                if (i == len(lines)-1) or  track_ID_new != track_ID:

                    track_meta_data.append(np.array([[duration, np.mean(areas)]]))
                    areas = []
                    duration = 0
                    track_ID = track_ID_new
        
        return np.concatenate(track_meta_data,0)
    
    def _UTILS_ignore_tracks(self, track_meta_data):
        # only use tracks longer than a certain hyper-param
        
        return (track_meta_data[:,0] >= self.min_track_duration).astype(int)
    
    def _UTILS_read_video_tracks(self, video):
        # read the video face track data, and also at this point choose to ignore some tracks 
        
        # ------------------------------------------------
        # read the track features and data
        # ------------------------------------------------
        if not os.path.isfile(os.path.join(self.save_path, video, video+'_TrackFeats.pk')):
            raise Exception('cannot find track features for this video')
        with open(os.path.join(self.save_path, video, video+'_TrackFeats.pk'),'rb') as f:
            track_features = np.concatenate([*pickle.load(f).values()],0)
        
        if not os.path.isfile(os.path.join(self.save_path, video, video+'.txt')):
            raise Exception('cannot find track detections for this video')
        track_meta_data = self._UTILS_read_detection_meta_data(os.path.join(self.save_path, video, video+'.txt'))

        # ------------------------------------------------
        # ignore misc tracks
        # ------------------------------------------------
        
        tracks_to_use = self._UTILS_ignore_tracks(track_meta_data)
        
        return track_features, tracks_to_use
        
    def _UTILS_expand_face_dict(self, annotations, video_track_features):
        # create a new face dictionary for a specific video, where the features are
        # aggregated with the features from annotations already made. This is a form
        # of "query expansion" which will move the features in the dictionary closer
        # to the domain of the videos being annotated
        
        # ------------------------------------------------
        # create a feature dictionary with features of the annotated tracks for each name 
        # (same size as self.face_dictionary_feats)
        # ------------------------------------------------
        expanded_face_dict_feats = np.zeros_like(self.face_dictionary_feats)
        annotation_locations = np.where( (np.expand_dims(annotations,-1) == np.expand_dims(self.face_dictionary_names,0).repeat(annotations.shape[0],0)).astype(int)==1)
        for ID_ind in range(self.face_dictionary_feats.shape[0]):
            # get the annotated features for this identity in the face dict
            if np.sum(annotation_locations[1] == ID_ind) > 0:
                expanded_face_dict_feats[ID_ind] = ImageProcessor.tracker.AverageVectorfunc(video_track_features[annotation_locations[0][np.where(annotation_locations[1]==ID_ind)[0]]])
            else:
                expanded_face_dict_feats[ID_ind] = self.face_dictionary_feats[ID_ind]

        # ------------------------------------------------
        # average pool this with the original face feature dictionary + L2 normalise
        # ------------------------------------------------
        expanded_face_dict_feats = ((self.face_dictionary_feats + expanded_face_dict_feats) / np.expand_dims(np.linalg.norm((self.face_dictionary_feats + expanded_face_dict_feats),axis=1),1))
        
        return expanded_face_dict_feats
        
    def _ANNOTATE_tracks(self, video_track_features, face_dict_features=None):
        # annotate the video track feautures using the face bank features
        
        # ------------------------------------------------
        # compute similarity between track features and feature bank
        # ------------------------------------------------
        if face_dict_features is None:
            # then use the original computed face dictionary features
            face_dict_features = self.face_dictionary_feats
        
        similarities = np.dot(face_dict_features, np.transpose(video_track_features))

        # ------------------------------------------------
        # find any verification matches
        # ------------------------------------------------
        feat_bank_matches = np.argmax(similarities,0)
        threshold_match =  (np.max(similarities,0) > self.face_verification_threshold).astype(int)
        
        # ------------------------------------------------
        # fill the annotation data
        # ------------------------------------------------        
        return np.char.multiply(self.face_dictionary_names[feat_bank_matches],threshold_match)
        
    def run(self):
        
        # ------------------------------------------------
        # read the image face recognizer outputs
        # ------------------------------------------------
        self._UTILS_read_image_face_recognition()
        
        # ------------------------------------------------
        # read the processed videos
        # ------------------------------------------------
        self._UTILS_read_processed_videos()
            
        # ------------------------------------------------
        # for each of the videos
        # ------------------------------------------------
        
        for video in self.videos:
            
            # ------------------------------------------------
            # read the video data into some useuful form 
            # ------------------------------------------------
            video_track_features, tracks_to_use = self._UTILS_read_video_tracks(video[:-4])
             
            # ------------------------------------------------
            # annotate using face
            # ------------------------------------------------
            annotations = self._ANNOTATE_tracks(video_track_features)

            # ------------------------------------------------
            # query expansion
            # ------------------------------------------------
            # (1) update the face dictionary with the features from the annotaion face-tracks
            query_expanded_face_dict = self._UTILS_expand_face_dict(annotations, video_track_features) 
            # (2) re-annotate
            annotations = self._ANNOTATE_tracks(video_track_features, face_dict_features=query_expanded_face_dict)
           

            # ------------------------------------------------
            # optionally make an annotation video
            # ------------------------------------------------
            
            if self.make_annotation_video:
                utils.MakeVideo(video, os.path.join(self.temp_dir, video[:-4]), os.path.join(self.save_path,video[:-4]), os.path.join(self.path_to_vids,video), annotations=annotations)
            
            