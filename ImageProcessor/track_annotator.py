import os 
import pdb
import pickle

class TrackAnnotator:
    
    def __init__(self, save_path, 
                 path_to_vids,
                 path_to_input,
                 face_verification_threshold=0.7, 
                 query_expansion_threshold=0.7,
                 only_use_non_outlier_faces=True):
        
        self.face_verification_threshold = face_verification_threshold
        self.query_expansion_threshold = query_expansion_threshold
        self.only_use_non_outlier_faces = only_use_non_outlier_faces
        
        # ------------------------------------------------
        # read the image face recognizer outputs
        # ------------------------------------------------
        self._read_image_face_recognition(path_to_input, save_path)
        
        # ------------------------------------------------
        # read the video face tracker outputs
        # ACTUALLY I THINK DO THIS FOR EACH INDIVIDUAL VIDEO 
        # SO THAT ALL THE DATA CAN BE STORED IN RAM FOR MAKING THE VIDEO AFTER
        # ------------------------------------------------
        self._read_video_face_tracks(path_to_vids, save_path)
        
    
    def _read_video_face_tracks(self, path_to_vids, save_path):
        # read the video face track data, and also at this point choose to ignore some tracks 
        
        # ------------------------------------------------
        # read the track features and data
        # ------------------------------------------------
        
        
        # ------------------------------------------------
        # ignore misc tracks
        # ------------------------------------------------
        
        
        
        # ------------------------------------------------
        # aggregate the features
        # ------------------------------------------------
        None
        
        
    def _read_image_face_recognition(self, path_to_input, save_path):
        # read the image face recognition data into features, depending on 
        # if outliers are to be considered or not
        
        # ------------------------------------------------
        # read the features
        # ------------------------------------------------
        
        people = [f + '.pk' for f in os.listdir(path_to_input) if os.path.isfile(os.path.join(save_path,f+'.pk'))]
        self.face_dictionary_names = []
        self.face_dictionary_feats = []
        for person in people:
            # read the data 
            with open(os.path.join(save_path,person),'rb') as f:
                face_data = pickle.load(f)
                if not self.only_use_non_outlier_faces:
                    # then use all features from the face images 
                    self.face_dictionary_names.append(person[:-3])
                    self.face_dictionary_feats.append(face_data['aggregated_feature_all'])
                elif data['famous']:
                    # then only use the non-outlier features for the non-outlier identities
                    self.face_dictionary_names.append(person[:-3])
                    self.face_dictionary_feats.append(face_data['aggregated_feature_without_outliers'])
        
        # ------------------------------------------------
        # aggregate the features to single matrix
        # ------------------------------------------------
                 
       if len(self.face_dictionary_names) == 0:
           raise Exception('No identities to annotate with')
       else:
           self.face_dictionary_feats = np.concatenate(self.face_dictionary_feats, axis=0)
        
        
    def run(self, vieo_face_tracks, ID_dictionary):
        
        
        # ------------------------------------------------
        # for each of the videos
        # ------------------------------------------------
        
        
        # ------------------------------------------------
        # annotate using face
        # ------------------------------------------------
        
        
        # ------------------------------------------------
        # query expansion
        # ------------------------------------------------
        
        # ------------------------------------------------
        # optionally make an annotation video
        # ------------------------------------------------
        
        
        