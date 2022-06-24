import os
import pickle
import pdb
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from tqdm import tqdm
from os.path import join as osj


class FaceOutlierDetection:
    """this object has the functionality of determining the most common 'class' in a set of features using 
    clustering
    """

    def __init__(self, famous_threshold=30):
        
        self.famous_threshold = famous_threshold
        self.clusterer = AgglomerativeClustering(affinity='cosine', linkage='single', distance_threshold=0.32,
                                             n_clusters=None)
        
        # (a) cluster the first 100 to see whether famous or not 
        
        

            # - this requires getting the ranking of the images into the trackinfo object
        # (b) cluster all the features, then output whether each of them belongs to the main cluster or not 

    def run(self, trackinfo):
        # clustering = AgglomerativeClustering(affinity='cosine', linkage='single', distance_threshold=0.32,
                                            #  n_clusters=None).fit(features)
        pdb.set_trace()                                
        # (1) cluster the first 100
        labels = self.cluster_(track)    
        labels = clustering.labels_
        unique, counts = np.unique(labels, return_counts=True)


        if np.max(counts) > famous_thresh:

            correct_label = unique[np.argmax(counts)]
            correct_feats = []
            correct_inds = []

            for ind, cluster in enumerate(labels):
                if cluster == correct_label:
                    correct_feats.append(features[ind])
                    correct_inds.append(self.indexes[self.aggregate_level][ind])
            assert (len(correct_feats) == np.max(counts))
            self.cluster_size = len(correct_feats)
            aggregated_feature = AverageVectorfunc(correct_feats)
            self.correct_indexes = correct_inds
        else:
            aggregated_feature = None
            correct_feats = None
            self.correct_indexes = None

        return correct_feats, aggregated_feature
