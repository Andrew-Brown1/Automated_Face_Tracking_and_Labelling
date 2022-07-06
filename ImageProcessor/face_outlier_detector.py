from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pdb 


class FaceOutlierDetection:
    """this object has the functionality of determining the most common 'class' in a set of features using 
    clustering
    """

    def __init__(self, famous_threshold=30, distance_threshold=0.34):
        
        self.famous_threshold = famous_threshold
        self.clusterer = AgglomerativeClustering(affinity='cosine', linkage='single', distance_threshold=distance_threshold,
                                             n_clusters=None)
        
    def cluster(self, features):
        # cluster the features, return the labels, a bool for whether the largest class
        # is larger than a threshold, and the label for the most frequent class
        
        clustering = self.clusterer.fit(features)
        labels = clustering.labels_
        unique, counts = np.unique(labels, return_counts=True)
        
        return labels, np.max(counts) >= self.famous_threshold, unique[np.argmax(counts)]
    
    def run(self, trackinfo):

        
        # (1) cluster the first 100 to see whether the person is there is one "dominant class"
        _, has_dominant_class, _ = self.cluster(trackinfo['Features'][:100])    

        # (2) cluster all the features
        labels, _, dominant_class = self.cluster(trackinfo['Features'])   
                
        # (3) return an array with the labels, and the boolean label 
        return has_dominant_class, (labels==dominant_class).astype('int')
