import os
import pickle
import pdb
from sklearn.cluster import AgglomerativeClustering
from data_utils import AverageVectorfunc
import numpy as np
from tqdm import tqdm
from os.path import join as osj
import cv2
import multiprocessing as mp
import editdistance

class Downloaded_Image_Processor:
    """this class looks at the downloaded images, decides whether the person is famous or not, and if it is, cleans the images

    - methods to pre-process the images, classify as famous or not and also to output a single aggregated feature vector
    for that identity
    """

    def __init__(self, pathtotextfiles, pathtofeatures, name, min_detection_size=1800, small_famous_threshold=7, big_famous_threshold = 30, cosine_threshold=0.8, face_det_conf=0.999, write_files=False ):
        self.pathtotextfiles = pathtotextfiles
        self.pathtofeatures = pathtofeatures
        self.write_files = write_files
        self.face_det_conf = face_det_conf
        self.cosine_threshold = cosine_threshold
        self.name = name
        self.min_detection_size = min_detection_size
        self.small_famous_threshold = max(int(np.round(big_famous_threshold*(7/30))),1)
        self.big_famous_threshold = big_famous_threshold

        self.pre_process()
        self.famous_non_famous()

        self.aggregate_features()
        if not self.aggregated_feature is None:
            # take the first 5 correct ones
            self.meta = {}
            self.meta['ROI'] = []
            self.meta['ImageNames'] = []
            tot = 0
            for ind in range(len(self.correct_indexes)):

                if os.path.isfile(os.path.join('/work/abrown/Celebrity_Feature_Bank/Downloaded_People',name,self.big_out['ImageNames'][self.correct_indexes[ind]])):
                    tot += 1
                    self.meta['ROI'].append([int(self.big_out['x'][self.correct_indexes[ind]]),int(self.big_out['y'][self.correct_indexes[ind]]),int(self.big_out['x'][self.correct_indexes[ind]]+self.big_out['w'][self.correct_indexes[ind]]),int(self.big_out['y'][self.correct_indexes[ind]]+self.big_out['h'][self.correct_indexes[ind]])])
                    self.meta['ImageNames'].append(self.big_out['ImageNames'][self.correct_indexes[ind]])

                if tot > 4:
                    break

    def aggregate_features(self):
        if not self.aggregated_feature is None:
            if self.aggregate_level == 'top':
                features = self.features['middle']
                features.extend(self.features['bottom'])
            elif self.aggregate_level == 'middle':
                features = self.features['bottom']
            elif self.aggregate_level == 'bottom':
                features = []
            features = np.asarray(features)

            if not len(features) == 0:

                try:
                    scores = np.dot(self.aggregated_feature,np.transpose(features))
                except:
                    pdb.set_trace()
                    None
                for ind, score in enumerate(scores[0]):

                    if score > self.cosine_threshold:

                        self.all_features.append(features[ind])
                #print("total faces inferred for aggregated feature = "+str(len(self.all_features)))
                aggregated_feature = AverageVectorfunc(self.all_features)
                self.aggregated_feature = aggregated_feature

            self.total_aggregated = len(self.all_features)

    def pre_process(self):
        """pre-processes the downloaded images, removing the detections that are below a certain resolution or confidence

        puts the ranks and features into a dictionary for easy manipulation that is in the appropriate ranks already

        """
        # read the accompanying text files:
        with open(os.path.join(self.pathtotextfiles,self.name+'20.txt')) as f:
            filelines = f.readlines()
            topnames = [x.strip() for x in filelines]
        with open(os.path.join(self.pathtotextfiles,self.name+'100.txt')) as f:
            filelines = f.readlines()
            middlenames = [x.strip() for x in filelines]
        with open(os.path.join(self.pathtotextfiles,self.name+'400.txt')) as f:
            filelines = f.readlines()
            bottomnames = [x.strip() for x in filelines]

        outputs = {}
        indexes = {}
        indexes['top'] = []
        indexes['middle'] = []
        indexes['bottom'] = []
        outputs['top'] = []
        outputs['middle'] = []
        outputs['bottom'] = []
        # read the features and also the
        with open(os.path.join(self.pathtofeatures,self.name+'.pk'), 'rb') as f:
            features = pickle.load(f)
        self.big_out = features

        for ind in range(len(features['x'])):
            # check if the detection is above the minimum size
            if features['conf'][ind] > self.face_det_conf:
                if int(features['w'][ind])*int(features['h'][ind]) > self.min_detection_size:
                    found = False
                    if features['ImageNames'][ind] in topnames:
                        found = True
                        outputs['top'].append(features['Features'][ind])
                        indexes['top'].append(ind)
                    if features['ImageNames'][ind] in middlenames:
                        found = True
                        outputs['middle'].append(features['Features'][ind])
                        indexes['middle'].append(ind)
                    if features['ImageNames'][ind] in bottomnames:
                        found = True
                        outputs['bottom'].append(features['Features'][ind])
                        indexes['bottom'].append(ind)

                    if not found:
                        outputs['bottom'].append(features['Features'][ind])
                        indexes['bottom'].append(ind)
        self.features=outputs
        self.indexes=indexes
        #print('top features = ' + str(len(self.features['top'])))
        #print('middle features = ' + str(len(self.features['middle'])))
        #print('bottom features = ' + str(len(self.features['bottom'])))

    def cluster_features(self, features, famous_thresh):
        clustering = AgglomerativeClustering(affinity='cosine', linkage='single', distance_threshold=0.32,
                                             n_clusters=None).fit(features)
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

    def famous_non_famous(self):
        """takes the 20 top ranked images and then classifies if the person is famous or not using clustering techniques

        need to change this to recursively try each

        """
        self.all_features = None
        self.aggregated_feature = None
        no_images = False
        if len(self.features['top']) > self.small_famous_threshold:
            # if there were images in the top 20 ranks
            self.aggregate_level = 'top'
            features = np.asarray(self.features['top'])
            famous_thresh = self.small_famous_threshold

            correct_feats, aggregated_feature = self.cluster_features(features, famous_thresh)
            self.all_features = correct_feats
            self.aggregated_feature = aggregated_feature

        if (len(self.features['middle']) > self.big_famous_threshold) and (self.aggregated_feature is None):
            self.aggregate_level = 'middle'
            features = np.asarray(self.features['middle'])
            famous_thresh = self.big_famous_threshold

            correct_feats, aggregated_feature = self.cluster_features(features, famous_thresh)
            self.all_features = correct_feats
            self.aggregated_feature = aggregated_feature


        if (len(self.features['bottom']) > self.big_famous_threshold) and (self.aggregated_feature is None):
            self.aggregate_level = 'bottom'
            features = np.asarray(self.features['bottom'])
            famous_thresh = self.big_famous_threshold

            correct_feats, aggregated_feature = self.cluster_features(features, famous_thresh)
            self.all_features = correct_feats
            self.aggregated_feature = aggregated_feature

        if self.aggregated_feature is None:

            self.all_features = None
            self.aggregated_feature = None
            self.correct_indexes = None


def create_feature_alpha(pathtopeople, pathtofeatures, good_names, people, alpha):
    print(alpha)
    feature_bank = {}
    person_count = 0
    count = 0

    print(alpha)
    for person_ind, person in enumerate(tqdm(people)):
        if person.lower() in good_names:
            person_count += 1
            if os.path.isfile(os.path.join(pathtofeatures, person + '.pk')):
                count += 1

                if os.path.isdir(os.path.join(
                        '/scratch/shared/nfs1/abrown/celeb_feat_bank/scratch/shared/beegfs/abrown/Celebrity_Feature_Bank/Downloaded_People',
                        person)):
                    pathtopeople = '/scratch/shared/nfs1/abrown/celeb_feat_bank/scratch/shared/beegfs/abrown/Celebrity_Feature_Bank/Downloaded_People'
                elif os.path.isdir(
                        os.path.join('/scratch/local/hdd/abrown/British_Library_Videos/AllBritishLibraryPeople',
                                     person)):
                    pathtopeople = '/scratch/local/hdd/abrown/British_Library_Videos/AllBritishLibraryPeople'
                else:
                    counter += 1
                    print(person + ' not found: ' + str(counter))

                if os.path.isfile(osj(pathtopeople, person + '20.txt')) and os.path.isfile(
                        osj(pathtopeople, person + '100.txt')) and os.path.isfile(
                        osj(pathtopeople, person + '400.txt')):

                    Processor = Downloaded_Image_Processor(pathtopeople, pathtofeatures, person, write_files=False,
                                                           big_famous_threshold=alpha)

                    aggregatedfeature = Processor.aggregated_feature

                    if not aggregatedfeature is None:
                        # print("famous --> "+person)
                        feature_bank[person] = {}
                        feature_bank[person]['feature'] = aggregatedfeature
                        # feature_bank[person]['all_feats'] = Processor.all_features
                        feature_bank[person]['cluster_size'] = Processor.cluster_size
                        feature_bank[person]['total_aggregated'] = Processor.total_aggregated

                    else:
                        # print("non-famous --> " + person)
                        None

            else:
                pdb.set_trace()

                # with open('../data/feature_bank_mark.pk', 'wb') as f:
                #     pickle.dump(feature_bank, f)
                # pdb.set_trace()

    with open('/scratch/shared/beegfs/abrown/BBC_work/ICMR_BL/Annotate_Tracks/ablation_banks/feature_bank' + str(alpha) + '.pk', 'wb') as f:
        pickle.dump(feature_bank, f)

def create_feature_banks_for_ablation(pathtopeople, pathtofeatures, French=False):
    """create the aggregated features"""
    print('loading names')
    #people = [f for f in os.listdir(pathtopeople) if os.path.isdir(os.path.join(pathtopeople, f))]

    people = [f[:-3] for f in os.listdir(pathtofeatures)]
    counter = 0

    with open('temp/all_names_combined.pk','rb') as f:
        good_names = pickle.load(f)

    pool = mp.Pool(15)
    print('running')
    pool.starmap(create_feature_alpha, [(pathtopeople, pathtofeatures, good_names, people, alpha) for alpha in range(1,101)])

    pool.close()

    # split it into chunks of 100, and send that to parralel workers

def GetIMDBNamesToSearch():
    IMDBTextFiles = ['/scratch/shared/nfs1/abrown/beegfs_back_up/Google_Scraper/GoogleImageScraper/BLTextFiles/BL_CNN_Newsroom_names.txt',
                                 '/scratch/shared/nfs1/abrown/beegfs_back_up/Google_Scraper/GoogleImageScraper/BLTextFiles/The_Alex_Salmond_show_names.txt',
    '/scratch/shared/nfs1/abrown/beegfs_back_up/Google_Scraper/GoogleImageScraper/BLTextFiles/BL_RT.txt',
                     '/scratch/shared/nfs1/abrown/beegfs_back_up/Google_Scraper/GoogleImageScraper/BLTextFiles/new.txt',
                      '/scratch/shared/nfs1/abrown/beegfs_back_up/Google_Scraper/GoogleImageScraper/BLTextFiles/BL_RT_2.txt']

    #IMDBTextFiles = ['BLTextFiles/small.txt']
    Names = []
    for NameFile in IMDBTextFiles:

        with open(NameFile) as f:

            FileLines = f.readlines()

        FileLines = [x.strip() for x in FileLines]

        for name in FileLines:
            Names.append(name.replace(' ','_'))

    Names = list(set(Names))
    return Names

def create_feature_bank(pathtopeople, pathtofeatures, French=False, short=False):
    """create the aggregated features"""
    print('loading names')


    if short:
        people = GetIMDBNamesToSearch()
    else:
        people = [f for f in os.listdir(pathtopeople) if os.path.isdir(os.path.join(pathtopeople, f))]

    # people = [f[:-3] for f in os.listdir(pathtofeatures)]
    counter = 0
    feature_bank = {}

    pdb.set_trace()
    # split it into chunks of 100, and send that to parralel workers
    for person_ind, person in enumerate(tqdm(people)):

        if os.path.isfile(os.path.join(pathtofeatures,person+'.pk')):

    Processor = Downloaded_Image_Processor(pathtopeople,pathtofeatures, person, write_files=False)

    aggregatedfeature = Processor.aggregated_feature

    if not aggregatedfeature is None:
        #print("famous --> "+person)
        feature_bank[person] = {}
        feature_bank[person]['feature'] = aggregatedfeature
        #feature_bank[person]['all_feats'] = Processor.all_features
        feature_bank[person]['cluster_size'] = Processor.cluster_size
        feature_bank[person]['total_aggregated'] = Processor.total_aggregated
        feature_bank[person]['meta'] = Processor.meta
    else:
        #print("non-famous --> " + person)
        None

    with open('/work/abrown/BBC/Russian_Annotations/feature_bank_all.pk','wb') as f:
        pickle.dump(feature_bank,f)

if __name__ == "__main__":
    # create_feature_bank('/scratch/shared/beegfs/abrown/datasets/mediaeval2015/downloaded_images','/scratch/shared/beegfs/abrown/datasets/mediaeval2015/downloaded_images_features', French=True)
    # create_feature_bank('/scratch/shared/nfs1/abrown/celeb_feat_bank/scratch/shared/beegfs/abrown/Celebrity_Feature_Bank/Downloaded_People','/scratch/shared/beegfs/abrown/Celebrity_Feature_Bank/DSFD_Downloaded_People_Feats')

    create_feature_bank('/work/abrown/Celebrity_Feature_Bank/Downloaded_People','/work/abrown/Celebrity_Feature_Bank/DSFD_Downloaded_People_Feats')

