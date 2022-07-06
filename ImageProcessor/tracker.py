import os
import scipy.io
import numpy as np
from scipy.optimize import linear_sum_assignment
import math
from sklearn import preprocessing
import pickle
from scipy.signal import medfilt
import numpy
import math
import itertools


def Track(Info, save_path, num_zeros=6, MatchThreshold=0.3, LinkThreshold=0.85, RemoveSmallTracks=True, InterpDistance=5):
    
    max = 0
    
    for image in Info['ImageNames']:
        if int(image[:-4]) > max:
            max = int(image[0:-4])
    LastImage = max

    Matches = []

    # iterate through the frames in sequential pairs 
    for index in range(0, LastImage):

        # find the detections in frame A and B
        try:
            ImageA = ('%0'+str(num_zeros)+'d.jpg') % index
            ImageB = ('%0'+str(num_zeros)+'d.jpg') % (index + 1)
        except:
            pdb.set_trace()
        ImageAFeats = []
        ImageAROI = []
        ImageBFeats = []
        ImageBROI = []
        ImageA_inds = []
        ImageB_inds = []
        for im_ind, im in enumerate(Info['ImageNames']):

            if im == ImageA:
                ImageAFeats.append(Info['Features'][im_ind])
                ImageAROI.append([Info['x'][im_ind], Info['y'][im_ind], Info['w'][im_ind], Info['h'][im_ind]])
                ImageA_inds.append(im_ind)
            if im == ImageB:
                ImageBFeats.append(Info['Features'][im_ind])
                ImageBROI.append([Info['x'][im_ind], Info['y'][im_ind], Info['w'][im_ind], Info['h'][im_ind]])
                ImageB_inds.append(im_ind)
                
        # assemble the matrix of scores between them (scores = combination of IOU and feature similarity)
        if ImageAFeats or ImageBFeats:

            ScoreMatrix = np.zeros((len(ImageAFeats) + len(ImageBFeats), len(ImageAFeats) + len(ImageBFeats)))
            ScoreMatrix[:, :] = MatchThreshold

            for A_ind, FeatA in enumerate(ImageAFeats):
                for B_ind, FeatB in enumerate(ImageBFeats):
                    Cost, Overlap = IOUCost(ImageAROI[A_ind], ImageBROI[B_ind], 0.3, 0.5)
                    ScoreMatrix[A_ind, B_ind] = np.dot(FeatA, np.transpose(FeatB)) * Cost
                    # add in the distance measurement. Dependant upon IOU. Currently is simple in that
                    # if the IOU is below threshold cost = 0, otherwise cost = 1

            ScoreMatrix = -ScoreMatrix

            row_ind, col_ind = linear_sum_assignment(ScoreMatrix)

            # find connections

            for ind, row_element in enumerate(row_ind):

                if row_element <= len(ImageA_inds) - 1 and col_ind[ind] <= len(ImageB_inds) - 1:
                    Matches.append([ImageA_inds[row_element], ImageB_inds[col_ind[ind]]])

    # now look through the matches and link together the tracks that share a detection.
    Tracks = []

    while Matches:
        NewTrack = []
        NewTrackInds = []

        MatchToTrack = Matches[0]
        NewTrack.append(MatchToTrack)
        NewTrackInds.append(0)

        for match_ind, match in enumerate(Matches):

            if match[0] == MatchToTrack[1]:
                NewTrack.append(match)
                NewTrackInds.append(match_ind)

                MatchToTrack = match

        Tracks.append(NewTrack)
        NewMatches = []
        for match_ind, match in enumerate(Matches):
            if match_ind not in NewTrackInds:
                NewMatches.append(match)

        Matches = NewMatches

    for trackind, Track in enumerate(Tracks):
        Indices = []
        for Pair in Track:
            for Element in Pair:
                if Element not in Indices:
                    Indices.append(Element)

        Tracks[trackind] = Indices


    # Interpolation over missing detections
    
    NewTracks = []
    for Track in Tracks:
        if Track:
            NewTracks.append(Track)

    Tracks = NewTracks

    Tracks, Info = InterpolateMissingFrames(Tracks, Info, InterpDistance, num_zeros)

    # here remove the temporally short tracks (likely noise) if specified by the user

    if RemoveSmallTracks:
        NewTracks = []
        for Track in Tracks:
            if len(Track) > 5:
                NewTracks.append(Track)
        Tracks = NewTracks

    # here analyse the confidences of each of the tracks

    Confidences = []
    TrackConfidences = []
    for Track in Tracks:
        Temp = []
        for element in Track:
            Temp.append(Info['conf'][element])
        Confidences.append(Temp)
        TrackConfidences.append(np.mean(Temp))

    NewTracks = []
    for ind, Track in enumerate(Tracks):
        if TrackConfidences[ind] > 0.45:
            NewTracks.append(Track)
    Tracks = NewTracks
    
    # write the tracks

    WriteTracks(Tracks, Info, save_path, num_zeros=num_zeros)


def WriteTracks(Tracks, Info, save_path, OriginalNames = None, num_zeros=5):
    # write the tracks to an interpretable format

    # make a list with information about each detection.

    tempfeat = []
    track_features = {}
    DetectionInfo = []     # frame number, trackID, x, y, w, h, 1, -1, -1
    for TrackID, Track in enumerate(Tracks):
        smoothx1 = []
        smoothy1 = []
        smoothx2 = []
        smoothy2 = []

        for Element in Track:

            smoothx1.append(Info['x'][Element])
            smoothy1.append(Info['y'][Element])
            smoothx2.append(Info['x'][Element]+Info['w'][Element])
            smoothy2.append(Info['y'][Element]+Info['h'][Element])

        if len(smoothx1) > 10:
            smoothx1 = smooth(np.asarray(smoothx1),4,'hamming')
            smoothy1 = smooth(np.asarray(smoothy1),4,'hamming')
            smoothx2 = smooth(np.asarray(smoothx2),4,'hamming')
            smoothy2 = smooth(np.asarray(smoothy2),4,'hamming')

        for index, Element in enumerate(Track):
            FrameNumber = int(Info['ImageNames'][Element][:-4])

            tempfeat.append(Info['Features'][Element])

            DetectionInfo.append([FrameNumber, TrackID, Info['x'][Element], Info['y'][Element],
                                  Info['w'][Element], Info['h'][Element], 1, -1, -1, Info['conf'][Element], Info['Features'][Element],
                                  [smoothx1[index],smoothy1[index],smoothx2[index]-smoothx1[index],smoothy2[index]-smoothy1[index]]])

        track_features[TrackID] = AverageVectorfunc(tempfeat)

        tempfeat = []

    output_tracks = []
    
    with open(save_path[:-4]+'_face_detections.txt', 'w+') as f:

        for Detection in DetectionInfo:
            Det = Detection
            f.write(
                str(Det[0]) + ',' + str(Det[1]) + ',' + str(Det[2]) + ',' + str(Det[3]) + ',' + str(Det[4]) + ',' + str(
                    Det[5])
                + ',1,-1,-1,' + str(Det[9])+','+str(Det[11][0])+','+str(Det[11][1])+','+str(Det[11][2])+','+str(Det[11][3]))
            f.write('\n')
            output_tracks.append(Det[10])
            
    with open(save_path[:-4]+'_face_track_aggregated_features.pk','wb') as f:
        pickle.dump(track_features, f)


def InterpolateMissingFrames(Tracks, Info, InterpDistance, num_zeros):


    TrackIms = []
    TrackMatches = []
    for Track in Tracks:
        ImagesInTrack = []
        for element in Track:
            ImagesInTrack.append(Info['ImageNames'][element])
        TrackIms.append(ImagesInTrack)
    maxInd = len(Info['x'])

    # for each track, look through the frames, and see if there are any gaps that are less
    # than the threshold.
    for TrackNo, Track in enumerate(TrackIms):
        for index in range(0,len(Track)-1):

            if int(Track[index+1][0:-4])-int(Track[index][0:-4]) <= InterpDistance and int(Track[index+1][0:-4])-int(Track[index][0:-4]) > 1:
                # interpolate missing frames.
                NumberMissing = int(Track[index+1][0:-4])-int(Track[index][0:-4])
                rect1 = [Info['x'][Tracks[TrackNo][index]], Info['y'][Tracks[TrackNo][index]], Info['x'][Tracks[TrackNo][index]]+Info['w'][Tracks[TrackNo][index]], Info['y'][Tracks[TrackNo][index]]+Info['h'][Tracks[TrackNo][index]]]
                rect2 = [Info['x'][Tracks[TrackNo][index+1]], Info['y'][Tracks[TrackNo][index+1]], Info['x'][Tracks[TrackNo][index+1]]+Info['w'][Tracks[TrackNo][index+1]], Info['y'][Tracks[TrackNo][index+1]]+Info['h'][Tracks[TrackNo][index+1]]]
                frame1 = int(Track[index][0:-4])

                for ind in range(1,NumberMissing):
                    Newx1 = (((rect2[0] - rect1[0]) / NumberMissing) * ind) + rect1[0]
                    Newy1 = (((rect2[1] - rect1[1]) / NumberMissing) * ind) + rect1[1]
                    Newx2 = (((rect2[2] - rect1[2]) / NumberMissing) * ind) + rect1[2]
                    Newy2 = (((rect2[3] - rect1[3]) / NumberMissing) * ind) + rect1[3]
                    NewFrame = frame1+ind

                    # append to track

                    Tracks[TrackNo].extend([maxInd])
                    Info['x'] = np.append(Info['x'], Newx1)
                    Info['y'] = np.append(Info['y'], Newy1)
                    Info['w'] = np.append(Info['w'], Newx2-Newx1)
                    Info['h'] = np.append(Info['h'], Newy2-Newy1)
                    Info['conf'] = np.append(Info['conf'], 0.6)
                    Info['ImageNames'].append(("%0"+str(num_zeros)+"d.jpg")%NewFrame)
                    Info['Features'].append(Info['Features'][Tracks[TrackNo][index]])
                    maxInd += 1


    return Tracks, Info


# ==========================================================================================================================
#                                                 utility functions
# ==========================================================================================================================

def AverageVectorfunc(SetOfVectors):

    AverageVector = []

    for element in range(len(SetOfVectors[0])):
        TestElement = 0
        for vector in SetOfVectors:
            TestElement += vector[element]
        AverageVector.append(TestElement)

    Array = np.asarray(AverageVector).reshape(1,-1)
    Normed = preprocessing.normalize(Array,norm='l2')
    return Normed

def IntersectionOverUnion(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2]+boxA[0], boxB[2]+boxB[0])
    yB = min(boxA[3]+boxA[1], boxB[3]+boxB[1])

    # compute the area of intersection rectangle

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def IOUCost(A_ROI, B_ROI, Threshold, Threshold2=None):
    # assigns a cost dependant on how close the boxes are

    IOU = IntersectionOverUnion(A_ROI, B_ROI)

    if IOU <= Threshold:
        Cost = 0
    else:
        Cost = 1
    if Threshold2:
        if IOU > Threshold and IOU < Threshold2:
            Cost = 0.5



    return Cost, IOU

def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise Exception("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise Exception("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise Exception("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y