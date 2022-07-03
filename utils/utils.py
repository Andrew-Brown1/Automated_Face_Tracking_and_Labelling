import os
import pdb
import time
import scipy
import cv2
import numpy as np

import inspect


class AttrDict(dict):
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value
            
            
def auto_init_args(obj, tgt=None, can_overwrite=False):
    # autoassign constructor arguments
    frame = inspect.currentframe().f_back  # the frame above
    params = frame.f_locals
    nparams = frame.f_code.co_argcount
    paramnames = frame.f_code.co_varnames[1:nparams]
    if tgt is not None:
        if not can_overwrite:
            assert not hasattr(obj, tgt)
        setattr(obj, tgt, AttrDict())
        tgt_attr = getattr(obj, tgt)
    else:
        tgt_attr = obj
        
    for name in paramnames:
        # print("\t autosetting %s -> %s" % (name, str(params[name])))
        setattr(tgt_attr, name, params[name])
        
        
class Timer:
    def __init__(self):
        self.time = 0
    
    def _start(self, process_name, verbose):
        self.time = time.time()
        if verbose:
            print('starting '+process_name)
            
            
    def _log_end(self, process_name, verbose):
        if verbose:
            print(process_name + ' = '+str(time.time()-self.time))


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:

        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            if ((fullPath[-3:] == 'mkv') or (fullPath[-3:] == 'mp4') or (fullPath[-3:] == 'avi')):
                allFiles.append(fullPath)

    return allFiles


def post_process(detection,scale):
    j = 0
    boxes = []
    scores = []
    while detection[j,0] >= 0.1:
        score = detection[j,0]
        pt = (detection[j, 1:] * scale).cpu().numpy()
        boxes.append([pt[0], pt[1], pt[2], pt[3]])
        scores.append(score.item())
        j += 1
        if j >= detection.size(0):
            break
    return [boxes, scores]

def visualise(track_array, episode, temp_dir):
    None
    if not os.path.isdir(os.path.join(temp_dir,episode)):
        os.mkdir(os.path.join(temp_dir,episode))

    for track_ind, track in enumerate(track_array):
        for frame_ind, frame in enumerate(track):
            scipy.misc.imsave(os.path.join(temp_dir,episode,str(track_ind)+'_'+str(frame_ind)+'.jpg'), frame)


def shots_and_interpolate(Tracks, shots, length):
    # this function divides the tracks by shot and also does the interpolation if necessary

    # first divide by shot

    NewTracks ={}
    trackind = 0
    for track in Tracks:

        for shot in shots:
            NewTrack = []
            for detection in Tracks[track]:
                if (detection[0] > shot['start']) and (detection[0] <= shot['end']):
                    NewTrack.append(detection)

            # if a track doesn't run the whole length then interpolate

            if (not len(NewTrack) == length) and (not len(NewTrack) == 0):
                pdb.set_trace()

                start = NewTrack[0][0]
                end = NewTrack[-1][0]
                for i in range((shot['start']+1),start):

                    x = NewTrack[0]
                    x[0] = i
                    NewTrack = [x] + NewTrack

                for i in range(end, shot['end']):
                    x = NewTrack[-1]
                    x[0] = i+1
                    NewTrack = NewTrack + [x]

                pdb.set_trace()
                None

            if not len(NewTrack) == 0:

                NewTracks[trackind] = NewTrack
                trackind += 1

    return NewTracks

def expandrect(rect, expand_fac, shape):

    centre = ((rect[0] + ((rect[2] - rect[0]) / 2)), (rect[1] + ((rect[3] - rect[1]) / 2)))

    x1 = centre[0] - (((rect[2] - rect[0]) / 2) * (1 + expand_fac))
    y1 = centre[1] - (((rect[3] - rect[1]) / 2) * (1 + expand_fac*0.7))
    x2 = centre[0] + (((rect[2] - rect[0]) / 2) * (1 + expand_fac))
    y2 = centre[1] + (((rect[3] - rect[1]) / 2) * (1 + expand_fac*0.7))

    x1 = max(0,x1)
    x2 = min(shape[1],x2)
    y1 = max(0,y1)
    y2 = min(shape[0],y2)

    return [x1, y1, x2, y2]

def convert_det_to_track(detections):

    Track_info = {}
    Track_info['x'] = []
    Track_info['y'] = []
    Track_info['w'] = []
    Track_info['h'] = []
    Track_info['conf'] = []
    Track_info['ImageNames'] = []

    for frame in detections:

        for ind, det in enumerate(detections[frame][2]):
            Track_info['x'].append(det[0])
            Track_info['y'].append(det[1])
            Track_info['w'].append(det[2]-det[0])
            Track_info['h'].append(det[3] - det[1])
            Track_info['conf'].append(detections[frame][3][ind])
            Track_info['ImageNames'].append(frame)

    return Track_info

def parse_vid(video_path, fr_=None, to_=None):
    vidcap = cv2.VideoCapture(video_path)
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
    height = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    imgs = []
    count = 0
    while True:
        if count >= fr_[0] and count < to_[0]:
            image = vidcap.grab()
            success, image = vidcap.retrieve()
            if not success:
                break
            imgs.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        elif count >= fr_[1] and count < to_[1]:
            image = vidcap.grab()
            success, image = vidcap.retrieve()
            if not success:
                break
            imgs.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif count >= fr_[2] and count < to_[2]:
            image = vidcap.grab()
            success, image = vidcap.retrieve()
            if not success:
                break
            imgs.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        else:

            if count >= to_[2]:
                break
            else:
                _ = vidcap.grab()
        count+=1

    vidcap.release()
    video_np = np.stack(imgs, axis=0)
    return video_np

def compute_array(Tracks, vid, size):

    output = []

    for Track in Tracks:

        # for each frame in the track, resize the face_det and stack

        stack = np.zeros((len(Tracks[Track]),size,size,3))

        for index, detection in enumerate(Tracks[Track]):

            ROI = detection[-1]

            rect = [ROI[0],ROI[1],ROI[0]+ROI[2],ROI[1]+ROI[3]]

            expanded_rect = expandrect(rect, 0.7, vid[0].shape)

            face_det = vid[detection[0]-1][int(expanded_rect[1]):int(expanded_rect[3]),int(expanded_rect[0]):int(expanded_rect[2]),:]

            face = scipy.misc.imresize(face_det,(size,size),interp='bilinear')

            stack[index] = face

        output.append(stack)

    out = np.stack(output, axis=0)
    return out


def extract_frames_from_video(full_episode, temp_dir, temp_file_name):

    
    # (a) find the resolution and fps of the videos
    vid = cv2.VideoCapture(full_episode)
    vid_resolution = [int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    vid_fps = vid.get(cv2.CAP_PROP_FPS)

    # (b) extract the frames (if not done already)    
    if not os.path.isdir(os.path.join(temp_dir, temp_file_name)):
        os.mkdir(os.path.join(temp_dir, temp_file_name))

        Command = "ffmpeg -i " + full_episode + " -threads 1 -deinterlace -q:v 1 -s "+str(vid_resolution[0])+":"+str(vid_resolution[1])+" -vf fps="+str(vid_fps) + " " + temp_dir + "/" + temp_file_name + "/%06d.jpg"
        os.system(Command)

    return vid_fps
