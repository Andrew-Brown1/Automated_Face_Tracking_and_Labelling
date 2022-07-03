import os
import cv2
import random
import pdb
from tqdm import tqdm
from utils import extract_frames_from_video

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

def MakeSquare(det,img, extension=0.65, ex2=0.4):

    width = det[2] - det[0]
    height = det[3] - det[1]
    # Length = (width + height) / 2
    centrepoint = [int(det[0]) + (width / 2), int(det[1]) + (height / 2)]
    x1 = int(centrepoint[0] - int((1 + extension + ex2) * width / 2))
    y1 = int(centrepoint[1] - int((1 + extension) * height / 2))
    x2 = int(centrepoint[0] + int((1 + extension + ex2) * width / 2))
    y2 = int(centrepoint[1] + int((1 + extension) * height / 2))
    x1 = max(1, x1)
    y1 = max(1, y1)
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])

    bbox = [x1, y1, x2, y2]

    centre_y = (bbox[3] + bbox[1]) / 2
    centre_x = (bbox[2] + bbox[0]) / 2
    if (bbox[3] - bbox[1]) < (bbox[2] - bbox[0]):
        short_side = bbox[3] - bbox[1]
        side = bbox[2] - bbox[0]
        ratio = side / short_side
        y_min = max(0, centre_y - ((centre_y - bbox[1]) * ratio))
        y_max = min(img.shape[0], centre_y + ((bbox[3] - centre_y) * ratio))
        x_min = max(0, bbox[0])
        x_max = min(img.shape[0], bbox[2])

    else:
        short_side = bbox[2] - bbox[0]
        side = bbox[3] - bbox[1]
        ratio = side / short_side
        x_min = max(0, centre_x - ((centre_x - bbox[0]) * ratio))
        x_max = min(img.shape[1], centre_x + ((bbox[2] - centre_x) * ratio))
        y_min = max(0, bbox[1])
        y_max = min(img.shape[1], bbox[3])

    [x1, y1, x2, y2] = [int(x_min), int(y_min), int(x_max), int(y_max)]

    return [x1, y1, x2, y2]



def MakeVideo(Video, temp_directory, ResultsDirectory, original_video_path, fps=None, annotations=None):


    # check that the frames are extracted - if not, then extract 

    if not os.path.isdir(os.path.join(temp_directory,Video)):
        fps = extract_frames_from_video(original_video_path,temp_directory, Video )
    

    TrackColourDictionary = {}
    with open(os.path.join(ResultsDirectory, Video[:-4]+'.txt')) as f:
        FileLines = f.readlines()

    FileLines = [x.strip() for x in FileLines]

    for ind, LineEntry in enumerate(tqdm(FileLines)):
        
        try:

            Entry = LineEntry.split(',')
            Frame = "%06d.jpg"%int(Entry[0])
            image = cv2.imread(os.path.join(temp_directory, Video, Frame))

            rect = [int(float(Entry[2])),int(float(Entry[3])),int(float(Entry[2]))+int(float(Entry[4])),int(float(Entry[3]))+int(float(Entry[5]))]

            TrackID = Entry[1]
            if not TrackID in TrackColourDictionary.keys():
                TrackColourDictionary[TrackID] = (round(random.random() * 255), round(random.random() * 255), round(random.random() * 255))
                #TrackColourDictionary[TrackID] = (0,255,255)
            TrackColour = TrackColourDictionary[TrackID]

            rect = expandrect(rect, 0.6, image.shape)
            
            image = cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), TrackColour, 7)
            image = cv2.putText(image, str(Entry[1]), (int(rect[0]) + 30, int(rect[1]) + 50), 0, 1, (0, 255, 0), 3)
            if annotations is not None:
                name = annotations[int(TrackID)]
                image = cv2.putText(image, name, (int(rect[0]) + 30, int(rect[1]) + 80), 0, 1, (0, 255, 0), 3)
                
            cv2.imwrite(os.path.join(temp_directory, Video, Frame), image)

        except:
            pdb.set_trace()
    
    os.mkdir(os.path.join(temp_directory, 'videos'))
    
    # now make the video from the frames.
    os.system('ffmpeg -r '+str(fps)+' -start_number 0 -i ' + os.path.join(temp_directory, Video) + '/%06d.jpg -vf fps='+str(fps)+' ' + os.path.join(temp_directory,'videos', Video))
    
    # extract the audio
    os.system("ffmpeg -i " + original_video_path +" "+ os.path.join(temp_directory,'videos','audio.mp3'))
    
    # begin a convoluted a probably not necessary process to result in an mp4
    # add the audio to the video - making a .mkv
    os.system("ffmpeg -i "+os.path.join(temp_directory, 'videos', Video) + " -i "+os.path.join(temp_directory,'videos','audio.mp3')+" -shortest " + os.path.join(temp_directory,'videos', Video[:-4]+ '.mkv'))
    
    # turn the mkv into an mp4
    if annotations is None:
        os.system('ffmpeg -i '+os.path.join(temp_directory, 'videos',Video[:-4]+ '.mkv')+' -c copy -c:a aac -strict -2 ' + os.path.join(ResultsDirectory, Video))
    else:
        os.system('ffmpeg -i '+os.path.join(temp_directory, 'videos',Video[:-4]+ '.mkv')+' -c copy -c:a aac -strict -2 ' + os.path.join(ResultsDirectory, Video[:-4] + '_annotated.mp4'))



