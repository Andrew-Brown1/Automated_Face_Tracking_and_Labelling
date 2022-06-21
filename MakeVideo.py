import os
import cv2
import random
import pdb
from tqdm import tqdm

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

def Make_syncnet(det,img, extension=0.2):

    # first make the bounding box square - this is becuase for profile faces the width can become very small - so
    # making the side length always = to average side length can give bad and fluctuating results
    bbox = [det[0], det[1], det[2], det[3]]

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

    width = x_max - x_min
    height = y_max - y_min

    assert(width == height)

    side_length = width*(1+extension)

    centre_y += height/4 # shift down the centre so it is mouth centric

    x1 = int(centre_x - (side_length / 2))
    x2 = int(centre_x + (side_length / 2))
    y1 = int(centre_y - (side_length / 2))
    y2 = int(centre_y + (side_length / 2))
    x1 = max(1, x1)
    y1 = max(1, y1)
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])

    return [x1, y1, x2, y2]

def MakeVideos(VidNames, OutputDirectory, ResultsDirectory, syncnet=False):

    for Video in VidNames:

        TrackColourDictionary = {}
        with open(os.path.join(ResultsDirectory, Video+'.txt')) as f:
            FileLines = f.readlines()

        FileLines = [x.strip() for x in FileLines]

        for ind, LineEntry in enumerate(tqdm(FileLines)):
            try:

                Entry = LineEntry.split(',')
                Frame = "%06d.jpg"%int(Entry[0])
                image = cv2.imread(os.path.join(OutputDirectory, Video, Frame))

                rect = [int(float(Entry[2])),int(float(Entry[3])),int(float(Entry[2]))+int(float(Entry[4])),int(float(Entry[3]))+int(float(Entry[5]))]

                TrackID = Entry[1]
                if not TrackID in TrackColourDictionary.keys():
                    TrackColourDictionary[TrackID] = (round(random.random() * 255), round(random.random() * 255), round(random.random() * 255))
                    #TrackColourDictionary[TrackID] = (0,255,255)
                TrackColour = TrackColourDictionary[TrackID]

                #image = cv2.imread(os.path.join(OutputDirectory, Video, Frame))

                if not syncnet:
                    rect = expandrect(rect, 0.6, image.shape)
                else:
                    rect = Make_syncnet(rect, image)
              
                image = cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), TrackColour, 7)
                image = cv2.putText(image, str(Entry[1]), (int(rect[0]) + 30, int(rect[1]) + 50), 0, 1, (0, 255, 0), 3)
                cv2.imwrite(os.path.join(OutputDirectory, Video, Frame), image)
            except:
                pdb.set_trace()
                None

        # now make the video from the frames.
        FFMPEGCall = 'ffmpeg -r 25 -start_number 0 -i ' + os.path.join(OutputDirectory, Video) + '/%06d.jpg -c:v libx264 -vf fps=25 -pix_fmt yuv420p ' + os.path.join(ResultsDirectory, Video) + '.mp4'
        os.system(FFMPEGCall)


def MakeDetectionVideos(VidNames, OutputDirectory, ResultsDirectory, syncnet=False):
    for Video in VidNames:

        TrackColourDictionary = {}
        with open(os.path.join(ResultsDirectory, Video + '.txt')) as f:
            FileLines = f.readlines()

        FileLines = [x.strip() for x in FileLines]

        for ind, LineEntry in enumerate(FileLines):
            try:
                if ind % 1000 == 0:
                    print(str(ind / len(FileLines)))

                Entry = LineEntry.split(',')
                Frame = "%05d.jpg" % int(Entry[0])
                image = cv2.imread(os.path.join(OutputDirectory, Video, Frame))

                rect = [int(float(Entry[2])), int(float(Entry[3])), int(float(Entry[2])) + int(float(Entry[4])),
                        int(float(Entry[3])) + int(float(Entry[5]))]

                TrackID = Entry[1]
                if not TrackID in TrackColourDictionary.keys():
                    TrackColourDictionary[TrackID] = (
                    round(random.random() * 255), round(random.random() * 255), round(random.random() * 255))
                    # TrackColourDictionary[TrackID] = (0,255,255)
                TrackColour = TrackColourDictionary[TrackID]

                # image = cv2.imread(os.path.join(OutputDirectory, Video, Frame))

                if not syncnet:
                    rect = expandrect(rect, 0.6, image.shape)
                else:
                    rect = Make_syncnet(rect, image)

                image = cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), TrackColour, 7)
                image = cv2.putText(image, str(Entry[9][:5]), (int(rect[0]) + 30, int(rect[1]) + 50), 0, 1, (0, 255, 0),
                                    3)
                cv2.imwrite(os.path.join(OutputDirectory, Video, Frame), image)
            except:
                pdb.set_trace()
                None

        # now make the video from the frames.
        FFMPEGCall = 'ffmpeg -r 25 -start_number 0 -i ' + os.path.join(OutputDirectory,
                                                                       Video) + '/%05d.jpg -c:v libx264 -vf fps=25 -pix_fmt yuv420p ' + os.path.join(
            ResultsDirectory, Video) + '.mp4'
        os.system(FFMPEGCall)
