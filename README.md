# Automated Face Tracking and Labelling
This repository contains code for the paper ["Automated Video Labelling: Identifying Faces by Corroborative Evidence"](https://arxiv.org/abs/2102.05645) (MIPR 2021 - winner of **_Best Student Paper_**). Given inputs of (1) person images containing outliers, and (2) a video, we present a method for computing and annotating face-tracks in the video with the names of the people in the images, as shown in the example output video below.

Input - Person Images (containing outliers)   
:-------------------------:
![face_images](https://user-images.githubusercontent.com/44160842/177529261-2b57e30a-6af8-47c0-bba5-545bef2465dc.jpg)

Input - Video             |  Output - Annotated Video
:-------------------------:|:-------------------------:
![MF_OG](https://user-images.githubusercontent.com/44160842/177526502-509af5ce-37a6-4207-ade1-b49ca398a1b0.gif)  |  ![annotated_gif](https://user-images.githubusercontent.com/44160842/177524977-7bedc208-41dc-4253-b619-e0c8b6b9eaac.gif)

**Summary**
---
This repository contains the code for automatically tracking and annotating faces in videos, given a set of directories containing images of people that could have been collected from image search engines (and so might include outliers):

+ **Video Face Annotating**
    - Given videos and face images as input, annotate the face-tracks in the videos with the names of the people in the face images (as demonstrated in the video above). This involves: 
        * **Video Face Tracking** - computing the face-tracks for an input video
        * **Face-Image Processing** - given a directory of images, detect and recognise faces, and optionally remove outliers
        * **Face-Track Annotation** - Annotate the face-tracks with the names of the people in the face-images
    


**Setup**
---
1) Clone this repository
2) Create directories: "../save_dir/", "../temporary_dir/", and "../weights_and_data/"
3) Download the "weights" directory from [this Google Drive](https://drive.google.com/drive/folders/180Kx3DH2gvqnMKBIn7baE6vr8KeLXKWM?usp=sharing), and place it in "../weights_and_data/"
3) ```pip install -r requirements.txt```
4) requires ```ffmpeg``` for video processing

**Video Face Annotating - Demo**
---
For creating the video at the top of this README (video will be saved as "../save_dir/MF_vid/MF_vid_annotated.mp4"). 
+ First, download the demo data via ```./utils/download_demo_data.sh```, then run:
```
python VideoFaceAnnotator.py --make_annotation_video True -demo_example True
```

For annotating your own videos
+ place the video(s) to be annotated in "../weights_and_data/videos/"
+ place the directories with person images in "../weights_and_data/person_images/"
+ Run the following:
```
python VideoFaceAnnotator.py --make_annotation_video True
```

The default operation is to assume that the person images contain outliers, which are automatically ignored (see paper). If you know that the face-images do not contain outliers, use the flag ```--only_use_non_outlier_faces False```. For faster processing, do not make the videos i.e. ```--make_annotation_video False```

**Video Face Tracking**
---
For only the video face-tracking functionality, place the video(s) in "../weights_and_data/videos/" and run:
```
python VideoFaceTracker.py
```
Tracks and ID representing features will be written to "../save_dir/". For writing a video with the face-tracks, add ```--make_video True```

**Image Face Recognising**
---
For only the image face recognition functionality, place the directories of images in "../weights_and_data/person_images". This command will detect faces, and save the detections and ID representing features: 
```
python ImageFaceRecognise.py 
```

Pre-Trained Representations
---
In this work, we use pre-trained representations for detecting faces, and extracting ID-discriminating features from face images:
+ for face detection, we use [Retina Face](https://github.com/biubug6/Pytorch_Retinaface)
+ For ID-discriminating face features, we use a [VGGFace2 trained SE-Net50](https://github.com/ox-vgg/vgg_face2)


## Paper

If you find anything in this repository useful, please cite:
```
@InProceedings{Brown21,
    title={Automated Video Labelling: Identifying Faces by Corroborative Evidence},
    author={Andrew Brown and Ernesto Coto and Andrew Zisserman},
    year={2021},
    booktitle={Multimedia Information Processing and Retrieval (MIPR)}
}
```
