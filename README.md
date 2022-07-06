# Automated Face Tracking and Labelling
This repository contains code for the paper "Automated Video Labelling: Identifying Faces by Corroborative Evidence" (MIPR 2021). This includes code for annotating face-tracks in videos with names, given the videos, and face-images (can include outliers) as inputs. 


**Summary**
---
This repository contains the code for:

+ **Video Face Tracking**
    - Given videos as input, compute face-tracks and ID representing features for all faces in videos
+ **Image Face Recognising**
    - Given face images as input, detect faces and compute ID representing features. Additionally remove outliers
+ **Video Face Annotating**
    - Full Pipeline. Given videos and face images as input, annotate the face-tracks in the videos with the names of the people in the face images (as demonstrated in the video above)


**setup**
---
1) Clone this repository
2) in the parent directory, create the directory "save_dir/"
3) in the parent directory, create the directory "temporary_dir"
4) in the parent directory, create the directory "weights_and_data/"
5) Download the folder "weights" from this Google Drive, and place it in "weights_and_data/"

**Video Face Annotating**
---

The following command will 
+ compute face-tracks and extract features for the faces in the videos placed in "weights_and_data/videos"
+ detect faces and extract features for the directories of images placed in "weights_and_data/face_images"
+ annotate the face-tracks in the videos with the names of the face-image directories
+ save outputs to "save_dir"

```
python VideoFaceAnnotator.py
```
+ For creating a video visualising the annotations and face-tracks (like the one at the top of this README), run the following command
```
python VideoFaceAnnotator.py --make_annotation_video True
```
+ The above commands presume that the images in "face_images" are noisey. If you know them to be clean, then run:
```
python VideoFaceAnnotator.py --make_annotation_video True 
```
**Video Face Annotating Example**
---
+ For creating the video at the top of this README:
    - download the "videos" directory and place it in "weights_and_data"
    - download the "face_images" directory and place it in "weights_and_data"
+ Run the following command, the video will be saved as "../save_dir/out/out_annotated.mp4"
```
python VideoFaceAnnotator.py --make_annotation_video True
```



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


**Video Face Tracking**
---



**Image Face Recognising**
---








**Summary**
---


(1) 

1) finish off the tracking code 

a) use Smooth-AP (and cite it at the bottom) (just need to verify that it works)

2) Annotation Code

(a) make the video - this involves actually re-extracting the frames which is a bit dumb

(b) debug - something isn't quite right with the query expansion - check this out (CHANGE THE DOWN-RES)

c) make video 

3) find a good demo video and show it in the README


4) processing scripts AND readme

(i) links to download images, videos, weights

(a) full pipeline for everything

(b) individual pipelines for:

(i) tracking and recognition on videos

(ii) detect and recognise directory

(6) README with all this information

