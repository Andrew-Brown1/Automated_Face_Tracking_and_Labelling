# Automated_Face_Tracking_and_Labelling
Code for the paper "Automated Video Labelling: Identifying Faces by Corroborative Evidence"

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


## Paper

If you find this repository useful, please cite:

```
(a) the video face tracking and annotation method:

@InProceedings{Brown21,
    title={Automated Video Labelling: Identifying Faces by Corroborative Evidence},
    author={Andrew Brown and Ernesto Coto and Andrew Zisserman},
    year={2021},
    booktitle={Multimedia Information Processing and Retrieval (MIPR)}
}

(b) the deep representation used:

@InProceedings{Brown20,
  author       = "Andrew Brown and Weidi Xie and Vicky Kalogeiton and Andrew Zisserman ",
  title        = "Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval",
  booktitle    = "European Conference on Computer Vision (ECCV), 2020.",
  year         = "2020",
}
```
