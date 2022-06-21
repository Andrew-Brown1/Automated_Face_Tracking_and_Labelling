# Automated_Face_Tracking_and_Labelling
Code for the paper "Automated Video Labelling: Identifying Faces by Corroborative Evidence"

1) finish off the tracking code 

a) use Smooth-AP (and cite it at the bottom)

2) get the code to go from the tracking outputs to the inputs that the annotator takes (in some BBC ICMR folder somewhere)

3) code to go from directory of messy person images, to "clean" face vectors. 

(a) make it a feature that the repo can also just detect faces and extract feats from directory of images without the "cleaning"

4) annotation code, which annotates the videos with the names from the directories, including the query expansion

5) processing scripts

(a) full pipeline of everything from directories of images and videos 

(b) individual pipelines for:

(i) tracking and recognition

(ii) detect and recognise directory

(6) README with all this information


## Paper

If you find this repository useful, please consider citing:

```
@InProceedings{Brown21,
    title={Automated Video Labelling: Identifying Faces by Corroborative Evidence},
    author={Andrew Brown and Ernesto Coto and Andrew Zisserman},
    year={2021},
    booktitle={Multimedia Information Processing and Retrieval (MIPR)}
}

@InProceedings{Brown20,
  author       = "Andrew Brown and Weidi Xie and Vicky Kalogeiton and Andrew Zisserman ",
  title        = "Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval",
  booktitle    = "European Conference on Computer Vision (ECCV), 2020.",
  year         = "2020",
}
```
