# Automated_Face_Tracking_and_Labelling
Code for the paper "Automated Video Labelling: Identifying Faces by Corroborative Evidence"

1) finish off the tracking code 

a) use Smooth-AP (and cite it at the bottom)

2) write the ImageFaceRecogniser - this is an object that will go from noisey image dir to aggregated face features (with returning all of the metadata and potentially not doing the cleaning if not needed, and can batch if needed - because with web images the batch size is 1

- main work here is on the famous not-famous code which needs a lot of cleaning and re-writing 

3) Full pipeline code 

- read the outputs from both the above 2, and start writing the annotator 

- the first step of the annotator is to get the inputs in the correct form

- this has to have the query expansion part

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
