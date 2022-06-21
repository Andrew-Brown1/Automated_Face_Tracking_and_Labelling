# Automated_Face_Tracking_and_Labelling
Code for the paper "Automated Video Labelling: Identifying Faces by Corroborative Evidence"

1) get the code for tracking and recognising faces 

a) use Smooth-AP (and cite it at the bottom)

b) do not extract all of the frames (or maybe do, who cares - do not do unecessary extra work, and for very long videos it makes sense because can't load the whole video into RAM)

c) get all model related files into one directory - also try 

d) calls upon several different models - make this clear and link to the GitHubs 

2) clean it up 

3) remove all of the excess code 

4) get it working a little better

5) get the code to go from a directory of messy person images to a "clean" dictionary of face vectors 

6) get the annotation code for annotating the people from these face dictionaries (with query expansion)

7) get it cleaned

8) get a README with all of the informaiton

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
