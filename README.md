# Multimodal Feature Extractors

This repo contains a collection of feature extractors for multimodal emotion recognition.

Currently, these modalities are covered:
1. [Video](#video)


---

### Video
#### OpenFace + Face VGG:
This feature extractor contains

###### Setup

Clone this repository:

`$ git clone --recurse-submodules https://github.com/gangeshwark/multimodal_feature_extractors.git`

1. Install FFMPEG and OpenCV from source.
2. Install the packages as specified in requirements.txt
Then use:

`from src.video.models import OpenFace_VGG` in your data processing code.

---


### TODO:
1. Add a general video feature extractor.
2. Add text feature extractor.
3. Add audio feature extractor.



### Credits:
1. Authors of [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow), [openface](https://github.com/cmusatyalab/openface)