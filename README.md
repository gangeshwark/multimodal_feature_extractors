# Multimodal Feature Extractors

This repo contains a collection of feature extractors for multimodal emotion recognition.

### Setup

Clone this repository:

`$ git clone --recurse-submodules https://github.com/gangeshwark/multimodal_feature_extractors.git`

1. Install FFMPEG and OpenCV from source.
2. Install the packages as specified in requirements.txt

Currently, these modalities are covered:
1. [Video](#video)


---

### Video
#### OpenFace + Face VGG:
This feature extractor contains uses Openface to extract and align faces and uses Face VGG to extract facial features from every frame.

Module:
`from src.video.models import OpenFace_VGG` in your data processing code.

---


### Tasks:
- [x] Video feature extractor.
- [ ] Add text feature extractor.
- [ ] Add audio feature extractor.
- [ ] Code cleanup.


### Credits:
1. Authors of [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow), [openface](https://github.com/cmusatyalab/openface)