[![Build Status](https://travis-ci.com/5starkarma/face-smoothing.svg?branch=main)](https://travis-ci.com/5starkarma/face-smoothing) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

OpenCV implementation of facial smoothing. Facial detection is done using an pretrained TensorFlow face detection model. Facial smoothing is accomplished using the following steps:

- Change image from BGR to HSV colorspace
- Create mask of HSV image
- Apply a bilateral filter to the Region of Interest
- Apply filtered ROI back to original image

before_install:
    python -m venv venv
    .\venv\Scripts\Activate
    pip install numpy opencv-python
    pip install -U PyYAML
    pip install -r requirements.txt