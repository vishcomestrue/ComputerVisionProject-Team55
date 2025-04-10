# To compare traditional and deep learning methods in Image Stitching
> Course Name: CSL 7360 Computer Vision <br />
> Instructor : Pratik Mazumder <br />
> Team ID : 55 <br />

This repository explores different approaches to homography estimation and image stitching, ranging from classical computer vision methods to deep learning-based models. Each subdirectory contains an independent pipeline that demonstrates a unique method for aligning and stitching overlapping images.

## Folder Structure
```
├── OpenCV-pipeline/
├── cv2Stitcher/
├── deep-learning/
├── no-library-from-scratch/
├── requirements.txt
└── README.md
```

---

## 1. OpenCV-pipeline

**Description**:  
A step-by-step classical computer vision pipeline for homography estimation using ORB features, brute-force matching, and RANSAC. Warping and blending are performed manually using OpenCV functions.

**Key Steps**:
- Keypoint detection (ORB)
- Descriptor matching (Brute Force)
- Homography estimation (RANSAC)
- Image warping and stitching

---

## 2. cv2Stitcher

**Description**:  
A concise implementation using OpenCV’s high-level `cv2.Stitcher_create()` API. This is the simplest and most abstracted way to stitch images using OpenCV.

**Key Steps**:
- Load images
- Initialize stitcher
- Call `stitch()` to get the panorama

---

## 3. deep-learning

**Description**:  
Implements a deep learning approach to estimate homographies from image patches. A CNN (HomographyNet) is trained to predict the displacement of corners between two corresponding patches.

**Key Features**:
- Uses a self-supervised label generation strategy (based on ORB + RANSAC)
- Predicts 8 parameters (corner displacements)
- Recovers homography matrix from predicted corners
- Warps second image and stitches with the first

---

## 4. no-library-from-scratch

**Description**:  
A minimal example to implement homography computation and stitching without using OpenCV or deep learning frameworks. Matrix operations are manually coded using NumPy.

**Key Steps**:
- Manual feature selection or hardcoded correspondences
- Direct computation of homography using the DLT algorithm
- Warping with custom interpolation logic

---

## Requirements

Install common dependencies using: `pip install -r requirements.txt`

## License

This repository is intended for research and educational use only.

