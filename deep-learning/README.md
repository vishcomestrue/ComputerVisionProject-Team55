# Deep Homography Estimation for Image Stitching

This repository presents a deep learning-based method for estimating homographies between two real-world images and stitching them together. Unlike traditional approaches that rely on feature detection and matching, this pipeline learns homography transformations directly from raw pixel patches using a convolutional neural network.

## Overview

The goal of this project is to compare traditional feature-based methods with a deep learning approach for image stitching. The deep learning pipeline works by:

- Estimating a homography matrix between two real images using traditional ORB + RANSAC as supervision.
- Generating a dataset of patch pairs and their corresponding corner displacements.
- Training a convolutional neural network (HomographyNet) to regress displacements.
- Using the predicted displacements to recover a homography and perform image stitching.

This implementation is inspired by the "Deep Image Homography Estimation" approach by DeTone et al., and demonstrates its application on a small real-world dataset.

## How it Works

1. **Homography Estimation using ORB (Traditional Baseline)**  
   We extract ORB keypoints and descriptors from both images and match them using the brute-force matcher. The best matches are used to estimate a ground-truth homography using RANSAC.

2. **Patch Generation and Labeling**  
   From the input image pair, a random patch is selected from image A. Its corresponding location in image B is determined using the known homography. This pair forms the input to the neural network. The label is the displacement between the four corners of the patch.

3. **Network Architecture**  
   The model accepts a 6-channel input formed by stacking RGB patch A and patch B. It passes through several convolutional layers followed by fully connected layers to output an 8-dimensional vector representing the displacements: `[Δx1, Δy1, Δx2, Δy2, Δx3, Δy3, Δx4, Δy4]`

4. **Training**  
The model is trained using mean squared error loss between the predicted and actual displacements. The loss function is:

$$
\mathcal{L} = \frac{1}{8} \sum_{i=1}^{4} \left[ (\Delta x_i - \hat{\Delta x}_i)^2 + (\Delta y_i - \hat{\Delta y}_i)^2 \right]
$$

5. **Prediction and Stitching**  
Once trained, the network predicts corner displacements on a test patch. These are used to compute a homography via `cv2.getPerspectiveTransform()`. The second image is warped using this homography and stitched with the first image using blending.

## Files

- `image_stitching.ipynb`: The complete training and inference pipeline.
- `room_2.jpg`, `room_3.jpg`: Sample input images for stitching.
- `README.md`: Project documentation.

## Requirements

Install the required packages using pip:
`pip install torch torchvision opencv-python matplotlib`

Make sure that PyTorch is installed with GPU support if you're using a CUDA-compatible system.

## How to Run

1. Place two overlapping input images (e.g., `room_2.jpg`, `room_3.jpg`) in the root directory.
2. Run the Jupyter Notebook

The script performs the following steps:

- Estimates the ground-truth homography using ORB.
- Generates a dataset of patch pairs.
- Trains `HomographyNet` using the generated dataset.
- Predicts displacements from a test patch.
- Computes a homography, warps image 2, and stitches it with image 1.
- Displays the final stitched output.

## Notes

- Input images are resized to 320×240 before processing.
- Patch size is fixed at 128×128.
- For best results, images should have some overlapping content and be taken from nearby viewpoints.
- You can customize the patch location and image size inside the `predict_and_stitch()` function for different scenarios.

## Sample Output

The output is a stitched image where the second image is aligned and blended onto the first using the predicted homography.

## References

- DeTone, D., Malisiewicz, T., & Rabinovich, A. (2016). Deep Image Homography Estimation. CVPR Workshops.
- OpenCV ORB: https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html

## License

This project is for educational and research purposes only.
