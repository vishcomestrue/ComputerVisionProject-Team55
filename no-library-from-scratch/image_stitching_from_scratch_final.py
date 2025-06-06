# -*- coding: utf-8 -*-
"""Image_Stitching_from_Scratch_Final.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Zuq28wVsP_XLdwfQdGevDxXMgOuhXBQp
"""



import streamlit as st
import numpy as np
import cv2
from skimage import transform
from skimage.transform import rescale
from scipy.ndimage import gaussian_filter
from PIL import Image
import itertools
from collections import deque
import io

#st.set_page_config(layout="wide")
st.title("Implementation from scratch")
st.markdown("Upload 2 or more overlapping images and get a stitched panorama!")

# --- SIFT Feature Extraction ---
def generate_keypoints_descriptors(image):
    gray_img = (image * 255).astype(np.uint8)
    sift = cv2.SIFT_create(nfeatures=5000)
    kp, desc = sift.detectAndCompute(gray_img, None)
    keypoints = np.array([k.pt[::-1] for k in kp])
    return keypoints, desc

def match_descriptors(desc1, desc2, ratio_thresh=0.75):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = [(m.queryIdx, m.trainIdx) for m, n in matches if m.distance < ratio_thresh * n.distance]
    return good_matches

def compute_homography(point1, point2):
    A = []
    for (x1, y1), (x2, y2) in zip(point1, point2):
        A.append([-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    return H / H[2, 2]

def ransac(keypoints1, keypoints2, matches, threshold=5.0, iterations=500):
    best_H, max_inliers, best_inliers = None, 0, []
    if len(matches) < 4:
        return None, []
    for _ in range(iterations):
        sample = np.random.choice(len(matches), 4, replace=False)
        pts1 = np.float32([keypoints1[matches[i][0]][::-1] for i in sample])
        pts2 = np.float32([keypoints2[matches[i][1]][::-1] for i in sample])
        H = compute_homography(pts1, pts2)
        inliers = []
        for i, (a, b) in enumerate(matches):
            pt1 = np.array([*keypoints1[a][::-1], 1])
            projected = H @ pt1
            projected /= projected[2]
            pt2 = np.array(keypoints2[b][::-1])
            if np.linalg.norm(projected[:2] - pt2) < threshold:
                inliers.append((a, b))
        if len(inliers) > max_inliers:
            best_H, best_inliers, max_inliers = H, inliers, len(inliers)
    if max_inliers < 10:
        return None, []
    return best_H, best_inliers

def find_all_matches(images):
    keypoints, descriptors = [], []
    for img in images:
        kp, desc = generate_keypoints_descriptors(img)
        keypoints.append(kp)
        descriptors.append(desc)

    pairwise_matches = {}
    match_graph = {i: [] for i in range(len(images))}
    for (i, j) in itertools.combinations(range(len(images)), 2):
        matches = match_descriptors(descriptors[i], descriptors[j])
        if len(matches) >= 10:
            H, inliers = ransac(keypoints[i], keypoints[j], matches)
            if H is not None and len(inliers) >= 10:
                pairwise_matches[(i, j)] = (H, inliers)
                pairwise_matches[(j, i)] = (np.linalg.inv(H), [(b, a) for (a, b) in inliers])
                match_graph[i].append(j)
                match_graph[j].append(i)
    return pairwise_matches, keypoints, match_graph

def compute_global_homographies(pairwise_matches, match_graph, num_images):
    center = max(match_graph, key=lambda k: len(match_graph[k]))
    homographies = {center: np.eye(3)}
    visited = set([center])
    queue = deque([center])

    while queue:
        curr = queue.popleft()
        for neighbor in match_graph[curr]:
            if neighbor in visited:
                continue
            if (neighbor, curr) in pairwise_matches:
                H, _ = pairwise_matches[(neighbor, curr)]
                homographies[neighbor] = homographies[curr] @ H
            elif (curr, neighbor) in pairwise_matches:
                H, _ = pairwise_matches[(curr, neighbor)]
                homographies[neighbor] = homographies[curr] @ np.linalg.inv(H)
            visited.add(neighbor)
            queue.append(neighbor)

    return homographies

def stitch_all(images, homographies):
    corners = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        pts = np.array([[0,0],[0,h],[w,h],[w,0]])
        pts_homog = np.hstack([pts, np.ones((4,1))])
        warped = (homographies[i] @ pts_homog.T).T
        warped = warped[:, :2] / warped[:, 2:]
        corners.append(warped)

    all_pts = np.vstack(corners)
    min_x, min_y = np.floor(all_pts.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_pts.max(axis=0)).astype(int)
    output_shape = (max_y - min_y, max_x - min_x, 3)

    result = np.zeros(output_shape)
    weight_sum = np.zeros(output_shape[:2])

    for i, img in enumerate(images):
        H = homographies[i]
        shift = transform.SimilarityTransform(translation=(-min_x, -min_y))
        H_shifted = shift.params @ H
        warped = transform.warp(img, np.linalg.inv(H_shifted), output_shape=output_shape[:2], order=3, preserve_range=True)
        mask = (warped.sum(axis=-1) > 0).astype(float)

        dist = cv2.distanceTransform((mask > 0).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
        dist = np.clip(dist / (dist.max() + 1e-8), 0.0, 1.0)

        for c in range(3):
            result[..., c] += warped[..., c] * dist
        weight_sum += dist

    result /= weight_sum[..., None]
    result = gaussian_filter(result, sigma=0.5)
    return result

def crop_black_borders(image):
    image = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > 0
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return image[y:y+h, x:x+w]

# --- Main App Logic ---
uploaded_files = st.file_uploader("Upload at least 2 overlapping images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) >= 2:
    st.info("Processing images...")
    images = []
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = rescale(img_np, 0.5, channel_axis=-1)
        images.append(img_np)

    pairwise_matches, keypoints, match_graph = find_all_matches(images)
    homographies = compute_global_homographies(pairwise_matches, match_graph, len(images))
    panorama = stitch_all(images, homographies)
    panorama = crop_black_borders(panorama)

    st.success("Stitching complete!")
    st.image(panorama, caption=" Final Panorama", use_container_width=True)

    result_img = Image.fromarray(panorama.astype(np.uint8))
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG")
    # st.download_button("Download Panorama", buf.getvalue(), "stitched_panorama.jpg", "image/jpeg")

else:
    st.warning("Please upload at least 2 images to proceed.")
