# Importing necessary libraries
import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from io import BytesIO

# Crop black borders from the stitched panorama
def crop_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return image[y:y+h, x:x+w]

# Feather blending function
def feather_blend(base, warped, mask_warped):
    base_float = base.astype(np.float32)
    warped_float = warped.astype(np.float32)
    mask_base = (base > 0).astype(np.float32)
    mask_warped = mask_warped.astype(np.float32)

    blend_mask = mask_base + mask_warped
    blend_mask[blend_mask == 0] = 1.0

    result = (base_float * mask_base + warped_float * mask_warped) / blend_mask
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

# Stitch two images using SIFT and blending
def stitch_pair_sift_with_blending(base_img, new_img):
    gray1 = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return None

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 10:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    h1, w1 = base_img.shape[:2]
    h2, w2 = new_img.shape[:2]
    corners_new = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_new, H)
    corners_base = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

    all_corners = np.concatenate((warped_corners, corners_base), axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    trans = np.array([[1, 0, -xmin],
                      [0, 1, -ymin],
                      [0, 0, 1]])

    warped = cv2.warpPerspective(new_img, trans @ H, (xmax - xmin, ymax - ymin))
    mask_warped = cv2.warpPerspective(np.ones_like(new_img, dtype=np.uint8) * 255, trans @ H, (xmax - xmin, ymax - ymin))
    translated_base = np.zeros_like(warped)
    translated_base[-ymin:h1 - ymin, -xmin:w1 - xmin] = base_img

    blended = feather_blend(translated_base, warped, mask_warped)
    return blended

# Stitch multiple images sequentially
def stitch_images_sequence(images):
    base = images[0]
    for i in range(1, len(images)):
        stitched = stitch_pair_sift_with_blending(base, images[i])
        if stitched is not None:
            base = stitched
    return crop_black_borders(base)

# Streamlit app
def main():
    st.title("OpenCV Pipeline")
    st.write("Upload 2 or more overlapping images to stitch them into a panorama.")

    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) >= 2:
        images = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            images.append(image_np)

        with st.spinner("Stitching images..."):
            result = stitch_images_sequence(images)

        if result is not None:
            st.success("Panorama created!")
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Stitched Panorama", use_container_width=True)

            # Save to memory and provide download
            result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            buf = BytesIO()
            result_pil.save(buf, format="JPEG")
            st.download_button("Download Panorama", buf.getvalue(), file_name="stitched_panorama.jpg", mime="image/jpeg")
        else:
            st.error("Failed to stitch the images. Try different images or order.")
    else:
        st.info("Please upload at least two images.")

# if __name__ == "__main__":
#     main()
main()
