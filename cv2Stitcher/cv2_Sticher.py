import streamlit as st
import cv2
import numpy as np
import tempfile
import os

st.title("cv.stitcher()")
st.write("Upload 2 overlapping images to create a panorama.")

uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    images = []
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        images.append(img)

    if len(images) >= 2:
        stitcher = cv2.Stitcher_create()
        status, result = stitcher.stitch(images)

        if status == cv2.Stitcher_OK:
            # result_resized = cv2.resize(result, (800, 600))
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Stitched Panorama", use_container_width=True)
        else:
            st.error(f"Stitching failed. Error code: {status}")
    else:
        st.warning("Please upload at least two images.")
