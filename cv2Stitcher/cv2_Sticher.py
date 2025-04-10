# -*- coding: utf-8 -*-
"""code.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1p71jwlQqzMwNvk-oolkrMuctTtBUcw7B
"""

'''# Installing required libraries
import cv2
import os

# Fetching the images
input_folder = 'inputs'
images = []

for i in os.listdir(input_folder):
    img = cv2.imread(os.path.join(input_folder,i))
    images.append(img)

# Performing Stitching using cv2.Sticher_create() class
stitcher = cv2.Stitcher_create()
_,res= stitcher.stitch(images)

#Scale and Display the resultant image after Stitching
from google.colab.patches import cv2_imshow
res_scaled = cv2.resize(res, (800,600))  #Scale to 800x600
cv2_imshow(res_scaled)
#cv2.waitKey(0)
#cv2.destroyAllWindows()'''

import streamlit as st
import cv2
import numpy as np
import tempfile
import os

st.title("Image Stitching with OpenCV")
st.write("Upload multiple overlapping images to create a panorama.")

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
            result_resized = cv2.resize(result, (800, 600))
            st.image(cv2.cvtColor(result_resized, cv2.COLOR_BGR2RGB), caption="Stitched Panorama", use_column_width=True)
        else:
            st.error(f"Stitching failed. Error code: {status}")
    else:
        st.warning("Please upload at least two images.")
