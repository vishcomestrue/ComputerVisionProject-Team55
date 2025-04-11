import streamlit as st
import os
from PIL import Image

#st.set_page_config(layout="wide")
st.title("Image Stitching Showcase")

st.write(os.path.dirname(os.path.abspath(__file__)))

# Define image paths (input1, input2, output)
image_sets = [
    ("images/sample_1.jpg", "images/sample_2.jpg", "outputs/sample.png"),
    ("images/room_2.jpg", "images/room_3.jpg", "outputs/room.png"),
    ("images/building_2.jpg", "images/building_3.jpg", "outputs/building.png"),
]

# Show each set
for idx, (img1_path, img2_path, output_path) in enumerate(image_sets):
    st.subheader(f"Image Pair {idx + 1}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1_path, caption="Input Image 1", use_column_width=True)
    with col2:
        st.image(img2_path, caption="Input Image 2", use_column_width=True)

    # Show result button
    show = st.button(f"View Result {idx + 1}", key=f"result_{idx}")
    if show:
        if os.path.exists(output_path):
            st.image(output_path, caption="Stitched Output", use_column_width=True)
        else:
            st.warning("Output image not found.")
    st.markdown("---")
