import streamlit as st
import importlib.util
import os

st.set_page_config(page_title="Team55 CV Project", layout="wide")

st.sidebar.title("Navigation")

# Map of display names to script paths
PAGES = {
    # "OpenCV Pipeline": "OpenCV-pipeline/from_scractch_using_cv_libraries.py",
    "CV2 Stitcher": "cv2Stitcher/cv2_Sticher.py",
    "Deep Learning Pipeline": "deep-learning/streamlit.py",
    "No-Library (From Scratch)": "no-library-from-scratch/image_stitching_from_scratch_final.py",
}

choice = st.sidebar.radio("Choose a pipeline:", list(PAGES.keys()))

# Load selected app dynamically
def run_script(script_path):
    spec = importlib.util.spec_from_file_location("module.name", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

run_script(PAGES[choice])
