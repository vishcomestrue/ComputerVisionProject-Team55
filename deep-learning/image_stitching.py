import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image
import io

class HomographyNet(nn.Module):
    def __init__(self):
        super(HomographyNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
        
class RealHomographyDataset(Dataset):
    def __init__(self, img1, img2, homography_matrix, patch_size=128, num_samples=500):
        self.img1 = cv2.resize(img1, (320, 240))
        self.img2 = cv2.resize(img2, (320, 240))
        self.H = homography_matrix
        self.patch_size = patch_size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        h, w, _ = self.img1.shape
        ps = self.patch_size
        x = random.randint(0, w - ps - 1)
        y = random.randint(0, h - ps - 1)

        pts1 = np.array([[x, y], [x+ps, y], [x+ps, y+ps], [x, y+ps]], dtype=np.float32).reshape(-1, 1, 2)
        pts2 = cv2.perspectiveTransform(pts1, self.H)

        patch_a = self.img1[y:y+ps, x:x+ps].copy()
        H_inv = np.linalg.inv(self.H)
        warped_img2 = cv2.warpPerspective(self.img2, H_inv, (w, h))
        patch_b = warped_img2[y:y+ps, x:x+ps].copy()

        pair = np.concatenate([patch_a, patch_b], axis=2).transpose(2, 0, 1).astype(np.float32) / 255.0
        label = (pts2.reshape(-1, 2) - pts1.reshape(-1, 2)).reshape(-1).astype(np.float32)

        return torch.tensor(pair), torch.tensor(label)

def estimate_homography(img1, img2):
    # For now: identity transformation
    return np.eye(3, dtype=np.float32)

def train_model(model, dataset):
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(5):  # reduce for demo
        total_loss = 0
        for img_pair, labels in loader:
            img_pair, labels = img_pair.cuda(), labels.cuda()
            pred = model(img_pair)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # st.write(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

def predict_and_stitch(model, img1, img2):
    img1 = cv2.resize(img1, (320, 240))
    img2 = cv2.resize(img2, (320, 240))

    x, y = 100, 60
    patch_size = 128
    patch_a = img1[y:y+patch_size, x:x+patch_size]
    patch_b = img2[y:y+patch_size, x:x+patch_size]

    pair = np.concatenate([patch_a, patch_b], axis=2).transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = torch.tensor(pair).unsqueeze(0).cuda()

    with torch.no_grad():
        pred = model(input_tensor).cpu().numpy().reshape(4, 2)

    pts1 = np.float32([[x, y], [x+patch_size, y], [x+patch_size, y+patch_size], [x, y+patch_size]])
    pts2 = pts1 + pred
    H_pred = cv2.getPerspectiveTransform(pts2, pts1)

    h, w = img2.shape[:2]
    corners_img2 = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_img2, H_pred)

    all_corners = np.concatenate((warped_corners, np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)), axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translation = [-xmin, -ymin]

    H_translation = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    warped_img2 = cv2.warpPerspective(img2, H_translation @ H_pred, (xmax - xmin, ymax - ymin))
    result = np.zeros_like(warped_img2)
    result[translation[1]:img1.shape[0]+translation[1], translation[0]:img1.shape[1]+translation[0]] = img1

    mask = (warped_img2 > 0).astype(np.uint8)
    stitched = warped_img2.copy()
    stitched[mask == 0] = result[mask == 0]

    stitched_rgb = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
    return stitched_rgb

# ---------------------- STREAMLIT APP ----------------------
st.title("Deep Learning based Image stitching")

img1_file = st.file_uploader("Upload Image 1", type=["jpg", "png", "jpeg"])
img2_file = st.file_uploader("Upload Image 2", type=["jpg", "png", "jpeg"])

if img1_file and img2_file:
    img1_bytes = np.array(Image.open(img1_file).convert("RGB"))
    img2_bytes = np.array(Image.open(img2_file).convert("RGB"))
    img1 = cv2.cvtColor(img1_bytes, cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(img2_bytes, cv2.COLOR_RGB2BGR)

    st.image([img1_bytes, img2_bytes], caption=["Image 1", "Image 2"], width=300)

    if st.button("Train & Stitch"):
        st.write("Estimating Homography ...")
        H = estimate_homography(img1, img2)
        dataset = RealHomographyDataset(img1, img2, H)
        model = HomographyNet().cuda()
        st.write("Training model ...")
        train_model(model, dataset)

        st.write("Generating stitched image...")
        stitched_img = predict_and_stitch(model, img1, img2)
        st.image(stitched_img, caption="Stitched Output", use_column_width=True)
