import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

st.set_page_config(layout="wide")
st.title("🔍 CYC 6.0 B- Check your Coin")

model = YOLO("yolov8n.pt")

def detect_objects(image_np):
    results = model.predict(image_np, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    return boxes, classes

def draw_boxes_pil(image_pil, boxes, classes):
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), f"Obj-{cls}", fill="red", font=font)
    return image_pil

def extract_edges(image_np, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    roi = image_np[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 120)
    return edges, (x1, y1)

def edges_to_colored_transparent(edges, rgb_color):
    h, w = edges.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask = edges > 0
    rgba[mask, :3] = rgb_color
    rgba[mask, 3] = 255
    return rgba

def create_full_edges_rgba(image_np, rgb_color):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 30, 120)
    return edges_to_colored_transparent(edges, rgb_color)

def combine_edges_overlay(edges1_rgba, edges2_rgba, diff_edges):
    h, w, _ = edges1_rgba.shape
    combined = np.zeros((h, w, 4), dtype=np.uint8)

    alpha1 = edges1_rgba[:, :, 3:] / 255.0
    combined[:, :, :3] = (combined[:, :, :3] * (1 - alpha1) + edges1_rgba[:, :, :3] * alpha1).astype(np.uint8)
    combined[:, :, 3] = np.clip(combined[:, :, 3] + edges1_rgba[:, :, 3], 0, 255)

    alpha2 = edges2_rgba[:, :, 3:] / 255.0
    combined[:, :, :3] = (combined[:, :, :3] * (1 - alpha2) + edges2_rgba[:, :, :3] * alpha2).astype(np.uint8)
    combined[:, :, 3] = np.clip(combined[:, :, 3] + edges2_rgba[:, :, 3], 0, 255)

    mask_diff = diff_edges > 0
    combined[mask_diff, :3] = np.array([255, 0, 0], dtype=np.uint8)
    combined[mask_diff, 3] = 255

    return combined

def overlay_diff_on_img(base_img, edge_diff, offset, color=(255, 0, 0)):
    overlay = base_img.copy()
    x_off, y_off = offset
    h, w = edge_diff.shape
    for y in range(h):
        for x in range(w):
            if edge_diff[y, x] > 20:
                if 0 <= y_off + y < overlay.shape[0] and 0 <= x_off + x < overlay.shape[1]:
                    overlay[y_off + y, x_off + x] = color
    return overlay

# --- Input Selection ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Coin Master (Image 1)")
    img1_camera = st.camera_input("📷 Take a picture (Image 1)")
    img1_upload = st.file_uploader("📁 Or upload image (Image 1)", type=["jpg", "jpeg", "png"], key="upload1")

with col2:
    st.markdown("### Coin to Analyse (Image 2)")
    img2_camera = st.camera_input("📷 Take a picture (Image 2)")
    img2_upload = st.file_uploader("📁 Or upload image (Image 2)", type=["jpg", "jpeg", "png"], key="upload2")

# Helper function: draw preview
def prepare_camera_preview(img_file):
    img_pil = Image.open(img_file).convert("RGB")
    np_img = np.array(img_pil)
    boxes, classes = detect_objects(np_img)
    img_with_boxes = draw_boxes_pil(img_pil, boxes, classes)
    return img_with_boxes

# Show previews
if img1_camera:
    st.image(prepare_camera_preview(img1_camera), caption="Image 1 with Detected Objects", use_container_width=True)
elif img1_upload:
    st.image(img1_upload, caption="Image 1 Uploaded", use_container_width=True)

if img2_camera:
    st.image(prepare_camera_preview(img2_camera), caption="Image 2 with Detected Objects", use_container_width=True)
elif img2_upload:
    st.image(img2_upload, caption="Image 2 Uploaded", use_container_width=True)

# Decide which images to compare
img1_file = img1_camera or img1_upload
img2_file = img2_camera or img2_upload

if img1_file and img2_file:
    img1 = Image.open(img1_file).convert("RGB")
    img2 = Image.open(img2_file).convert("RGB")

    img2 = img2.resize(img1.size)

    np1 = np.array(img1)
    np2 = np.array(img2)

    st.subheader("🔍 Edge Detection Preview with Transparency")

    edges1_rgba = create_full_edges_rgba(np1, (0, 255, 0))       # green edges
    edges2_rgba = create_full_edges_rgba(np2, (255, 191, 0))     # amber edges

    gray1 = cv2.cvtColor(np1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np2, cv2.COLOR_RGB2GRAY)
    blur1 = cv2.GaussianBlur(gray1, (3, 3), 0)
    blur2 = cv2.GaussianBlur(gray2, (3, 3), 0)
    edges1 = cv2.Canny(blur1, 30, 120)
    edges2 = cv2.Canny(blur2, 30, 120)
    diff_edges = cv2.absdiff(edges1, edges2)

    combined_overlay = combine_edges_overlay(edges1_rgba, edges2_rgba, diff_edges)

    colE1, colE2, colE3 = st.columns(3)
    with colE1:
        st.image(edges1_rgba, caption="Image 1 Edges (Green)", use_container_width=True)
    with colE2:
        st.image(edges2_rgba, caption="Image 2 Edges (Amber)", use_container_width=True)
    with colE3:
        st.image(combined_overlay, caption="Overlay with Differences (Red)", use_container_width=True)

    st.subheader("🎯 Detected Objects and Comparison")
    boxes1, classes1 = detect_objects(np1)
    boxes2, classes2 = detect_objects(np2)

    if len(classes1) != len(classes2) or not np.array_equal(np.sort(classes1), np.sort(classes2)):
        st.error("⚠️ Objects in the two images are not identical or matched.")
        st.stop()

    comparison_img = np2.copy()
    diff_count = 0
    object_id = 1

    for i, box in enumerate(boxes1):
        edge1, offset1 = extract_edges(np1, box)
        edge2, offset2 = extract_edges(np2, boxes2[i])
        if edge1.shape != edge2.shape:
            continue
        diff = cv2.absdiff(edge1, edge2)
        total_diff = np.sum(diff)
        if total_diff > 0:
            comparison_img = overlay_diff_on_img(comparison_img, diff, offset2, color=(255, 0, 0))
            cv2.putText(comparison_img, f"Obj-{object_id}", (offset2[0], offset2[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            diff_count += 1
            object_id += 1

    st.write(f"**Objects with Line/Curve Differences:** `{diff_count}`")

    st.subheader("🌀 Animation Toggle View")
    toggle = st.radio("View Mode", ["Original", "Difference Overlay"], horizontal=True)
    if toggle == "Original":
        st.image(np2, caption="Original Image 2", use_container_width=True)
    else:
        st.image(comparison_img, caption="Differences Highlighted (Red = Delta Edges)", use_container_width=True)

else:
    st.warning("🚨 Please provide images for BOTH 'Coin Master (Image 1)' AND 'Coin to Analyse (Image 2)'.")
