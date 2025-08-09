import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title="Marigold Petal Counter", layout="wide")

st.title("🌼 Marigold Petal Counter")
st.write("Upload a marigold photo — the app detects fine edge segments and estimates the petal count.")

# Sidebar parameters (you can tune these)
st.sidebar.header("Tuning parameters (optional)")
MAX_WIDTH = st.sidebar.slider("Max image width (px)", 300, 1600, 900)
h_lower = st.sidebar.slider("HSV lower Hue", 0, 50, 5)
h_upper = st.sidebar.slider("HSV upper Hue", 10, 60, 40)
s_lower = st.sidebar.slider("HSV lower Sat", 0, 255, 60)
v_lower = st.sidebar.slider("HSV lower Val", 0, 255, 60)
morph_kernel = st.sidebar.slider("Morph kernel size", 3, 15, 5)
morph_iters = st.sidebar.slider("Morph close iterations", 1, 6, 2)
canny_low = st.sidebar.slider("Canny low threshold", 5, 100, 30)
canny_high = st.sidebar.slider("Canny high threshold", 50, 300, 120)
min_area = st.sidebar.slider("Min contour area (px)", 1, 200, 8)
angle_bins = st.sidebar.slider("Angular histogram bins", 60, 360, 180)
smooth_win = st.sidebar.slider("Smoothing window (bins)", 1, 21, 5)
peak_factor = st.sidebar.slider("Peak threshold factor", 1, 50, 10) / 100.0
merge_gap = st.sidebar.slider("Merge nearby peaks (bins)", 0, 10, 3)

uploaded = st.file_uploader("Upload an image (jpg / png)", type=["jpg", "jpeg", "png"])

def read_image(file) -> np.ndarray:
    data = file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(img)
    return arr[:, :, ::-1]  # RGB -> BGR for OpenCV

def resize_keep_aspect(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def mask_flower_hsv(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_lower, s_lower, v_lower])
    upper = np.array([h_upper, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def find_center_and_radius(mask, fallback_img):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = fallback_img.shape[:2]
        return w/2, h/2, min(w,h)/3  # fallback center & radius
    # choose largest contour
    areas = [cv2.contourArea(c) for c in cnts]
    idx = int(np.argmax(areas))
    c = cnts[idx]
    M = cv2.moments(c)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w/2
        cy = y + h/2
    x, y, wbox, hbox = cv2.boundingRect(c)
    radius = max(wbox, hbox) * 0.6
    return cx, cy, radius, c

def compute_edge_candidates(img_bgr, cx, cy, radius):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # equalize to boost contrast
    eq = cv2.equalizeHist(blur)
    edges = cv2.Canny(eq, canny_low, canny_high)
    # mask edges to a circular region around flower to avoid background noise
    h, w = edges.shape
    Y, X = np.ogrid[:h, :w]
    dist2 = (X - cx)**2 + (Y - cy)**2
    mask_circle = dist2 <= (radius * 1.35) ** 2
    edges_masked = edges.copy()
    edges_masked[~mask_circle] = 0
    # small closing/dilation to join fragments
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edges_masked = cv2.dilate(edges_masked, k, iterations=1)
    edges_masked = cv2.erode(edges_masked, k, iterations=1)
    # find contours on edges
    cnts, _ = cv2.findContours(edges_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        rx, ry, rw, rh = cv2.boundingRect(c)
        cx_seg = rx + rw/2
        cy_seg = ry + rh/2
        dist = np.hypot(cx_seg - cx, cy_seg - cy)
        if dist > radius * 0.15 and dist < radius * 1.3:  # keep those on outer ring
            candidates.append((int(cx_seg), int(cy_seg)))
    return edges_masked, candidates

def angular_histogram_count(candidates, cx, cy, bins=180, smooth_w=5, peak_factor_local=0.15, merge_gap_local=3):
    if not candidates:
        return 0, []
    hist = np.zeros(bins, dtype=np.int32)
    for (px, py) in candidates:
        angle = np.arctan2(py - cy, px - cx)  # -pi..pi
        if angle < 0:
            angle += 2*np.pi
        idx = int((angle / (2*np.pi)) * bins) % bins
        hist[idx] += 1
    # smoothing
    if smooth_w > 1:
        kernel = np.ones(smooth_w, dtype=np.float32) / smooth_w
        smooth = np.convolve(hist, kernel, mode='same')
    else:
        smooth = hist.astype(np.float32)
    # dynamic threshold
    peak_thresh = max(1.0, peak_factor_local * smooth.max())
    # find local peaks
    peaks = []
    for i in range(bins):
        prev = smooth[(i-1) % bins]
        cur = smooth[i]
        nxt = smooth[(i+1) % bins]
        if cur >= prev and cur >= nxt and cur >= peak_thresh:
            peaks.append(i)
    # merge nearby peaks
    if not peaks:
        return 0, []
    merged = [peaks[0]]
    for p in peaks[1:]:
        if p - merged[-1] <= merge_gap_local:
            # keep the stronger
            if smooth[p] > smooth[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)
    # If first and last are close around wrap, try merging
    if len(merged) > 1 and (merged[0] + bins - merged[-1]) <= merge_gap_local:
        # choose stronger of two
        if smooth[merged[0]] >= smooth[merged[-1]]:
            merged.pop(-1)
        else:
            merged.pop(0)
    # Convert merged bin indices to angles (radians)
    peak_angles = [ (p / bins) * 2*np.pi for p in merged ]
    return len(merged), peak_angles

def draw_visualization(img_bgr, mask, center_x, center_y, radius, candidates, peak_angles):
    vis = img_bgr.copy()
    # draw the mask contour (if any)
    if mask is not None:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            big = max(cnts, key=cv2.contourArea)
            cv2.drawContours(vis, [big], -1, (200, 200, 255), 2)
    # draw center and radius
    cv2.circle(vis, (int(center_x), int(center_y)), 3, (0,0,255), -1)
    cv2.circle(vis, (int(center_x), int(center_y)), int(radius), (0,0,255), 2)
    # draw candidate points
    for (px, py) in candidates:
        cv2.circle(vis, (px, py), 2, (0,255,0), -1)
    # draw detected peaks as radial lines
    for i, ang in enumerate(peak_angles):
        x2 = int(center_x + np.cos(ang) * radius * 1.05)
        y2 = int(center_y + np.sin(ang) * radius * 1.05)
        cv2.line(vis, (int(center_x), int(center_y)), (x2, y2), (255,0,0), 2)
        # small label
        lx = int(center_x + np.cos(ang) * radius * 1.15)
        ly = int(center_y + np.sin(ang) * radius * 1.15)
        cv2.putText(vis, str(i+1), (lx-6, ly+6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,20,20), 1, cv2.LINE_AA)
    # top-left text
    return vis

if uploaded is not None:
    img_bgr = read_image(uploaded)
    img_bgr = resize_keep_aspect(img_bgr, MAX_WIDTH)

    # Mask flower by HSV color to find center
    mask = mask_flower_hsv(img_bgr)
    try:
        cx, cy, radius, flower_contour = find_center_and_radius(mask, img_bgr)
    except Exception:
        # older fallback if find_center_and_radius returned center w/out contour
        cx, cy, radius = (img_bgr.shape[1]/2, img_bgr.shape[0]/2, min(img_bgr.shape[:2])/3)
        flower_contour = None

    edges_masked, candidates = compute_edge_candidates(img_bgr, cx, cy, radius)
    count, peak_angles = angular_histogram_count(candidates, cx, cy, bins=angle_bins, smooth_w=smooth_win, peak_factor_local=peak_factor, merge_gap_local=merge_gap)

    vis = draw_visualization(img_bgr, mask, cx, cy, radius, candidates, peak_angles)

    # Display original and processed images side-by-side
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
    with col2:
        st.subheader("Processed (candidates + peaks)")
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Display the sentence result below
    st.markdown("---")
    if count == 0:
        st.error("Result: Could not detect petals reliably. Try a clearer image or tweak parameters in the sidebar (HSV range, Canny thresholds).")
    else:
        st.success(f"Result: This marigold has approximately **{count} petals**.")
    # Show diagnostics
    st.caption(f"Candidates detected: {len(candidates)} • Angular bins: {angle_bins} • Smoothing window: {smooth_win}")
    # Optionally display the edge map
    if st.checkbox("Show internal edge map (debug)"):
        st.image(edges_masked, caption="Edge map masked to flower area", use_column_width=True, clamp=True)
else:
    st.info("Upload an image to begin. Use the sidebar to tune detection parameters if results look off.")
