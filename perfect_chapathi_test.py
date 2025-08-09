import cv2
import numpy as np
import streamlit as st
from PIL import Image

def calculate_roundness(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    roundness = (4 * np.pi * area) / (perimeter * perimeter)
    return roundness * 100

def detect_toast_level(img, mask):
    # Convert to LAB space for accurate color separation
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    # Normalize lighting using CLAHE on the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_eq = clahe.apply(L)

    # Define color thresholds for golden brown (tune values as needed)
    # These operate in the LAB space: medium lightness + yellowish tone
    brown_mask = cv2.inRange(L_eq, 90, 160)
    yellow_mask = cv2.inRange(B, 135, 200)

    # Focus only on the chapati area
    combined = cv2.bitwise_and(brown_mask, yellow_mask)
    combined = cv2.bitwise_and(combined, combined, mask=mask)

    total = cv2.countNonZero(mask)
    browns = cv2.countNonZero(combined)
    if total == 0:
        return 0.0

    toast_percent = (browns / total) * 100
    return 40 + toast_percent



def detect_brown_spots(gray_img, mask):
    # Detect dark spots, but only within the chapathi mask
    spot_thresh = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY_INV)[1]
    spot_thresh = cv2.bitwise_and(spot_thresh, spot_thresh, mask=mask)  # mask applied here

    contours, _ = cv2.findContours(spot_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Keep only spots fully inside the chapathi and above size threshold
    brown_spots = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:
            # Check if the spot contour lies completely inside the chapathi mask
            mask_check = np.zeros_like(mask)
            cv2.drawContours(mask_check, [cnt], -1, 255, -1)
            if cv2.countNonZero(cv2.bitwise_and(mask_check, mask)) == cv2.countNonZero(mask_check):
                brown_spots.append(cnt)
    return brown_spots

def process_image(uploaded_image):
    img = np.array(uploaded_image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Adaptive threshold (better for uneven lighting)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2)
    
    # Optional: invert threshold if background is dark
    if np.mean(gray) < 127:
        thresh = cv2.bitwise_not(thresh)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found")
        return img, 0, 0, 0

    img_area = gray.shape[0] * gray.shape[1]

    # Filter contours by area: ignore too small or too big
    valid_contours = [cnt for cnt in contours if 0.01*img_area < cv2.contourArea(cnt) < 0.9*img_area]
    if not valid_contours:
        print("No valid contours found after area filtering")
        return img, 0, 0, 0

    biggest = max(valid_contours, key=cv2.contourArea)

    roundness = calculate_roundness(biggest)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [biggest], -1, 255, -1)

    toast_level = detect_toast_level(img, mask)
    brown_spots = detect_brown_spots(gray, mask)

    output = img.copy()
    cv2.drawContours(output, [biggest], -1, (0, 255, 0), 5)

    for cnt in brown_spots:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return output, roundness, toast_level, len(brown_spots)



def get_rating_message(score):
    if score > 90:
        return "🥇 Perfect! Your chapathi deserves an award."
    elif score > 75:
        return "👏 Pretty round! Mom would be proud."
    elif score > 60:
        return "🙂 Almost there! Try a rolling pin with more love."
    elif score > 40:
        return "😬 Hmm... Artistic attempt?"
    else:
        return "🤡 Did you drop this on the floor and stamp it?"

def get_toast_message(toast_level):
    if toast_level < 20:
        return "🌞 Pale beauty — did you even light the stove?"
    elif toast_level < 40:
        return "🥱 Lightly kissed by the pan — elegant but shy."
    elif toast_level < 60:
        return "🍯 Golden perfection — Instagram-worthy!"
    elif toast_level < 80:
        return "🔥 A bit on the adventurous side — crunchy vibes!"
    else:
        return "💀 Charcoal edition — perfect for BBQ lovers."


# Streamlit UI
st.title("🫓 Chapathi Roundness Rater (Now with Toast & Spot Detection!)")
st.write("Upload a top-view image of your chapathi and prepare to be judged 😈.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1,col2 = st.columns(2)
    with col1:
     st.image(image, caption="Uploaded Chapathi", width=300)

 
    result_img, roundness, toast_level, spot_count = process_image(image)
    with col2:
     st.image(result_img, caption=f"Detected Chapathi (Roundness: {roundness:.2f}%)", width=300)
    
    


    st.subheader("📊 Results:")
    st.markdown(f"**Roundness Score:** {roundness:.2f}%")
    st.markdown(f"**Toast Level %:** {toast_level:.2f}%")
    st.markdown(f"**Brown Spots Detected:** {spot_count}")

    st.subheader("Verdict:")
    st.markdown("##### 🔵 Roundness:")
    st.success(get_rating_message(roundness))
    st.markdown("##### 🍞 Toast:")
    st.success(get_toast_message(toast_level))

