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

def detect_toast_level(gray_img, mask):
    mean_brightness = cv2.mean(gray_img, mask=mask)[0]
    toast_level = 100 - (mean_brightness / 255) * 100
    return toast_level

def detect_brown_spots(gray_img, mask):
    # Detect dark spots inside the chapathi
    spot_thresh = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY_INV)[1]
    spot_thresh = cv2.bitwise_and(spot_thresh, spot_thresh, mask=mask)
    contours, _ = cv2.findContours(spot_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter out tiny noise
    brown_spots = [cnt for cnt in contours if cv2.contourArea(cnt) > 30]
    return brown_spots

def process_image(uploaded_image):
    img = np.array(uploaded_image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(gray) < 127:
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, 0, 0, 0

    biggest = max(contours, key=cv2.contourArea)
    roundness = calculate_roundness(biggest)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [biggest], -1, 255, -1)

    toast_level = detect_toast_level(gray, mask)
    brown_spots = detect_brown_spots(gray, mask)

    output = img.copy()
    cv2.drawContours(output, [biggest], -1, (0, 255, 0), 3)

    # Draw brown spots
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

# Streamlit UI
st.title("🫓 Chapathi Roundness Rater (Now with Toast & Spot Detection!)")
st.write("Upload a top-view image of your chapathi and prepare to be judged 😈.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chapathi", use_container_width=True)

    result_img, roundness, toast_level, spot_count = process_image(image)
    st.image(result_img, caption=f"Detected Chapathi (Roundness: {roundness:.2f}%)", use_container_width=True)

    st.subheader("📊 Results:")
    st.markdown(f"**Roundness Score:** {roundness:.2f}%")
    st.markdown(f"**Toast Level (Overcooked %):** {toast_level:.2f}%")
    st.markdown(f"**Brown Spots Detected:** {spot_count}")

    st.subheader("Verdict:")
    st.success(get_rating_message(roundness))

