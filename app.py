import streamlit as st
import cv2
from ultralytics import YOLO
import time

st.set_page_config(page_title="AI Object Detection", page_icon="🎯", layout="wide")

st.title("🎯 AI Real-Time Object Detection System")
st.write("Real-time object detection using YOLOv8 and Streamlit")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

if "run" not in st.session_state:
    st.session_state.run = False

col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Detection"):
        st.session_state.run = True

with col2:
    if st.button("⏹ Stop Detection"):
        st.session_state.run = False

frame_placeholder = st.empty()

if st.session_state.run:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera not accessible")

    else:
        last_spoken = ""
        last_time = time.time()

        while st.session_state.run:

            ret, frame = cap.read()

            if not ret:
                st.error("Unable to read camera")
                break

            results = model(frame, verbose=False)

            detected_objects = []

            for result in results:
                if result.boxes is not None:

                    for box in result.boxes:

                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = model.names[cls]

                        if conf > 0.6:

                            detected_objects.append(label)

                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            cv2.putText(
                                frame,
                                f"{label} {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )

                            if label != last_spoken and time.time() - last_time > 3:
                                st.toast(f"{label} detected")
                                last_spoken = label
                                last_time = time.time()

            if detected_objects:
                st.write("### Detected Objects")
                st.write(", ".join(set(detected_objects)))

            frame_placeholder.image(frame, channels="BGR")

        cap.release()
