import streamlit as st
import cv2
import os

from model_trainer import train_classifier
from app import generate_dataset, recognize, detect

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
noseCascade = cv2.CascadeClassifier("Nariz.xml")
mouthCascade = cv2.CascadeClassifier("Mouth.xml")

if faceCascade.empty() or eyesCascade.empty() or noseCascade.empty() or mouthCascade.empty():
    st.error("❌ Error loading Haar Cascades.")
    st.stop()

clf = None
if os.path.exists("classifier.yml"):
    try:
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.yml")
    except Exception as e:
        st.warning(f"⚠️ Error loading model: {e}")
else:
    st.warning("⚠️ 'classifier.yml' not found. Please train model first.")

st.title("🧠Real_Time_Face_Recognition ")
option = st.radio("Choose Task:", ["📸 Capture Dataset", "🧠 Train Model", "🔍 Recognize Face"])

if option == "📸 Capture Dataset":
    user_id = st.number_input("Enter User ID", min_value=1, step=1)
    if st.button("Start Capture"):
        cap = cv2.VideoCapture(0)
        img_id = 0
        stframe = st.empty()

        while img_id < 50:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from webcam.")
                break
            frame = detect(frame, faceCascade, eyesCascade, noseCascade, mouthCascade, img_id, user_id)
            img_id += 1
            stframe.image(frame, channels="BGR", caption=f"Captured {img_id}/50")

        cap.release()
        cv2.destroyAllWindows()
        st.success("✅ Dataset captured successfully!")

elif option == "🧠 Train Model":
    if st.button("Train Now"):
        train_classifier("data")
        st.success("✅ Model trained and saved.")

elif option == "🔍 Recognize Face":
    if clf is None:
        st.warning("Please train the model first.")
    elif st.button("Start Recognition"):
        cap = cv2.VideoCapture(0)
        st.info("Press 'q' to quit the recognition window.")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam error.")
                break
            img = recognize(frame, clf, faceCascade)
            cv2.imshow("Face Recognition", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
