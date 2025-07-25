import numpy as np
from PIL import Image
import os
import cv2

def train_classifier(data_dir):
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    faces = []
    ids = []

    for image_path in image_paths:
        img = Image.open(image_path).convert('L')
        image_np = np.array(img, 'uint8')
        id = int(os.path.split(image_path)[1].split(".")[1])
        faces.append(image_np)
        ids.append(id)

    ids = np.array(ids)

    if not hasattr(cv2, 'face'):
        raise Exception("cv2.face is not available. Please install opencv-contrib-python.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, ids)
    recognizer.write("classifier.yml")
    print("✅ Model trained and saved as classifier.yml")
