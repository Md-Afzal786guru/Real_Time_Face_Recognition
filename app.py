import os
import cv2


def generate_dataset(img, id, img_id):
    os.makedirs("data", exist_ok=True)
    filename = f"data/user.{id}.{img_id}.jpg"
    cv2.imwrite(filename, img)
    print(f"✅ Saved: {filename}")


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf=None):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        if clf is not None:
            id, _ = clf.predict(gray_img[y:y + h, x:x + w])
            # 👇 Changed: dynamic user label instead of hardcoded "Afzal"
            name = f"User {id}"
            cv2.putText(img, name, (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

        coords = [x, y, w, h]

    return coords


def recognize(img, clf, faceCascade):
    color = {"white": (255, 255, 255)}
    draw_boundary(img, faceCascade, 1.1, 10, color["white"], "Face", clf)
    return img


def detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade, img_id, user_id):
    color = {"blue": (255, 0, 0)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color["blue"], "Face", clf=None)

    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
        if roi_img.size != 0:
            generate_dataset(roi_img, user_id, img_id)

    return img
