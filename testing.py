import cv2
import numpy as np
import tensorflow as tf
import time
from cvzone.HandTrackingModule import HandDetector
import pyttsx3

model = tf.keras.models.load_model("trend_model/keras_model.h5")

with open("trend_model/labels.txt", "r") as f:
    labels = [line.strip().split(' ')[-1] for line in f.readlines()]

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

engine = pyttsx3.init()
engine.setProperty('rate', 160)

IMG_SIZE = 300
OFFSET = 20
CONF_THRESHOLD = 0.75

last_label = ""
last_time = 0
cooldown = 2 

print("System Testing Started... Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    output = frame.copy()
    hands, frame = detector.findHands(frame)

    if hands:
        x, y, w, h = hands[0]['bbox']

        y1, y2 = max(0, y - OFFSET), min(frame.shape[0], y + h + OFFSET)
        x1, x2 = max(0, x - OFFSET), min(frame.shape[1], x + w + OFFSET)

        crop = frame[y1:y2, x1:x2]

        if crop.size != 0:
            white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255

            ratio = h / w

            if ratio > 1:
                scale = IMG_SIZE / h
                new_w = int(scale * w)
                resized = cv2.resize(crop, (new_w, IMG_SIZE))
                gap = (IMG_SIZE - new_w) // 2
                white[:, gap:gap+new_w] = resized
            else:
                scale = IMG_SIZE / w
                new_h = int(scale * h)
                resized = cv2.resize(crop, (IMG_SIZE, new_h))
                gap = (IMG_SIZE - new_h) // 2
                white[gap:gap+new_h, :] = resized

            img = cv2.resize(white, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            preds = model.predict(img)
            idx = np.argmax(preds)
            confidence = preds[0][idx]

            detected_label = labels[idx] if idx < len(labels) else "Unknown"

            cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(output, f"{detected_label} ({confidence:.2f})",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 0, 0), 2)

            current_time = time.time()
            if confidence > CONF_THRESHOLD:
                if detected_label != last_label or (current_time - last_time) > cooldown:
                    engine.say(detected_label)
                    engine.runAndWait()
                    last_label = detected_label
                    last_time = current_time

            cv2.imshow("Processed Input", white)

    cv2.imshow("Testing Window", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()