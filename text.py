import cv2
import numpy as np
import tensorflow as tf
import math
from cvzone.HandTrackingModule import HandDetector
import pyttsx3
import time

# Load model
model_path = r"C:\Users\Sahil Siddique\Desktop\Sign language detection\trend_model\keras_model.h5"
model = tf.keras.models.load_model(model_path)

labels_path = r"C:\Users\Sahil Siddique\Desktop\Sign language detection\trend_model\labels.txt"
with open(labels_path, "r") as f:
    labels = [line.strip().split(' ', 1)[-1] for line in f.readlines()]

# Camera
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 400

# 🔊 Voice setup
engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

# Control variables
prev_label = ""
last_spoken_time = 0

# Sentence
sentence = []

print("Model loaded successfully! Starting camera...")

while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Preprocess
        imgRGB = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
        imgRGB = cv2.resize(imgRGB, (224, 224))
        imgRGB = imgRGB / 255.0
        imgRGB = np.expand_dims(imgRGB, axis=0)

        # Prediction
        prediction = model.predict(imgRGB, verbose=0)
        index = np.argmax(prediction)
        confidence = prediction[0][index]

        label = labels[index] if index < len(labels) else "Unknown"

        if confidence > 0.75:
            current_time = time.time()

            
            if label != prev_label or (current_time - last_spoken_time) > 2:
                engine.say(label)
                engine.runAndWait()

                sentence.append(label)
                prev_label = label
                last_spoken_time = current_time

        # Draw UI
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (0, 255, 0), 3)

        text = f"{label} ({round(confidence*100,1)}%)"
        text_y = y - 35 if y - 35 > 35 else y + h + 45

        cv2.putText(imgOutput, text, (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Processed", imgWhite)

    # Sentence display
    cv2.putText(imgOutput, "Sentence: " + " ".join(sentence[-6:]),
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Sign Detection", imgOutput)

    key = cv2.waitKey(1)

    if key == ord('c'):
        sentence = []

    if key == ord('v'):
        full_sentence = " ".join(sentence)
        engine.say(full_sentence)
        engine.runAndWait()

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()