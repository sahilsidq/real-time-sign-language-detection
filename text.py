import cv2
import numpy as np
import tensorflow as tf
import math
from cvzone.HandTrackingModule import HandDetector

model_path = r"C:\Users\Sahil Siddique\Desktop\Sign language detection\trend_model\keras_model.h5"

model = tf.keras.models.load_model(model_path)

labels_path = r"C:\Users\Sahil Siddique\Desktop\Sign language detection\trend_model\labels.txt"
with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 400

print("Model loaded successfully! Starting camera")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to access camera.")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Crop hand region safely
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
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize

        imgRGB = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
        imgRGB = cv2.resize(imgRGB, (224, 224))       
        imgRGB = imgRGB / 255.0                      
        imgRGB = np.expand_dims(imgRGB, axis=0)

        prediction = model.predict(imgRGB)
        index = np.argmax(prediction)
        label = labels[index] if index < len(labels) else "Unknown"

        cv2.putText(imgOutput, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (0, 255, 0), 4)

        cv2.imshow('Hand Crop', imgCrop)
        cv2.imshow('Processed', imgWhite)

    cv2.imshow('Sign Detection', imgOutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
