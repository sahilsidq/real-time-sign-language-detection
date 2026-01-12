import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

# Labels list (change as per your signs)
labels = ["Hello", "Yes", "No", "ThankYou", "Please", "LoveYou", "Help"]
current_label = labels[0]

# Base folder
base_folder = "Data"

# Create folders automatically
for label in labels:
    path = os.path.join(base_folder, label)
    os.makedirs(path, exist_ok=True)

counter = 0

while True:
    success, img = cap.read()
    if not success:
        continue

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        if imgCrop.size == 0:
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap+wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap+hCal, :] = imgResize

        cv2.imshow("ImageWhite", imgWhite)

    cv2.putText(img, f"Label: {current_label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)

    # Save image
    if key == ord('s'):
        folder = os.path.join(base_folder, current_label)
        cv2.imwrite(f'{folder}/Image_{counter}.jpg', imgWhite)
        counter += 1
        print(f"{current_label} Images:", counter)

    # Change label
    elif key == ord('n'):
        idx = labels.index(current_label)
        current_label = labels[(idx + 1) % len(labels)]
        counter = 0
        print("Switched to:", current_label)

    # Quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()