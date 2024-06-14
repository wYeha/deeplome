import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# cap1 = cv2.VideoCapture(0) # Первая веб-камера
# cap2 = cv2.VideoCapture(1) # Вторая веб-камера

cap1 = cv2.VideoCapture("top1.mp4")
cap2 = cv2.VideoCapture("top2.mp4")

active_cam = 1

model1 = YOLO("YOLO-weights/yolov8x.pt")
model2 = YOLO("YOLO-weights/yolov8x.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove",
              "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

tracker1 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker2 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits1 = [300, 0, 300, 600]  # for webcam
limits2 = [300, 0, 300, 600]  # for phonecam
totalCounts = []
totalCounts1 = []
totalCounts2 = []
removed_ids = []

while True:
    success1, img1 = cap1.read()
    success2, img2 = cap2.read()

    # mask = np.ones(img1.shape, dtype=np.uint8) * 255
    #
    # rect_left = (0, 0, 100, img1.shape[0])
    # rect_right = (img1.shape[1] - 100, 0, img1.shape[1], img1.shape[0])
    #
    # cv2.rectangle(mask, rect_left[0:2], rect_left[2:4], (0, 0, 0), -1)
    # cv2.rectangle(mask, rect_right[0:2], rect_right[2:4], (0, 0, 0), -1)
    #
    # img1_masked = cv2.bitwise_and(img1, mask)
    # img2_masked = cv2.bitwise_and(img2, mask)
    #
    # results1 = model(img1_masked, stream=True)
    # results2 = model(img2_masked, stream=True)

    results1 = model1(img1, stream=True)
    results2 = model2(img2, stream=True)

    detections1 = np.empty((0, 5))
    detections2 = np.empty((0, 5))

    #Webcam
    for r in results1:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            if classNames[cls] == "person" and conf > 0.5:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img1, (x1, y1, w, h), colorR=(229, 244, 10), rt=2)
                currentArray1 = np.array([x1, y1, x2, y2, conf])
                detections1 = np.vstack((detections1, currentArray1))

    #Phonecam
    for r in results2:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            if classNames[cls] == "person" and conf > 0.5:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img2, (x1, y1, w, h), colorR=(229, 244, 10), rt=2)
                currentArray2 = np.array([x1, y1, x2, y2, conf])
                detections2 = np.vstack((detections2, currentArray2))

    resultsTracker1 = tracker1.update(detections1)
    resultsTracker2 = tracker2.update(detections2)

    cv2.line(img1, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 255, 0), 3)
    cv2.line(img2, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 255, 0), 3)

    #Webcam
    for result in resultsTracker1:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        # cvzone.putTextRect(img1, f'{Id}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(229, 244, 10))

        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img1, (cx, cy), 5, (229, 244, 10), cv2.FILLED)

        # На вход
        if limits1[1] < cy < limits1[3] and limits1[0]-40 < cx < limits1[0]-10:
            if totalCounts1.count(Id) == 0 and totalCounts2.count(Id) == 0:
                totalCounts1.append(Id)

        if limits1[1] < cy < limits1[3] and limits1[0]-10 < cx < limits1[0]+10:
            if totalCounts1.count(Id) != 0 and totalCounts.count(Id) == 0:
                totalCounts.append(Id)

        # На выход
        if limits1[1] < cy < limits1[3] and limits1[0] + 10 < cx < limits1[0] + 40:
            if totalCounts1.count(Id) == 0 and totalCounts2.count(Id) == 0:
                totalCounts2.append(Id)

        if limits1[1] < cy < limits1[3] and limits1[0]-10 < cx < limits1[0]+10:
            if totalCounts2.count(Id) != 0 and len(totalCounts) != 0 and Id not in removed_ids:
                totalCounts.pop()
                removed_ids.append(Id)


    #Phonecam
    for result in resultsTracker2:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        # cvzone.putTextRect(img2, f'{Id}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(229, 244, 10))

        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img2, (cx, cy), 5, (229, 244, 10), cv2.FILLED)

        # На вход
        if limits2[1] < cy < limits2[3] and limits2[0]-40 < cx < limits2[0]-10:
            if totalCounts1.count(Id) == 0 and totalCounts2.count(Id) == 0:
                totalCounts1.append(Id)

        if limits2[1] < cy < limits2[3] and limits2[0]-10 < cx < limits2[0]+10:
            if totalCounts1.count(Id) != 0 and totalCounts.count(Id) == 0:
                totalCounts.append(Id)

        # На выход
        if limits2[1] < cy < limits2[3] and limits2[0] + 10 < cx < limits2[0] + 40:
            if totalCounts1.count(Id) == 0 and totalCounts2.count(Id) == 0:
                totalCounts2.append(Id)

        if limits1[1] < cy < limits1[3] and limits1[0] - 10 < cx < limits1[0] + 10:
            if totalCounts2.count(Id) != 0 and len(totalCounts) != 0 and Id not in removed_ids:
                totalCounts.pop()
                removed_ids.append(Id)

    # cv2.putText(img1, f'People count: {len(totalCounts)}', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # cv2.putText(img2, f'People count: {len(totalCounts)}', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Переключение между камерами
    # key = cv2.waitKey(1)
    # if key == ord('c') or key == ord('с'):
    #     active_cam = 1 if active_cam == 2 else 2
    #
    # if active_cam == 1:
    #     cv2.imshow("Active cam", img1)
    # elif active_cam == 2:
    #     cv2.imshow("Active cam", img2)
    #
    # if key == ord('q') or key == ord('й'):
    #     break

    cv2.imshow("Active cam1", img1)
    cv2.imshow("Active cam2", img2)
    cv2.waitKey(0)

cap1.release()
cap2.release()
cv2.destroyAllWindows()
