import numpy as np
import pandas as pd
import cv2 as cv
from ultralytics import YOLO
import torch
import time
from tracker import Tracker

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

device = "cpu"
model.to(device)


# Create a mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


# connecting callback function with the mouse
cv.namedWindow("RGB")
cv.setMouseCallback("RGB", RGB)

# read the video file
cap = cv.VideoCapture("traffic.mp4")


# get the parameters of the video
fps = int(cap.get(cv.CAP_PROP_FPS))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))


# write video file
writer = cv.VideoWriter(
    "result.mp4", cv.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
)

# class list in the COCO image dataset
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# area to define our region of interest(ROI)
# area = [(350, 169), (390, 140), (1050, 85), (1200, 116)]
area = [(width - 1570, 169), (width - 1530, 140), (width - 870, 85), (width - 720, 116)]
tracker = Tracker()

area_c = set()


while True:
    ret, frame = cap.read()
    if ret == True:
        frame = cv.resize(frame, (1200, 600))

        results = model.predict(frame)
        bboxes = results[0].boxes.data
        bboxes_pd = pd.DataFrame(bboxes).astype("float")
        list = []

        for index, row in bboxes_pd.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]

            if "car" in c:
                list.append([x1, y1, x2, y2])
        bbox_id = tracker.update(list)
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx = int(x3 + x4) // 2
            cy = int(y3 + y4) // 2
            results = cv.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
            if results >= 0:
                cv.circle(frame, (cx, cy), 5, (154, 85, 255), -1)
                cv.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                area_c.add(id)
            cv.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        cv.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 255), 3)
        vehicles = len(area_c)
        cv.putText(
            frame,
            f"Vehicle count: {vehicles}",
            (50, 80),
            cv.FONT_HERSHEY_COMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

        time.sleep(1 / fps)
        cv.imshow("RGB", frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

    else:
        break

cap.release()
cv.destroyAllWindows()
