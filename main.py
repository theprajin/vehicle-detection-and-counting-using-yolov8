import cv2 as cv
import numpy as np
import time


cap = cv.VideoCapture("traffic.mp4")
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
print(width, height)

if not cap.isOpened():
    print("Error File Not Found or Wrong Codec Used")

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    frame = cv.resize(frame, (width, height))

    cv.line(
        frame, (width - 1900, height - 500), (width - 20, height - 500), (255, 0, 0), 3
    )

    time.sleep(1 / fps)

    # Display the resulting frame
    cv.imshow("frame", frame)
    if cv.waitKey(1) == ord("q"):
        break


cap.release()
cv.destroyAllWindows()
