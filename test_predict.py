from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture("bee_video.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, 30, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()
    out.write(annotated)

cap.release()
out.release()
