# from ultralytics import YOLO
# import cv2
# import time

# model = YOLO("yolo11n.pt")
# cap = cv2.VideoCapture("./video/bee_video.mp4")

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("output.mp4", fourcc, 30, (640, 480))
# start_time = time.time()
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)
#     annotated = results[0].plot()

# end_time = time.time()
# total = end_time - start_time
# print(f"Total processing time: {total:.2f} seconds")
# cap.release()

from ultralytics.models.yolo.detect import DetectionPredictor
import time

args = dict(model="yolo11n.pt", source="./video/bee_video.mp4")
predictor = DetectionPredictor(overrides=args)
start_time = time.time()
predictor.predict_cli()
end_time = time.time()
total = end_time - start_time
print(f"Total processing time: {total:.2f} seconds")
