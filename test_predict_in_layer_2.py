from ultralytics.models.yolo.detect import DetectionPredictor
import time

args = dict(model="yolo11n.pt", source="./bus.jpg")
predictor = DetectionPredictor(overrides=args)
start_time = time.time()
predictor.predict_cli()
end_time = time.time()
total = end_time - start_time
print(f"Total processing time: {total:.2f} seconds")
