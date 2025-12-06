from engine.yolo.predict import Split_Learning_DetectionPredictor
import time

args = dict(model="fedavg_model_layer_1.pt", source="./video/bee_video.mp4")
predictor = Split_Learning_DetectionPredictor(overrides=args)
start_time = time.time()
predictor.predict_cli()
end_time = time.time()
total = end_time - start_time
print(f"Total processing time: {total:.2f} seconds")