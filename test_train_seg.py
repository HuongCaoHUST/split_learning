from ultralytics.models.yolo.segment import SegmentationTrainer

args = dict(model="./yolo11n-seg.pt", data="./datasets/coco128-seg.yaml", epochs=1)
trainer = SegmentationTrainer(overrides=args)
trainer.train()