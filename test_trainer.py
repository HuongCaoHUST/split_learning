from engine.train import Split_Learning_Trainer

args = dict(model="./yolo11n", data="./datasets/coco128.yaml", epochs=1)
trainer = Split_Learning_Trainer(overrides=args)
trainer.train()