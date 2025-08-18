from engine.train import Split_Learning_Trainer

args = dict(model="./yolo11n", data="./datasets/livingroom_2.yaml", epochs=5)
trainer = Split_Learning_Trainer(overrides=args)
trainer.train()