from engine.train import Split_Learning_Trainer

args = dict(model="./yolo11n", data="./datasets/livingroom_4_1.yaml", epochs=1)
trainer = Split_Learning_Trainer(overrides=args)
trainer.train()