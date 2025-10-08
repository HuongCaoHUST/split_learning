# from engine.train import Split_Learning_Trainer

# args = dict(model="./yolo11n", data="./datasets/livingroom_2.yaml", epochs=5, )
# trainer = Split_Learning_Trainer(overrides=args)
# trainer.train()
from ultralytics import YOLO

model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="./datasets/livingroom_4_1.yaml", epochs=10, batch=16)