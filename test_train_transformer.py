from ultralytics import RTDETR
model = RTDETR("rtdetr-resnet50.yaml")  # build a new model from YAML
results = model.train(data="COCO8.yaml", epochs=2, batch = 4)