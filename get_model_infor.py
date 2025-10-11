from engine.train import Split_Learning_DetectionTrainer
args = dict(model="./yolo11l_custom.yaml",
                    data="./datasets/livingroom_4_1.yaml",
                    epochs=1)
trainer = Split_Learning_DetectionTrainer(overrides=args, client_id="abc",
                                         layer_id=1, num_client=2,
                                         cut_layer=0)
trainer.train()