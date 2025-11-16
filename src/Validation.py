from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.models.yolo.classify import ClassificationValidator
from ultralytics import YOLO
import torch
import src.Utils

class ModelValidator:
    def __init__(self, total_client, hybrid_training, cut_layer, best_model_layer_1, best_model_2, epoch_model_layer_1, epoch_model_layer_2, dataset_path, output_model):
        self.total_clients = total_client
        self.hybrid_training = hybrid_training
        self.cut_layer = cut_layer
        self.best_model_layer_1 = best_model_layer_1
        self.best_model_2 = best_model_2
        self.epoch_model_layer_1 = epoch_model_layer_1
        self.epoch_model_layer_2 = epoch_model_layer_2
        self.dataset_path = dataset_path
        self.output_model = output_model

    def validate_best_model(self):
        print("Best model layer 1 full: ", self.best_model_layer_1)
        merge_model = self.merge_yolo_models()
        model = YOLO(merge_model)
        metrics = model.val(data="./datasets/livingroom_4_1.yaml")
        print("metrics: ", metrics)
        return True
    
    def validate_epoch_model(self):
        epoch = 0
        for val1, val2 in zip(self.epoch_model_layer_1, self.epoch_model_layer_2):
            print("Epoch model layer 1: ", val1)
            print("Epoch model layer 2: ", val2)
            merge_model = self.merge_yolo_epoch_models(val1, val2)
            args = dict(model=merge_model, data=self.dataset_path[0])
            validator = DetectionValidator(args=args)
            results = validator()
            epoch += 1
            print(f"Epoch {epoch}: precision={results['metrics/precision(B)']:.4f}, "
                f"recall={results['metrics/recall(B)']:.4f}, "
                f"mAP50={results['metrics/mAP50(B)']:.4f}, "
                f"mAP50-95={results['metrics/mAP50-95(B)']:.4f}")
            src.Utils.log_to_csv(f"./log/log_validation.csv", {
                "epoch": epoch,
                "precision": results['metrics/precision(B)'],
                "recall": results['metrics/recall(B)'],
                "mAP50": results['metrics/mAP50(B)'],
                "mAP50-95": results['metrics/mAP50-95(B)']
            })
        return True
    
    def merge_yolo_models(self):
        output_path = self.output_model
        print("Total client: ", self.total_clients)
        print("Hybrid_training: ", self.hybrid_training)
        
        if self.total_clients[0] == 1:
            model1 = YOLO(self.best_model_layer_1[0])
            model2 = YOLO(self.best_model_2[0])
            output_path = self.output_model

            state_dict1 = model1.model.state_dict()
            state_dict2 = model2.model.state_dict()
            print("Self.cut_layer: ", self.cut_layer[0])
            new_state_dict = state_dict2.copy()

            for k in state_dict1.keys():
                if k.startswith("model."):
                    try:
                        layer_num = int(k.split('.')[1])
                        if layer_num <= self.cut_layer:
                            new_state_dict[k] = state_dict1[k]
                    except:
                        pass

            model2.model.load_state_dict(new_state_dict)

            model2.save(output_path)

            print("Test trong merge")
            print(f"Đã ghép xong model và lưu tại: {output_path}")
            return output_path
        elif self.total_clients[0] > 1 and self.hybrid_training == False:
            print("Ghép cho 2 client")
            state_dicts = []
            for model_path in self.best_model_layer_1:
                model = YOLO(model_path)
                state_dicts.append(model.model.state_dict())
            
            # Average weights
            avg_state_dict = {}
            for key in state_dicts[0].keys():
                if key.startswith("model."):
                    try:
                        layer_num = int(key.split('.')[1])
                        if layer_num <= self.cut_layer[0]:
                            weights = [sd[key] for sd in state_dicts]
                            avg_weight = sum(weights) / len(weights)
                            avg_state_dict[key] = avg_weight
                    except:
                        pass

            model2 = YOLO(self.best_model_2[0])
            state_dict2 = model2.model.state_dict()
            new_state_dict = state_dict2.copy()
            new_state_dict.update(avg_state_dict)

            model2.model.load_state_dict(new_state_dict)
            model2.save(output_path)
            print(f"Đã ghép xong model và lưu tại: {output_path}")
            return output_path
        
        elif self.total_clients[0] > 1 and self.hybrid_training == True:
            print("___HYBRID LEARNING___")
            cut1 = self.cut_layer[0]
            cut2 = self.cut_layer[1]
            
            cut_min = min(cut1, cut2)
            cut_max = max(cut1, cut2)

            # Load models tương ứng
            model_1a = YOLO(self.best_model_layer_1[0])
            model_1b = YOLO(self.best_model_layer_1[1])
            model2 = YOLO(self.best_model_2[0])

            state_1a = model_1a.model.state_dict()
            state_1b = model_1b.model.state_dict()
            state2  = model2.model.state_dict()

            # Model đại diện vùng B (cut lớn hơn)
            state_cut_high = state_1a if cut1 > cut2 else state_1b

            avg_state_dict = {}

            for key in state2.keys():
                if key.startswith("model."):
                    try:
                        layer_num = int(key.split('.')[1])

                        if layer_num <= cut_min:
                            weights = [state_1a[key], state_1b[key]]
                            avg_weight = sum(weights) / 2
                            avg_state_dict[key] = avg_weight

                        elif cut_min < layer_num <= cut_max:
                            weights = [state_cut_high[key], state2[key]]
                            avg_weight = sum(weights) / 2
                            avg_state_dict[key] = avg_weight

                        else:
                            avg_state_dict[key] = state2[key]

                    except:
                        pass

            new_state_dict = state2.copy()
            new_state_dict.update(avg_state_dict)
            model2.model.load_state_dict(new_state_dict)
            model2.save(output_path)

            print(f"Đã ghép xong model và lưu tại: {output_path}")
            return output_path
        elif self.total_clients[0] == 3 and self.hybrid_training == True:
            print("___HYBRID LEARNING___")
            cut1 = self.cut_layer[0]
            cut2 = self.cut_layer[1]
            cut3 = self.cut_layer[2]
            
            cuts = sorted([cut1, cut2, cut3])
            cut_min = cuts[0]
            cut_mid = cuts[1]
            cut_max = cuts[2]

            model_1a = YOLO(self.best_model_layer_1[0])
            model_1b = YOLO(self.best_model_layer_1[1])
            model_1c = YOLO(self.best_model_layer_1[2])
            model2 = YOLO(self.best_model_2[0])

            state_1a = model_1a.model.state_dict()
            state_1b = model_1b.model.state_dict()
            state_1c = model_1c.model.state_dict()
            state2 = model2.model.state_dict()

            state_cut_high = state_1a if cut1 == cut_max else (state_1b if cut2 == cut_max else state_1c)
            state_cut_mid = state_1a if cut1 == cut_mid else (state_1b if cut2 == cut_mid else state_1c)
            state_cut_low = state_1a if cut1 == cut_min else (state_1b if cut2 == cut_min else state_1c)

            avg_state_dict = {}

            for key in state2.keys():
                if key.startswith("model."):
                    try:
                        layer_num = int(key.split('.')[1])

                        if layer_num <= cut_min:
                            weights = [state_1a[key], state_1b[key], state_1c[key]]
                            avg_weight = sum(weights) / 3
                            avg_state_dict[key] = avg_weight

                        elif cut_min < layer_num <= cut_mid:
                            weights = [state_cut_mid[key], state_cut_low[key], state2[key]]
                            avg_weight = sum(weights) / 3
                            avg_state_dict[key] = avg_weight

                        elif cut_mid < layer_num <= cut_max:
                            weights = [state_cut_high[key], state2[key]]
                            avg_weight = sum(weights) / 2
                            avg_state_dict[key] = avg_weight

                        else:
                            avg_state_dict[key] = state2[key]

                    except:
                        pass

            new_state_dict = state2.copy()
            new_state_dict.update(avg_state_dict)
            model2.model.load_state_dict(new_state_dict)
            model2.save(output_path)

            print(f"Đã ghép xong model và lưu tại: {output_path}")
            return output_path
        
    def merge_yolo_epoch_models(self, model1_path = None, model2_path = None):
        output_path = './merged_epoch_model.pt'
        print("MERGE_EPOCH_MODEL")
        model1 = YOLO(model1_path)
        model2 = YOLO(model2_path)

        state_dict1 = model1.model.state_dict()
        state_dict2 = model2.model.state_dict()

        new_state_dict = state_dict2.copy()

        for k in state_dict1.keys():
            if k.startswith("model."):
                try:
                    layer_num = int(k.split('.')[1])
                    if layer_num <= self.cut_layer:
                        new_state_dict[k] = state_dict1[k]
                except:
                    pass

        model2.model.load_state_dict(new_state_dict)

        model2.save(output_path)

        print(f"Đã ghép xong model và lưu tại: {output_path}")
        return output_path
    

    def merge_partial_fedavg(self, model_paths_layer1, model_path_layer2, cut_layer, output_path):
        state_dicts = []
        for path in model_paths_layer1:
            model = YOLO(path)
            state_dicts.append({k: v.cpu() for k, v in model.model.state_dict().items()})
        avg_state_dict = {}
        for key in state_dicts[0].keys():
            try:
                layer_num = int(key.split('.')[0])
                if layer_num <= cut_layer:
                    weights = [sd[key] for sd in state_dicts]
                    avg_state_dict[key] = sum(weights) / len(weights)
            except:
                continue

        model2 = YOLO(model_path_layer2)
        base_state = model2.model.state_dict()
        base_state.update(avg_state_dict)
        model2.model.load_state_dict(base_state)
        model2.save(output_path)                    
        print(f"✅ Đã ghép model xong, lưu tại: {output_path}")
        return output_path
    
    def average_yolo_models(self, best_model_layer_1, output_path):
        print("Averaging 2 YOLO models...")

        model1 = YOLO(best_model_layer_1[0])
        model2 = YOLO(best_model_layer_1[1])

        sd1 = model1.model.state_dict()
        sd2 = model2.model.state_dict()

        if sd1.keys() != sd2.keys():
            raise ValueError("Hai model không có cùng kiến trúc. Keys không giống nhau.")

        avg_sd = {}
        for key in sd1.keys():
            avg_sd[key] = (sd1[key] + sd2[key]) / 2.0

        model_new = YOLO(best_model_layer_1[0])
        model_new.model.load_state_dict(avg_sd)

        model_new.save(output_path)
        print(f"✔ Đã ghép và lưu model tại: {output_path}")
        return output_path
