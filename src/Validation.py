from ultralytics.models.yolo.detect import DetectionValidator
import src.Utils

def validate_best_model():
        print("Best model layer 1 full: ", self.best_model_layer_1)
        merge_model = self.merge_yolo_models()
        args = dict(model=merge_model, data=self.dataset_path[0], project = './runs/detect',)
        validator = DetectionValidator(args=args)
        validator()
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