from ultralytics.models.yolo.detect import DetectionTrainer
from typing import Optional
from ultralytics.utils import RANK
from engine.model import Split_Learning_DetectionModel

class Split_Learning_Trainer(DetectionTrainer):
    def __init__(self, overrides, client_id=None, layer_id=None, num_client=None, cut_layer=None, address=None, username=None, password=None):
        super().__init__(overrides=overrides)
        self.client_id = client_id
        self.layer_id = layer_id
        self.num_client = num_client
        self.cut_layer = cut_layer
        self.address = address
        self.username = username
        self.password = password

    def get_model(self, cfg: Optional[str] = None, weights: Optional[str] = None, verbose: bool = True):
        model = Split_Learning_DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1,
                            layer_id=getattr(self, 'layer_id', None),
                            client_id=getattr(self, 'client_id', None),
                            num_client=getattr(self, 'num_client', None),
                            cut_layer=getattr(self, 'cut_layer', None),
                            address=getattr(self, 'address', None),
                            username=getattr(self, 'username', None),
                            password=getattr(self, 'password', None))
        if weights:
            model.load(weights)
        return model
    
    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        if self.layer_id == 1:
            return None
        elif self.layer_id == 2: 
            return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
                "Epoch",
                "GPU_mem",
                *self.loss_names,
                "Instances",
                "Size",
            )
        else:
            return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
            )