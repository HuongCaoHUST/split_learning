import torch
import time
import pickle
import threading
import uuid
from copy import deepcopy
from ultralytics.utils.torch_utils import initialize_weights
from ultralytics.nn.tasks import BaseModel, DetectionModel, ClassificationModel, yaml_model_load
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.ops import make_divisible
from ultralytics.utils import LOGGER, colorstr
import contextlib
from src import Utils
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    LRPCHead,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
)

class Split_Learning_DetectionModel(DetectionModel):
    def __init__(self, cfg=None, nc=None, ch=3, verbose=True, 
                 layer_id=None, client_id=None, num_client=None, cut_layer=None,
                 address=None, username=None, password=None, load_partial_model = False,
                 send_service=None):
        self.layer_id = layer_id
        self.client_id = client_id
        self.num_client = num_client
        self.cut_layer = cut_layer
        self.cut_layer_ids = None

        self.load_partial_model = load_partial_model
        self.is_training = False
        self.client_ids = None
        self.batch_id = None
        self.label = None
        self.send_service = None
        super(BaseModel, self).__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        self.yaml["channels"] = ch
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  
        self.model, self.save = self.parse_model_SL(deepcopy(self.yaml), ch=ch, verbose=verbose, layer_id=self.layer_id, cut_layer=self.cut_layer, load_partial_model=self.load_partial_model)

        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, YOLOEDetect, YOLOESegment
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Perform a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, YOLOESegment, Pose, OBB)) else self.forward(x)

            self.model.eval()  # Avoid changing batch statistics until training begins
            m.training = True  # Setting it to True to properly return strides
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            self.model.train()  # Set model back to training(default) mode
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")
        self.tensor_send_ids = self.get_tensor_send_id(self.cut_layer) if self.layer_id == 1 else []
        self.data_store=None
        self.input_data_id = None

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        dt, embeddings = [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        y = [None] * (self.cut_layer + 1)
        if self.is_training == True:
            print("Tensor 4: ", x[4].shape)
            print("Tensor 6: ", x[6].shape)
            print("Tensor 10: ", x[10].shape)
            y[4] = x[4]
            y[6] = x[6]
            y[10] = x[10]
            x = x[10]
            for m in self.model[self.cut_layer + 1:]:
                if m.f != -1:  # if not from previous layer
                    # print("M.F: ", m.f)

                    # for idx, yi in enumerate(y):
                    #     if yi is None:
                    #         print(f"y[{idx}] = None")
                    #     else:
                    #         print(
                    #             f"y[{idx}] shape={tuple(yi.shape)}, "
                    #             f"sample={yi.flatten()[:5].tolist()}"
                    #         )

                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if profile:
                    self._profile_one_layer(m, x, dt)
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output

                if visualize:
                    feature_visualization(x, m.type, m.i, save_dir=visualize)

                if m.i in embed:
                    embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                    if m.i == max_idx:
                        return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def get_tensor_send_id (self, cut_layer):
        if cut_layer <=3:
            return [cut_layer]
        elif cut_layer == 4:
            return [cut_layer]
        elif cut_layer == 5:
            return [4, cut_layer]
        elif cut_layer == 6:
            return [4, cut_layer]
        elif cut_layer == 7:
            return [4, 6, cut_layer]
        elif cut_layer == 8:
            return [4, 6, cut_layer]
        elif cut_layer == 9:
            return [4, 6, cut_layer]
        elif cut_layer == 10:
            return [4, 6, cut_layer]
        elif cut_layer == 11:
            return [4, 6, 10, cut_layer]
        elif cut_layer == 12:
            return [4, 10, cut_layer]
        elif cut_layer == 13:
            return [4, 10, cut_layer]
        elif cut_layer == 14:
            return [4, 10, 13, cut_layer]
        elif cut_layer == 15:
            return [10, 13, cut_layer]
        elif cut_layer == 16:
            return [10, 13, cut_layer]
        elif cut_layer == 17:
            return [10, 13, 16, cut_layer]
        elif cut_layer == 18:
            return [10, 16, cut_layer]
        elif cut_layer == 19:
            return [10, 16, cut_layer]
        elif cut_layer == 20:
            return [10, 16, 19, cut_layer]
        elif cut_layer == 21:
            return [16, 19, cut_layer]
        elif cut_layer == 22:
            return [16, 19, cut_layer]
        elif cut_layer == 23:
            return [16, 19, 22]

    def parse_model_SL(self, d, ch, verbose=True, layer_id = None, cut_layer = None, load_partial_model = False):
        """
        Parse a YOLO model.yaml dictionary into a PyTorch model.

        Args:
            d (dict): Model dictionary.
            ch (int): Input channels.
            verbose (bool): Whether to print model details.

        Returns:
            model (torch.nn.Sequential): PyTorch model.
            save (list): Sorted list of output layers.
        """
        import ast
        # Args
        legacy = True  # backward compatibility for v3/v5/v8/v9 models
        max_channels = float("inf")
        nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
        depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
        if scales:
            scale = d.get("scale")
            if not scale:
                scale = tuple(scales.keys())[0]
                LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]

        if act:
            Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
            if verbose:
                LOGGER.info(f"{colorstr('activation:')} {act}")  # print

        if verbose:
            LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
        ch = [ch]
        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        base_modules = frozenset(
            {
                Classify,
                Conv,
                ConvTranspose,
                GhostConv,
                Bottleneck,
                GhostBottleneck,
                SPP,
                SPPF,
                C2fPSA,
                C2PSA,
                DWConv,
                Focus,
                BottleneckCSP,
                C1,
                C2,
                C2f,
                C3k2,
                RepNCSPELAN4,
                ELAN1,
                ADown,
                AConv,
                SPPELAN,
                C2fAttn,
                C3,
                C3TR,
                C3Ghost,
                torch.nn.ConvTranspose2d,
                DWConvTranspose2d,
                C3x,
                RepC3,
                PSA,
                SCDown,
                C2fCIB,
                A2C2f,
            }
        )
        repeat_modules = frozenset(  # modules with 'repeat' arguments
            {
                BottleneckCSP,
                C1,
                C2,
                C2f,
                C3k2,
                C2fAttn,
                C3,
                C3TR,
                C3Ghost,
                C3x,
                RepC3,
                C2fPSA,
                C2fCIB,
                C2PSA,
                A2C2f,
            }
        )
        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
            m = (
                getattr(torch.nn, m[3:])
                if "nn." in m
                else getattr(__import__("torchvision").ops, m[16:])
                if "torchvision.ops." in m
                else globals()[m]
            )  # get module
            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
            n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
            if m in base_modules:
                c1, c2 = ch[f], args[0]
                if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                if m is C2fAttn:  # set 1) embed channels and 2) num heads
                    args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                    args[2] = int(
                        max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

                args = [c1, c2, *args[1:]]
                if m in repeat_modules:
                    args.insert(2, n)  # number of repeats
                    n = 1
                if m is C3k2:  # for M/L/X sizes
                    legacy = False
                    if scale in "mlx":
                        args[3] = True
                if m is A2C2f:
                    legacy = False
                    if scale in "lx":  # for L/X sizes
                        args.extend((True, 1.2))
                if m is C2fCIB:
                    legacy = False
            elif m is AIFI:
                args = [ch[f], *args]
            elif m in frozenset({HGStem, HGBlock}):
                c1, cm, c2 = ch[f], args[0], args[1]
                args = [c1, cm, c2, *args[2:]]
                if m is HGBlock:
                    args.insert(4, n)  # number of repeats
                    n = 1
            elif m is ResNetLayer:
                c2 = args[1] if args[3] else args[1] * 4
            elif m is torch.nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m in frozenset(
                    {Detect, WorldDetect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB, ImagePoolingAttn, v10Detect}
            ):
                args.append([ch[x] for x in f])
                if m is Segment or m is YOLOESegment:
                    args[2] = make_divisible(min(args[2], max_channels) * width, 8)
                if m in {Detect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB}:
                    m.legacy = legacy
            elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
                args.insert(1, [ch[x] for x in f])
            elif m is CBLinear:
                c2 = args[0]
                c1 = ch[f]
                args = [c1, c2, *args[1:]]
            elif m is CBFuse:
                c2 = ch[f[-1]]
            elif m in frozenset({TorchVision, Index}):
                c2 = args[0]
                c1 = ch[f]
                args = [*args[1:]]
            else:
                c2 = ch[f]

            m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace("__main__.", "")  # module type
            m_.np = sum(x.numel() for x in m_.parameters())  # number params
            m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
            if verbose:
                LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return torch.nn.Sequential(*layers), sorted(save)