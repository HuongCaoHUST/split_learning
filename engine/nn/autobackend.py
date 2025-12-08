from ultralytics.nn.autobackend import AutoBackend
from engine.model.model import SplitDetectionModel
from typing import Any, List, Optional, Union
from ultralytics.nn.tasks import attempt_load_weights
import numpy as np
import torch
from PIL import Image

class AutoBackend_SplitModel(AutoBackend):
    def __init__(self, *args, layer_id=None, cut_layer=None, **kwargs):
        self.layer_id = layer_id
        self.cut_layer = cut_layer
        super().__init__(*args, **kwargs)

        if self.layer_id == 2:
            self.model.__class__ = SplitDetectionModel

    def forward(
        self,
        im: torch.Tensor,
        augment: bool = False,
        visualize: bool = False,
        embed: Optional[List] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Run inference on an AutoBackend model.

        Args:
            im (torch.Tensor): The image tensor to perform inference on.
            augment (bool): Whether to perform data augmentation during inference.
            visualize (bool): Whether to visualize the output predictions.
            embed (list, optional): A list of feature vectors/embeddings to return.
            **kwargs (Any): Additional keyword arguments for model configuration.

        Returns:
            (torch.Tensor | List[torch.Tensor]): The raw output tensor(s) from the model.
        """
        if self.layer_id == 1:
            b, ch, h, w = im.shape  # batch, channel, height, width
        else:
            b, ch, h, w = im[0].shape  # batch, channel, height, width

        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        # PyTorch
        if self.pt or self.nn_module:
            if self.layer_id == 1:
                outputs = []
                x = im
                for i, m in enumerate(self.model.model):
                    x = m(x)
                    if i == 4 or i == 5:
                        outputs.append(x)
                y = outputs
            else:
                y = self.model(im, augment=augment, visualize=visualize, embed=embed, **kwargs)
                

        # TorchScript
        elif self.jit:
            y = self.model(im)

        # ONNX OpenCV DNN
        elif self.dnn:
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()

        # ONNX Runtime
        elif self.onnx or self.imx:
            if self.dynamic:
                im = im.cpu().numpy()  # torch to numpy
                y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
            else:
                if not self.cuda:
                    im = im.cpu()
                self.io.bind_input(
                    name="images",
                    device_type=im.device.type,
                    device_id=im.device.index if im.device.type == "cuda" else 0,
                    element_type=np.float16 if self.fp16 else np.float32,
                    shape=tuple(im.shape),
                    buffer_ptr=im.data_ptr(),
                )
                self.session.run_with_iobinding(self.io)
                y = self.bindings
            if self.imx:
                # boxes, conf, cls
                y = np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None]], axis=-1)

        # OpenVINO
        elif self.xml:
            im = im.cpu().numpy()  # FP32

            if self.inference_mode in {"THROUGHPUT", "CUMULATIVE_THROUGHPUT"}:  # optimized for larger batch-sizes
                n = im.shape[0]  # number of images in batch
                results = [None] * n  # preallocate list with None to match the number of images

                def callback(request, userdata):
                    """Place result in preallocated list using userdata index."""
                    results[userdata] = request.results

                # Create AsyncInferQueue, set the callback and start asynchronous inference for each input image
                async_queue = self.ov.AsyncInferQueue(self.ov_compiled_model)
                async_queue.set_callback(callback)
                for i in range(n):
                    # Start async inference with userdata=i to specify the position in results list
                    async_queue.start_async(inputs={self.input_name: im[i : i + 1]}, userdata=i)  # keep image as BCHW
                async_queue.wait_all()  # wait for all inference requests to complete
                y = np.concatenate([list(r.values())[0] for r in results])

            else:  # inference_mode = "LATENCY", optimized for fastest first result at batch-size 1
                y = list(self.ov_compiled_model(im).values())

        # TensorRT
        elif self.engine:
            if self.dynamic and im.shape != self.bindings["images"].shape:
                if self.is_trt10:
                    self.context.set_input_shape("images", im.shape)
                    self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                    for name in self.output_names:
                        self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
                else:
                    i = self.model.get_binding_index("images")
                    self.context.set_binding_shape(i, im.shape)
                    self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                    for name in self.output_names:
                        i = self.model.get_binding_index(name)
                        self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))

            s = self.bindings["images"].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]

        # CoreML
        elif self.coreml:
            im = im[0].cpu().numpy()
            im_pil = Image.fromarray((im * 255).astype("uint8"))
            # im = im.resize((192, 320), Image.BILINEAR)
            y = self.model.predict({"image": im_pil})  # coordinates are xywh normalized
            if "confidence" in y:
                raise TypeError(
                    "Ultralytics only supports inference of non-pipelined CoreML models exported with "
                    f"'nms=False', but 'model={w}' has an NMS pipeline created by an 'nms=True' export."
                )
                # TODO: CoreML NMS inference handling
                # from ultralytics.utils.ops import xywh2xyxy
                # box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                # conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float32)
                # y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            y = list(y.values())
            if len(y) == 2 and len(y[1].shape) != 4:  # segmentation model
                y = list(reversed(y))  # reversed for segmentation models (pred, proto)

        # PaddlePaddle
        elif self.paddle:
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]

        # MNN
        elif self.mnn:
            input_var = self.torch_to_mnn(im)
            output_var = self.net.onForward([input_var])
            y = [x.read() for x in output_var]

        # NCNN
        elif self.ncnn:
            mat_in = self.pyncnn.Mat(im[0].cpu().numpy())
            with self.net.create_extractor() as ex:
                ex.input(self.net.input_names()[0], mat_in)
                # WARNING: 'output_names' sorted as a temporary fix for https://github.com/pnnx/pnnx/issues/130
                y = [np.array(ex.extract(x)[1])[None] for x in sorted(self.net.output_names())]

        # NVIDIA Triton Inference Server
        elif self.triton:
            im = im.cpu().numpy()  # torch to numpy
            y = self.model(im)

        # RKNN
        elif self.rknn:
            im = (im.cpu().numpy() * 255).astype("uint8")
            im = im if isinstance(im, (list, tuple)) else [im]
            y = self.rknn_model.inference(inputs=im)

        # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
        else:
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model.serving_default(im)
                if not isinstance(y, list):
                    y = [y]
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                details = self.input_details[0]
                is_int = details["dtype"] in {np.int8, np.int16}  # is TFLite quantized int8 or int16 model
                if is_int:
                    scale, zero_point = details["quantization"]
                    im = (im / scale + zero_point).astype(details["dtype"])  # de-scale
                self.interpreter.set_tensor(details["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if is_int:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    if x.ndim == 3:  # if task is not classification, excluding masks (ndim=4) as well
                        # Denormalize xywh by image size. See https://github.com/ultralytics/ultralytics/pull/1695
                        # xywh are normalized in TFLite/EdgeTPU to mitigate quantization error of integer models
                        if x.shape[-1] == 6 or self.end2end:  # end-to-end model
                            x[:, :, [0, 2]] *= w
                            x[:, :, [1, 3]] *= h
                            if self.task == "pose":
                                x[:, :, 6::3] *= w
                                x[:, :, 7::3] *= h
                        else:
                            x[:, [0, 2]] *= w
                            x[:, [1, 3]] *= h
                            if self.task == "pose":
                                x[:, 5::3] *= w
                                x[:, 6::3] *= h
                    y.append(x)
            # TF segment fixes: export is reversed vs ONNX export and protos are transposed
            if len(y) == 2:  # segment with (det, proto) output order reversed
                if len(y[1].shape) != 4:
                    y = list(reversed(y))  # should be y = (1, 116, 8400), (1, 160, 160, 32)
                if y[1].shape[-1] == 6:  # end-to-end model
                    y = [y[1]]
                else:
                    y[1] = np.transpose(y[1], (0, 3, 1, 2))  # should be y = (1, 116, 8400), (1, 32, 160, 160)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]

        # for x in y:
        #     print(type(x), len(x)) if isinstance(x, (list, tuple)) else print(type(x), x.shape)  # debug shapes
        if isinstance(y, (list, tuple)):
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):  # segments and names not defined
                nc = y[0].shape[1] - y[1].shape[1] - 4  # y = (1, 32, 160, 160), (1, 116, 8400)
                self.names = {i: f"class{i}" for i in range(nc)}
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return y