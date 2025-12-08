from ultralytics.models.yolo.detect import DetectionPredictor
from src import Utils
from pathlib import Path
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from engine.nn.autobackend import AutoBackend_SplitModel
from ultralytics.utils.files import increment_path
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
import cv2
import torch
import pickle
import uuid
import re
from typing import List
import gc
import time
import numpy as np
from types import SimpleNamespace

class Split_Learning_DetectionPredictor(DetectionPredictor):
    def __init__(self, overrides, client_id=None, layer_id=None, num_client=None, cut_layer=None, address=None, username=None, password=None, load_partial_model=False):
        self.client_id = client_id
        self.layer_id = layer_id
        self.num_client = num_client
        self.client_ids = []
        self.cut_layer = cut_layer
        self.address = address
        self.username = username
        self.password = password
        self.load_partial_model = load_partial_model
        Utils.init_csv('./log/latency.csv', ['batch_id', 'start', 'end', 'latency'])
        Utils.init_csv('./log/com_cost.csv', ['batch_id', 'label/tensor', 'size'])
        super().__init__(overrides=overrides)
        self.channel = Utils.connect_rabbitmq(self.address, self.username, self.password)
        self.speed = {}
        self.is_inference = False
        self.is_done = False

    def inference(self, im: torch.Tensor, *args, **kwargs):
        """Run inference on a given image using the specified model and arguments."""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        output = self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)
        return output

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """
        Split inference stream method for YOLO detection models.
        """
        self.is_inference = True
        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            if self.layer_id == 1:
                self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                if self.imgsz is None and self.layer_id == 2:
                    self.imgsz = (640, 640)
                self.model.warmup(
                    imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, self.model.ch, *self.imgsz)
                )
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")

            if self.layer_id == 1:
                for self.batch in self.dataset:
                    self.run_callbacks("on_predict_batch_start")
                    paths, im0s, s = self.batch
                    # Preprocess
                    with profilers[0]:
                        im = self.preprocess(im0s)

                    # Inference
                    with profilers[1]:
                        preds = self.inference(im, *args, **kwargs)

                        if self.layer_id == 1:
                            success = self.send_to_intermediate_queue(preds, paths, s, self.dataset.count)
                            if not success:
                                print(f"Sending to intermediate queue failed.")

                        if self.args.embed:
                            yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                            continue
                    
                    n = len(im0s)
                    try:
                        for i in range(n):
                            self.seen += 1
                            self.speed = {
                                "preprocess": profilers[0].dt * 1e3 / n,
                                "inference": profilers[1].dt * 1e3 / n,
                            }
                            if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                                s[i] += self.write_results(i, Path(paths[i]), im, s)
                    except StopIteration:
                        break

                    # Print batch results
                    if self.args.verbose:
                        LOGGER.info("\n".join(s))
                    self.run_callbacks("on_predict_batch_end")
                
                self.send_to_intermediate_queue(is_done=True)

            if self.layer_id == 2:
                self.count = None
                self.source_type = SimpleNamespace(
                    stream=False,
                    screenshot=False,
                    from_img=False,
                    tensor=False,
                )
                self.dataset = SimpleNamespace(mode="video", fps=30)
                while True:
                    self.run_callbacks("on_predict_batch_start")

                    # Receive data from intermediate queue
                    with profilers[0]:
                        data_id, p3, p5, paths, s, is_done = self.wait_for_intermediate_queue()
                        if is_done:
                            break

                        p3 = torch.tensor(p3, dtype=torch.float32, device=self.device)
                        p5 = torch.tensor(p5, dtype=torch.float32, device=self.device)

                        im = [p3, p5]
                        im0s = [np.zeros((1080, 810, 3), dtype=np.uint8) for _ in range(16)]
                    
                        self.batch = (paths, im0s, s)

                    # Inference
                    with profilers[1]:
                        preds = self.inference(im, *args, **kwargs)
                        if self.args.embed:
                            yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                            continue
                    
                    # Postprocess
                    with profilers[2]:
                        im = self.preprocess(im0s)
                        self.results = self.postprocess(preds, im, im0s)
                    self.run_callbacks("on_predict_postprocess_end")

                    # Visualize, save, write results
                    n = len(self.results)
                    try:
                        for i in range(1):
                            self.seen += 1
                            self.results[i].speed = {
                                "preprocess": profilers[0].dt * 1e3 / n,
                                "inference": profilers[1].dt * 1e3 / n,
                                "postprocess": profilers[2].dt * 1e3 / n,
                            }
                            if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                                s[i] += self.write_results(i, Path(paths[i]), im, s)
                    except StopIteration:
                        break

                    # Print batch results
                    if self.args.verbose:
                        LOGGER.info("\n".join(s))

                    self.run_callbacks("on_predict_batch_end")
                    yield from self.results

        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        if self.args.show:
            cv2.destroyAllWindows()  # close any open windows

        # Print final results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), getattr(self.model, 'ch', 3), *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")

    def setup_model(self, model, verbose: bool = True):
        """
        Initialize YOLO model with given parameters and set it to evaluation mode.

        Args:
            model (str | Path | torch.nn.Module, optional): Model to load or use.
            verbose (bool): Whether to print verbose output.
        """
        self.model = AutoBackend_SplitModel(
            weights=model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            batch=self.args.batch,
            fuse=True,
            verbose=verbose,

            layer_id=self.layer_id,
        )

        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        if hasattr(self.model, "imgsz") and not getattr(self.model, "dynamic", False):
            self.args.imgsz = self.model.imgsz  # reuse imgsz from export metadata
        self.model.eval()
    
    def send_to_intermediate_queue(self, preds = None, paths=None, s=None, count=None, is_done=False):
        try:
            data_id = uuid.uuid4()
            queue_name = f'intermediate_queue_{self.layer_id}'
            self.channel.queue_declare(queue_name, durable=False)
            if not is_done:
                data = {
                    "data_id": str(data_id),
                    "p4": preds[0].detach().cpu().numpy(),
                    "p5": preds[1].detach().cpu().numpy(),
                    "paths": paths,
                    "s": s,
                    "count": count,
                }
            else:
                data = {
                    "is_done": True,
                }
            message = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

            self.channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=message
            )

            print(f"Sent {data_id}, size = {len(message)} bytes")

            return True

        finally:
            del preds
            del data
            del message
            gc.collect()

    def wait_for_intermediate_queue(self):
        while self.is_inference:
            queue_name = f'intermediate_queue_{self.layer_id - 1}'
            method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, auto_ack=True)
            if method_frame and body:
                received_data = pickle.loads(body)
                data_id = received_data.get("data_id", None)
                p4 = received_data.get("p4", None)
                p5 = received_data.get("p5", None)
                paths = received_data.get("paths", [])
                s = received_data.get("s", None)
                is_done = received_data.get("is_done", None)
                return data_id, p4, p5, paths, s, is_done
            else:
                time.sleep(0.5)

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """
        Post-process predictions and return a list of Results objects.

        This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
        further analysis.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): List of Results objects containing the post-processed predictions.

        Examples:
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo11n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "_feats", None) is not None
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results


    def write_results(self, i: int, p: Path, im: torch.Tensor, s: List[str]) -> str:
        """
        Write inference results to a file or directory.

        Args:
            i (int): Index of the current image in the batch.
            p (Path): Path to the current image.
            im (torch.Tensor): Preprocessed image tensor.
            s (List[str]): List of result strings.

        Returns:
            (str): String with result information.
        """
        string = ""  # print string
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 0 if frame undetermined
        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])
        if self.layer_id == 2:
            result = self.results[i]
            result.save_dir = self.save_dir.__str__()  # used in other locations
            string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

            # Add predictions to image
            if self.args.save or self.args.show:
                self.plotted_img = result.plot(
                    line_width=self.args.line_width,
                    boxes=self.args.show_boxes,
                    conf=self.args.show_conf,
                    labels=self.args.show_labels,
                    im_gpu=None if self.args.retina_masks else im[i],
                )

            # Save results
            if self.args.save_txt:
                result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
            if self.args.save_crop:
                result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
            if self.args.save:
                self.save_predicted_images(str(self.save_dir / p.name), frame)
        else:
            string += f"{self.speed['inference']:.1f}ms"

        if self.args.show:
            self.show(str(p))

        return string
