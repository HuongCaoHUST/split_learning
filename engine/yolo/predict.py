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

class Split_Learning_DetectionPredictor(DetectionPredictor):
    def __init__(self, overrides, client_id=None, layer_id=None, num_client=None, cut_layer=None, address=None, username=None, password=None, load_partial_model=False):
        self.client_id = client_id
        self.layer_id = 1
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
        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
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
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)

                    success = self.send_to_intermediate_queue_2(preds)
                    if not success:
                        print(f"Sending to intermediate queue failed.")

                    # torch.save({f"feat{i}": p for i, p in enumerate(preds)}, "features_intermediate.pth")
                    # print("Saved pred_features.pth")

                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue
                
                if self.layer_id == 2:
                    # Postprocess
                    with profilers[2]:
                        self.results = self.postprocess(preds, im, im0s)
                    self.run_callbacks("on_predict_postprocess_end")

                    # Visualize, save, write results
                    n = len(im0s)
                    try:
                        for i in range(n):
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
                else:
                    n = len(im0s)
                    try:
                        for i in range(n):
                            self.seen += 1
                            self.speed = {
                                "preprocess": profilers[0].dt * 1e3 / n,
                                "inference": profilers[1].dt * 1e3 / n,
                            }
                            # if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                            #     s[i] += self.write_results(i, Path(paths[i]), im, s)
                    except StopIteration:
                        break
                    # Print batch results
                    if self.args.verbose:
                        LOGGER.info("\n".join(s))

                    self.run_callbacks("on_predict_batch_end")

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

            layer_id=1,
        )

        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        if hasattr(self.model, "imgsz") and not getattr(self.model, "dynamic", False):
            self.args.imgsz = self.model.imgsz  # reuse imgsz from export metadata
        self.model.eval()

    def send_to_intermediate_queue(self, data_store):
        data_id = uuid.uuid4()
        queue_name = f'Server_queue'
        self.channel.queue_declare(queue_name, durable=False)

        message = pickle.dumps(
            {"data_id": data_id,
            "data_store": data_store}
        )

        self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=message
        )
        print(f"Data_store {data_id} đã được gửi tới {queue_name}, Kích thước: {len(message)} bytes")
        data_store = None
        return True
    
    def send_to_intermediate_queue_2(self, preds):
        try:
            data_id = uuid.uuid4()
            queue_name = 'Server_queue'
            self.channel.queue_declare(queue_name, durable=False)

            # Chuyển từng layer thành numpy + float16 để giảm size
            data = {
                "data_id": str(data_id),
                "p3": preds[0].detach().cpu().numpy().astype('float16'),
                "p5": preds[1].detach().cpu().numpy().astype('float16'),
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
