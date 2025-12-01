import torch
import time
import pickle
import threading
import uuid
from copy import deepcopy
from engine.yolo.model import Split_Learning_DetectionModel
from ultralytics.utils.plotting import feature_visualization

class Split_Learning_RTDETRDetectionModel(Split_Learning_DetectionModel):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class For Split Learning
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True,
                 layer_id=None, client_id=None, num_client=None, cut_layer=None,
                 address=None, username=None, password=None, load_partial_model = False):
        """
        Initialize the RTDETRDetectionModel.

        Args:
            cfg (str | dict): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Print additional information during initialization.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose,
                         layer_id=layer_id, client_id=client_id,
                         num_client=num_client, cut_layer=cut_layer,
                         address=address, username=username, password=password, load_partial_model=load_partial_model)

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions.

        Returns:
            loss_sum (torch.Tensor): Total loss value.
            loss_items (torch.Tensor): Main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch["batch_idx"]
        if self.is_training:
            gt_groups = batch["gt_groups"]
        else:
            gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            batch (dict, optional): Ground truth data for evaluation.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        data_store = {}
        self.saved_tensor = {}

        if self.layer_id == 1:
            for m in self.model:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if profile:
                    self._profile_one_layer(m, x, dt)
                    
                x = m(x)  # run
                print(f"Shape of detached tensor at layer {m.i}: {x.detach().shape}")
                if m.i == self.cut_layer or m.i == 7 or m.i == 3:
                    data_store[m.i] = x.detach().requires_grad_(True)

                y.append(x if m.i in self.save else None)  # save output
                if visualize:
                    feature_visualization(x, m.type, m.i, save_dir=visualize)
                if m.i in embed:
                    embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                    if m.i == max_idx:
                        return torch.unbind(torch.cat(embeddings, 1), dim=0)
                    
            data_id = f"{self.client_id}_{self.batch_id}"
            self.data_store = data_store
            success = self.send_to_intermediate_queue(data_id, data_store)
            if not success:
                print(f"Không thể gửi data_store tới intermediate_queue.")
            return x
        elif self.layer_id == 2 and self.is_training == True:
            start_layer = self.cut_layer + 1
            queue_name = f'intermediate_queue_{self.layer_id - 1}'
            y = [None] * 10

            while self.is_training:
                method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, auto_ack=True)
                if method_frame and body:
                    try:
                        received_data = pickle.loads(body)
                        data_store = received_data.get('data_store', {})
                        self.input_data_id = received_data.get('data_id', 'unknown')
                        x = data_store[9]
                        y[7]= data_store[7]
                        y[3]= data_store[3]

                        x = x.requires_grad_(True)
                        self.saved_tensor[9] = x

                        self.saved_tensor[7] = y[7].requires_grad_(True)
                        self.saved_tensor[3] = y[3].requires_grad_(True)

                        break
                    except (pickle.UnpicklingError, ValueError) as e:
                        print(f"Error processing queue data: {e}")

            for m in self.model[start_layer:-1]:  # except the head part
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if profile:
                    self._profile_one_layer(m, x, dt)
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
                if visualize:
                    feature_visualization(x, m.type, m.i, save_dir=visualize)
                if m.i in embed:
                    embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                    if m.i == max_idx:
                        return torch.unbind(torch.cat(embeddings, 1), dim=0)
            head = self.model[-1]
            x = head([y[j] for j in head.f], batch)  # head inference
            return x
        
        else: 
            for m in self.model[:-1]:  # except the head part
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if profile:
                    self._profile_one_layer(m, x, dt)
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
                if visualize:
                    feature_visualization(x, m.type, m.i, save_dir=visualize)
                if m.i in embed:
                    embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                    if m.i == max_idx:
                        return torch.unbind(torch.cat(embeddings, 1), dim=0)
            head = self.model[-1]
            x = head([y[j] for j in head.f], batch)  # head inference
            return x
        
