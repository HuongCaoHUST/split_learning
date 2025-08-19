import torch
import time
import pickle
import threading
import uuid
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.plotting import feature_visualization

class Split_Learning_DetectionModel(DetectionModel):
    def __init__(self, cfg=None, nc=None, ch=3, verbose=True, 
                 layer_id=None, client_id=None, num_client=None, cut_layer=None,
                 address=None, username=None, password=None):
        self.layer_id = layer_id
        self.client_id = client_id
        self.num_client = num_client
        self.cut_layer = cut_layer
        self.cut_layer_ids = None
        # RabbitMQ
        self.address = address
        self.username = username
        self.password = password

        self.is_training = False
        self.client_ids = None
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

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
        data_store = {}
        start_layer = self.cut_layer + 1 if self.is_training and self.layer_id == 2 else 0

        max_retries = 1000
        retry_delay = 1
        if self.is_training and self.layer_id == 2:
            queue_name = f'intermediate_queue_{self.layer_id - 1}'
            for attempt in range(max_retries):
                method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, auto_ack=True)
                if method_frame and body:
                    try:
                        received_data = pickle.loads(body)
                        data_store = received_data.get('data_store', {})
                        self.input_data_id = received_data.get('data_id', 'unknown')
                        client_id = self.input_data_id.split("_")[0]

                        client_id = uuid.UUID(client_id)
                        if client_id in self.client_ids:
                            index = self.client_ids.index(client_id)
                            start_layer = self.cut_layer_ids[index] + 1
                            selected_tensor_id = self.tensor_send_ids[index]

                        print("Start layer: ", start_layer)
                        print("Selectes_tensor_id: ", selected_tensor_id)
                        if not any(tid in data_store for tid in selected_tensor_id):
                            raise ValueError("Layer 2 output not found in data_store")
                        tensor_id = next(iter(selected_tensor_id))
                        x = data_store[tensor_id]
                        if not isinstance(x, torch.Tensor):
                            raise ValueError("Data from queue is not a valid tensor")

                        self.saved_tensor = {}
                        y = [None] * len(self.model)

                        # Vòng lặp gán Tensor
                        for tensor_id in selected_tensor_id:
                            if tensor_id not in data_store:
                                raise ValueError(f"Expected tensor_id {tensor_id} not found in data_store")
                            x = data_store[tensor_id]
                            if not isinstance(x, torch.Tensor):
                                raise ValueError(f"Data for tensor_id {tensor_id} is not a valid tensor")
                            print(f"Received tensor_id {tensor_id}, shape: {x.shape}")

                            x = x.detach().clone().requires_grad_(True)
                            self.saved_tensor[tensor_id] = x
                            y[tensor_id] = x
                        
                        print(f"Received TENSOR data_id: {self.input_data_id}")
                        break
                    except (pickle.UnpicklingError, ValueError) as e:
                        print(f"Error processing queue data: {e}")
                        if attempt == max_retries - 1:
                            raise RuntimeError("Failed to process data from queue after max retries")
                else:
                    # print(f"No data received from queue, attempt {attempt + 1}/{max_retries}")
                    if attempt == max_retries - 1:
                        raise RuntimeError("Queue is empty after max retries")
                    time.sleep(retry_delay)
            else:
                raise RuntimeError("Failed to retrieve data from queue")
        else:
            y = [None] * len(self.model)
            
        print("Self.cut_layer_2: ", self.cut_layer)
        for m in self.model[start_layer:]:
            if m.i == self.cut_layer + 1  and self.layer_id == 1:
                # print(f"Cut layer {m.i} reached, stopping forward pass.")
                break
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [y[j] if j != -1 else x for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)

            # print("M.F:", m.f)
            
            x = m(x)
            if m.i in self.save:
                y[m.i] = x
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

            if self.is_training and m.i in self.tensor_send_ids and self.layer_id == 1:
                # data_store[m.i] = x.detach().clone().requires_grad_(True)
                data_store[m.i] = x.detach().requires_grad_(True)
                print(f"Shape of detached tensor at layer {m.i}: {x.detach().shape}")

            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

        if self.is_training and self.layer_id == 1:
            self.data_store = data_store
            data_id = f"{self.client_id}_{uuid.uuid4()}"
            success = self.send_to_intermediate_queue(data_id, data_store)
            if not success:
                print(f"Không thể gửi data_store tới intermediate_queue.")

        self.end_batch_forward_time = time.time()
        return x
    
    def send_to_intermediate_queue(self, data_id, data_store):
        queue_name = f'intermediate_queue_{self.layer_id}'
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
        return True
    
    def get_tensor_send_id (self, cut_layer):
        tensor_send_id = []
        mf_values = []
        layer_indices = []
        for idx, m in enumerate(self.model):
            f = m.f
            if f != -1:
                if isinstance(f, int):
                    f = [f]
                for fi in f:
                    if fi != -1:
                        layer_indices.append(idx)
                        mf_values.append(fi)
        mf_values_sorted = sorted(mf_values)

        for value in mf_values_sorted:
            if value < cut_layer:
                tensor_send_id.append(value)

        indices_to_mf = dict(zip(layer_indices, mf_values))
        for idx, val in indices_to_mf.items():
            if idx <=cut_layer:
                tensor_send_id.remove(val)

        tensor_send_id.append(cut_layer)
        print ("SEND tensor id: ", tensor_send_id)
        return tensor_send_id
    
    
    def start_thread(self, forward_queue):
        """START THREADING"""
        thread = threading.Thread(target=self.check_foward, args= (forward_queue,), daemon=True)
        thread.start()

    def stop_thread(self):
        """STOP THREADING"""
        self.model.is_training = False
        print(f"Thread đã dừng.")

    def check_foward(self, forward_queue):
        queue_name = f'intermediate_queue_{self.layer_id - 1}'
        while True:
            try:
                if self.channel is not None and self.channel.is_open:
                    method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, auto_ack=True)
                    if method_frame and body:
                        received_data = pickle.loads(body)
                        data_id = received_data.get('data_id', {})
                        print("DATA_ID: ", data_id)
                else:
                    print("Thread channel is None or closed")
            except Exception as e:
                print("Error in check_forward thread:", e)
                break
            time.sleep(0.2)

class Split_Learning_SegmentationModel(Split_Learning_DetectionModel):
    """
    YOLO segmentation model for Split Learning.
    """

    def __init__(self, cfg=None, nc=None, ch=3, verbose=True,
                 layer_id=None, client_id=None, num_client=None, cut_layer=None,
                 address=None, username=None, password=None):

        super().__init__(cfg=cfg, nc=nc, ch=ch, verbose=verbose,
                         layer_id=layer_id, client_id=client_id,
                         num_client=num_client, cut_layer=cut_layer,
                         address=address, username=username, password=password)

    def init_criterion(self):
        """Initialize the loss criterion for the Split Learning SegmentationModel."""
        return v8SegmentationLoss(self)

    