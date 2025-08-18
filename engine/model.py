import torch
import time
import pickle
import threading
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.plotting import feature_visualization

class Split_Learning_DetectionModel(DetectionModel):
    def __init__(self, cfg=None, nc=None, ch=3, verbose=True, 
                 layer_id=None, client_id=None, num_client=None, cut_layer=None,
                 address=None, username=None, password=None):
        super().__init__()
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
        self.tensor_send_ids = self.get_tensor_send_id(self.cut_layer) if self.layer_id == 1 else []
        self.data_store=None
        self.input_data_id = None

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        print("Hế lô, tôi là Split_Learning_DetectionModel")
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
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