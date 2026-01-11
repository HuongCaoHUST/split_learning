import pickle

class SendService:
    def __init__(self, rabbitmq_connection):
        self.rabbitmq = rabbitmq_connection

    def send_number_batch_client_id(self, nb=None, client_id=None, client_cut_layer=None, tensor_send_ids=None):
        queue_name = 'number_batch_queue'
        self.rabbitmq.declare_queue(queue_name, durable=False)

        message = {
            "nb": nb,
            "client_id": client_id,
            "client_cut_layer": client_cut_layer,
            "tensor_send_ids": tensor_send_ids
        }

        self.rabbitmq.publish(
            exchange='',
            routing_key=queue_name,
            message=message
        )
        print(f"Number batch đã được gửi tới {queue_name}")
        return True

    def send_to_server(self, message):
        queue_name = 'Server_queue'
        self.rabbitmq.declare_queue(queue_name, durable=False)
        self.rabbitmq.publish(routing_key=queue_name, message=message)
        return None

    def send_to_intermediate_queue(self, layer_id, data_id, data_store, label):
        queue_name = f'intermediate_queue_{layer_id}'
        self.rabbitmq.declare_queue(queue_name, durable=False)

        message = {
            "data_id": data_id,
            "data_store": data_store,
            "label": label
        }

        self.rabbitmq.publish(
            exchange='',
            routing_key=queue_name,
            message=message
        )
        return True