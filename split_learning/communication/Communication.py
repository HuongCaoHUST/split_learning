import pika
import pickle
import time
import requests
from requests.auth import HTTPBasicAuth
import src.Log

class RabbitMQConnection:
    def __init__(self, address, username, password, port=5672, virtual_host='/'):
        self.address = address
        self.username = username
        self.password = password
        self.port = port
        self.virtual_host = virtual_host
        self.connection = None
        self.channel = None
        self.credentials = pika.PlainCredentials(self.username, self.password)
        self.parameters = pika.ConnectionParameters(
            host=self.address,
            port=self.port,
            virtual_host=self.virtual_host,
            credentials=self.credentials
        )

    def connect(self, blocking=True, retry_interval=1):
        while True:
            try:
                self.connection = pika.BlockingConnection(self.parameters)
                self.channel = self.connection.channel()
                return self.channel
            except pika.exceptions.AMQPConnectionError as e:
                if not blocking:
                    raise e
                time.sleep(retry_interval)

    def close(self):
        if self.connection and not self.connection.is_closed:
            self.connection.close()

    def get_channel(self):
        if self.connection is None or self.connection.is_closed:
            self.connect()
        if self.channel is None or self.channel.is_closed:
             self.channel = self.connection.channel()
        return self.channel

    def declare_queue(self, queue_name, durable=False):
        channel = self.get_channel()
        return channel.queue_declare(queue=queue_name, durable=durable)

    def publish(self, routing_key, message, exchange=''):
        channel = self.get_channel()
        channel.basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            body=pickle.dumps(message) if not isinstance(message, bytes) else message
        )

    def consume(self, queue_name, callback, auto_ack=False, prefetch_count=1):
        channel = self.get_channel()
        channel.basic_qos(prefetch_count=prefetch_count)
        channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=auto_ack)

    def start_consuming(self):
        if self.channel:
            self.channel.start_consuming()

    def get_message(self, queue_name, auto_ack=True):
        channel = self.get_channel()
        return channel.basic_get(queue=queue_name, auto_ack=auto_ack)

    def delete_queue(self, queue_name):
        try:
            channel = self.get_channel()
            channel.queue_delete(queue=queue_name)
            return True
        except Exception:
            return False

    @staticmethod
    def delete_old_queues(address, username, password, port=15672):
        url = f'http://{address}:{port}/api/queues'
        while True:
            try:
                response = requests.get(url, auth=HTTPBasicAuth(username, password))
                if response.status_code == 200:
                    break
                else:
                    src.Log.print_with_color(f"⚠️ Waiting for RabbitMQ API... Status: {response.status_code}", "yellow")
            except requests.exceptions.ConnectionError:
                src.Log.print_with_color("⏳ Waiting for RabbitMQ HTTP API to be ready...", "yellow")
            time.sleep(1)

        if response.status_code == 200:
            queues = response.json()
            # Use a temporary connection to delete queues
            temp_conn = RabbitMQConnection(address, username, password)
            channel = temp_conn.connect()

            for queue in queues:
                queue_name = queue['name']
                if queue_name.startswith("reply") or queue_name.startswith("intermediate_queue") or queue_name.startswith(
                        "gradient_queue") or queue_name.startswith("label_queue"):
                    try:
                        channel.queue_delete(queue=queue_name)
                        src.Log.print_with_color(f"Queue '{queue_name}' deleted.", "green")
                    except Exception as e:
                        src.Log.print_with_color(f"Failed to delete queue '{queue_name}': {e}", "yellow")
            temp_conn.close()
            return True
        else:
            src.Log.print_with_color(
                f"Failed to fetch queues from RabbitMQ Management API. Status code: {response.status_code}", "yellow")
            return False