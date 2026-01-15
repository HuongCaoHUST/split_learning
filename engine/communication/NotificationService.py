import pickle
import src.Log

class NotificationService:
    def __init__(self, rabbitmq_connection):
        self.rabbitmq_conn = rabbitmq_connection

    def send_to_client(self, client_id, message):
        reply_queue_name = f'reply_{client_id}'
        self.rabbitmq_conn.declare_queue(reply_queue_name, durable=False)
        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        self.rabbitmq_conn.publish(
            routing_key=reply_queue_name,
            message=pickle.dumps(message)
        )

    def notify_to_all_clients(self, list_clients, message):
        for (client_id, _) in list_clients:
            src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
            self.send_to_client(client_id, message)
