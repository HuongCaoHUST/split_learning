# Split Learning

![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![RabbitMQ](https://img.shields.io/badge/RabbitMQ-FF6600?style=for-the-badge&logo=rabbitmq&logoColor=white)

**Split learning** is a method of distributed deep learning training, where a machine learning model is split into multiple parts and trained across multiple machines or devices without needing to share the full data or model. This ensures the security and privacy of data, particularly in applications requiring privacy protection such as healthcare, finance, or when devices lack the computational resources to train the full model.

### How Split Learning Works:
- **Model splitting**: The deep learning model is divided into two (or more) parts. For example, part of the model may run on the user device (edge device), while the controller function run on the server.
- **Communication process**:
  - The user device trains a portion of the model based on its local data or data from another device and sends the intermediate activations to the next layer's machine instead of sending the full data.
  - The clients at the last layers will calculate the backward pass and return the corresponding gradients to the user device to update the part of the model. This process repeats until the model completes training.
  - After completing a training round, the server receives the parameters from the clients and continues distributing the new model to the necessary clients for the next round.

### Advantages of Split Learning:
- **Privacy protection**: Sensitive data does not need to be fully transferred between devices and servers. Instead, only the intermediate activations are shared.
- **Resource-saving**: Devices with limited computational resources (such as mobile phones) can share the load and jointly train a smaller portion of the model compared to the original DNN.
- **Model scalability**: Split learning allows the model to be split across multiple devices, extending the capability to handle larger models.

## Deployment Model

![deploy_model](pics/deploy_model.png)

To deploy the service, each blocks in the system (server and clients) need to ensure the following requirements:
- All must be successfully connected to the same queue server, which serves as the main communication hub for the entire system.
- The server must be correctly configured with the topology and the DNN model loaded.
- The clients must synchronize with each other when the model is split and joined according to the topology agreed upon with the server.

Clients training flow:

![sl_model](pics/sl_model.png)

## Required Packages
```
torch
torchvision
pika
tqdm
pyyaml
```

Set up a RabbitMQ server for message communication over the network environment. `docker-compose.yaml` file:

```yaml
version: '3'

services:
  rabbitmq:
    image: rabbitmq:management
    container_name: rabbitmq
    ports:
      - "5672:5672"   # RabbitMQ main port
      - "15672:15672" # Management UI
    environment:
      RABBITMQ_DEFAULT_USER: user
      RABBITMQ_DEFAULT_PASS: password
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
volumes:
  rabbitmq_data:
    driver: local
```

Then run the RabbitMQ container

```commandline
docker-compose up -d
```

## Configuration

Application configuration is in the `config.yaml` file:

```yaml
name: Split Learning
server:   # server configuration
  num-round: 1  # number of training rounds
  cut_layers:   # index of cutting layers 
    - 10
    - 20
  clients:  # Layer 1 has 3 clients, layer 2 has 2 clients, layer 3 has 1 client
    - 3
    - 2
    - 1
  model: VGG16      # class name of DNN model
  parameters:
    load: False     # allow to load parameters file
    save: False     # allow to save parameters file
                    # if turn on, server will be averaging all parameters
  validation: True  # allow to validate on server-side

rabbit:   # RabbitMQ connection configuration
  address: 127.0.0.1    # address
  username: admin
  password: admin

log_path: .   # logging directory

learning:
  learning-rate: 0.01
  momentum: 0.5
  batch-size: 256
  control-count: 3    # control count on client
```

This configuration is use for server.

### List of DNN model

For `server.model` field:

```
VGG16
```

## How to Run

Alter your configuration, you need to run the server to listen and control the request from clients.

### Server

```commandline
python server.py
```

### Client

Now, when server is ready, run clients simultaneously with total number of client that you defined.

**Layer 1**

```commandline
python client.py --layer_id 1
```

Where:
- `--layer_id` is the ID index of client's layer, start from 1.

If you want to use a specific device configuration for the training process, declare it with the `--device` argument when running the command line:

```commandline
python client.py --layer_id 1 --device cpu
```

## Parameter Files

On the server, the `*.pth` files are saved in the main execution directory of `server.py` after completing one training round.

If the `*.pth` file exists, the server will read the file and send the parameters to the clients. Otherwise, if the file does not exist, a new DNN model will be created with fresh parameters. Therefore, if you want to reset the training process, you should delete the `*.pth` files.

---

Version 2.0.0

The application is under development...

TODO:
- Inference mode and training mode
- Delete all queues alter finish
- Create close connection request
