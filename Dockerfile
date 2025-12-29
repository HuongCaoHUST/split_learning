FROM ultralytics/ultralytics:latest

RUN pip install --upgrade pip

RUN pip install \
    requests==2.32.3 \
    pika==1.3.2\
    pandas==2.2.3\
    mlflow==2.13.0

CMD ["bash"]

ENV PYTHONUNBUFFERED=1