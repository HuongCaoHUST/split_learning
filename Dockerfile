# Dockerfile
FROM python:3.11-slim

# Cài thư viện build cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set thư mục làm việc
WORKDIR /app

# Copy toàn bộ project
COPY . .

# Copy file requirements.txt từ environment.yaml (phần pip)
COPY requirements.txt .

# Cài thư viện Python
RUN pip install --upgrade pip && pip install -r requirements.txt

# Khi container start, vào bash để người dùng tương tác
CMD ["bash"]
