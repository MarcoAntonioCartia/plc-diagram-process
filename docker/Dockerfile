FROM nvcr.io/nvidia/pytorch:23.10-py3
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
CMD ["python", "src/detection/yolov8_train.py"]