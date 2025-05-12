# https://docs.ultralytics.com/modes/train/
# https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/
from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")  # pretrained nano model:contentReference[oaicite:0]{index=0}
    results = model.train(
        data="../../data/plc_symbols.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="plc_symbol_detector"
    )
    print("Training finished:", results)

if __name__ == "__main__":
    train()