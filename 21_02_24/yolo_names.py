from ultralytics import YOLO

model = YOLO('yolov8x-seg.pt')
print(model.names)