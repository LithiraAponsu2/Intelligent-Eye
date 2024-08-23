from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n-seg.pt")

# Run the evaluation
results = model.val()

# # Print specific metrics
# print("Class indices with average precision:", results.ap_class_index)
# print("Average precision for all classes:", results.box.all_ap)
# print("Mean average precision at IoU=0.50:", results.box.map50)
# print("Mean recall:", results.box.mr)



