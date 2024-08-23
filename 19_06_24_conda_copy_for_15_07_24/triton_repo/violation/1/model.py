import json
import numpy as np
import triton_python_backend_utils as pb_utils
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import joblib
import cv2
import os
from collections import defaultdict
import base64
import tempfile
from datetime import datetime

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.output_config = pb_utils.get_output_config_by_name(self.model_config, "OUTPUT0")
        self.output_dtype = pb_utils.triton_string_to_numpy(self.output_config["data_type"])
        
        # Load the YOLO model
        root = f'{args["model_repository"]}/{args["model_version"]}'
        self.yolo_model = YOLO(f"{root}/yolov8x-seg.pt")        
        self.tensorrt_model = YOLO(f"{root}/yolov8x-seg.pt")
        self.names = self.yolo_model.names

        # Load your pre-trained model
        self.loaded_model = joblib.load(f"{root}/xgb_model.pkl")

        # Region of interest for client 1
        self.roi_1 = np.array([[550, 664], [579, 723], [1120, 697], [998, 646]], np.int32)
        self.roi_binary_1 = np.zeros((1080, 1920), dtype=np.uint8)
        cv2.fillPoly(self.roi_binary_1, [self.roi_1], color=255)
        self.area_of_roi_1 = cv2.countNonZero(self.roi_binary_1)

        # Region of interest for client 2 (dummy example)
        self.roi_2 = np.array([[550, 664], [579, 723], [1120, 697], [998, 646]], np.int32)
        self.roi_binary_2 = np.zeros((1080, 1920), dtype=np.uint8)
        cv2.fillPoly(self.roi_binary_2, [self.roi_2], color=255)
        self.area_of_roi_2 = cv2.countNonZero(self.roi_binary_2)

        # Calculate line equations from ROI points
        p1_1, p2_1 = self.roi_1[1], self.roi_1[2]
        self.m_1, self.b_1 = self.calculate_line_equation(p1_1, p2_1)

        p1_2, p2_2 = self.roi_2[1], self.roi_2[2]
        self.m_2, self.b_2 = self.calculate_line_equation(p1_2, p2_2)

        # Initialize tracking history
        self.track_history = defaultdict(list)

        # Define a set to track IDs of persons on bikes
        self.persons_on_bikes = set()

    def calculate_line_equation(self, p1, p2):
        if p2[0] - p1[0] == 0:
            m = float('inf')
        else:
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - m * p1[0]
        return m, b

    def is_to_the_right(self, point, m, b):
        if m == float('inf'):
            return point[0] > b
        else:
            return point[1] > m * point[0] + b

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get the input tensors
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            client_number_tensor = pb_utils.get_input_tensor_by_name(request, "CLIENT_NUMBER")

            # Decode the base64 video data
            video_base64 = input_tensor.as_numpy()[0].decode()
            video_bytes = base64.b64decode(video_base64)
            
            # Get client number
            client_number = int(client_number_tensor.as_numpy()[0])

            # Select the appropriate ROI and parameters based on the client number
            if client_number == 1:
                roi_binary = self.roi_binary_1
                area_of_roi = self.area_of_roi_1
                m, b = self.m_1, self.b_1
            else:
                roi_binary = self.roi_binary_2
                area_of_roi = self.area_of_roi_2
                m, b = self.m_2, self.b_2

            # Save the video to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_video.write(video_bytes)
                temp_video_path = temp_video.name

            # Open the video file
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                output_tensor = pb_utils.Tensor("OUTPUT0", np.array([-1], dtype=self.output_dtype))
                inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(inference_response)
                return responses

            # Set save directory based on client number
            save_directory = f"xgb_capt_3/client_{client_number}"  # Modified to include client number
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            # Process video frames
            frame_count = 0
            while True:
                ret, im0 = cap.read()
                if not ret:
                    break  # Exit loop if video frame is empty

                # YOLO Object Detection
                results = self.tensorrt_model.track(im0, persist=True, classes=[0, 2, 3, 5, 7])
                
                # Model Inference
                masks = results[0].masks.xy
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confidences = results[0].boxes.conf.float().cpu().tolist()
                cls_nums = results[0].boxes.cls.int().cpu().tolist()

                annotator = Annotator(im0, line_width=2)

                # Initialize lists to store detected persons and vehicles
                persons = []
                vehicles = []

                # Identify all persons and bikes/motorcycles
                bike_vehicle_centers = []
                for mask, cls_num in zip(masks, cls_nums):
                    if mask.size > 0 and self.names[cls_num] in ['motorcycle']:
                        center = np.mean(mask, axis=0)
                        bike_vehicle_centers.append(center)

                # Data extraction loop
                for mask, track_id, confidence, cls_num in zip(masks, track_ids, confidences, cls_nums):
                    if mask.size == 0:
                        continue  # Skip if mask is empty
                    
                    max_point = max(mask, key=lambda x: (x[0], x[1]))
                    if self.is_to_the_right(max_point, m, b):
                        frame_count -= 1
                        continue  # Skip vehicles passing the upper bound
                    
                    center = np.mean(mask, axis=0)

                    # Check if this person is on a bike
                    if self.names[cls_num] == 'person' and any(np.linalg.norm(center - bike_center) < 100 for bike_center in bike_vehicle_centers):
                        self.persons_on_bikes.add(track_id)
                        continue  # Skip further processing for this person

                    mask_binary = np.zeros_like(roi_binary)
                    cv2.fillPoly(mask_binary, [np.array(mask, dtype=np.int32)], color=255)
                    intersection_binary = cv2.bitwise_and(mask_binary, roi_binary)
                    intersection_area = cv2.countNonZero(intersection_binary)
                    mask_area = cv2.countNonZero(mask_binary)

                    # Calculate IoU
                    iou = intersection_area / (mask_area + area_of_roi - intersection_area) if mask_area + area_of_roi - intersection_area > 0 else 0

                    if track_id in self.track_history:
                        velocity = np.linalg.norm(center - self.track_history[track_id][-1]) if self.track_history[track_id] else 0
                    else:
                        velocity = 0
                    self.track_history[track_id].append(center)

                    if self.names[cls_num] == 'person' and intersection_area > 0 and track_id not in self.persons_on_bikes:
                        persons.append((center, velocity, iou * 100))  # IoU percentage
                    elif self.names[cls_num] in ['car', 'motorcycle', 'bus', 'truck'] and intersection_area > 0:
                        vehicles.append((center, velocity, iou * 100))  # IoU percentage

                    annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), track_label=f'{track_id},{self.names[cls_num]}={confidence}')

                # Continue with prediction and frame saving logic here, now with the updated 'vehicles' list
                if persons and vehicles:
                    max_person = max(persons, key=lambda item: item[2])
                    max_vehicle = max(vehicles, key=lambda item: item[2])
                    # Round values to 5 decimal points before writing to CSV
                    data = [
                        round(max_person[0][0], 5),
                        round(max_person[0][1], 5),
                        round(max_person[1], 5),
                        round(max_person[2], 5),  # IoU percentage for person
                        round(max_vehicle[0][0], 5),
                        round(max_vehicle[0][1], 5),
                        round(max_vehicle[1], 5),
                        round(max_vehicle[2], 5)  # IoU percentage for vehicle
                    ]

                    prediction = self.loaded_model.predict([data])

                    # Check the prediction and save the frame if condition is met (e.g., prediction is 1)
                    if prediction[0] == 1:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        frame_path = os.path.join(save_directory, f"frame_{frame_count}_{timestamp}.jpg")
                        cv2.imwrite(frame_path, im0)

                frame_count += 1

            cap.release()
            cv2.destroyAllWindows()

            # Remove the temporary video file
            os.unlink(temp_video_path)

            # Create a response tensor
            output_tensor = pb_utils.Tensor("OUTPUT0", np.array([1], dtype=self.output_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print("Cleaning up...")
