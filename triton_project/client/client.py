import numpy as np
import tritonclient.grpc as grpcclient
import cv2

model_name = "yolo_model"
model_version = "1"
input_name = "input_0"
output_name = "output_0"

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (640, 640))
    image_normalized = image_resized / 255.0
    image_transposed = image_normalized.transpose(2, 0, 1)
    return image_transposed.astype(np.float32)

def infer_image(client, image):
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput(input_name, image.shape, "FP32"))
    inputs[0].set_data_from_numpy(image)
    outputs.append(grpcclient.InferRequestedOutput(output_name))

    results = client.infer(model_name=model_name,
                           model_version=model_version,
                           inputs=inputs,
                           outputs=outputs)
    return results.as_numpy(output_name)

def main(image_path):
    client = grpcclient.InferenceServerClient(url="localhost:8001")
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    result = infer_image(client, image)
    print(f"Inference result shape: {result.shape}")
    # Further processing of the result can be done here

image_path = "path_to_your_image.jpg"
main(image_path)
