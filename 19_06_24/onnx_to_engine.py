import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import onnx

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Define a function to build TensorRT engine
def build_trt_engine(onnx_file_path, engine_file_path):
    # Initialize TensorRT engine and builder
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # Parse the ONNX model
        with open(onnx_file_path, 'rb') as model:
            parser.parse(model.read())

        # Build and serialize the TensorRT engine
        serialized_engine = builder.build_serialized_network(network, config)
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)

        print(f"TensorRT engine saved to {engine_file_path}")

# Specify paths for ONNX and TensorRT engine files
onnx_model_path = "violation.onnx"
tensorrt_engine_path = "violation.engine"

# Build TensorRT engine from ONNX model
build_trt_engine(onnx_model_path, tensorrt_engine_path)
