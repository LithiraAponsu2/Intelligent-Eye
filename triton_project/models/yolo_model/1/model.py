import triton_python_backend_utils as pb_utils
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TritonPythonModel:
    def initialize(self, args):
        self.model_path = args['model_repository'] + '/1/yolov8x-seg.engine'
        
        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()

        # Allocate device memory
        self.inputs = []
        self.outputs = []
        self.allocations = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer to device bindings.
            self.allocations.append(device_mem)

            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.inputs.append(host_mem)
            else:
                self.outputs.append(host_mem)
        
        self.bindings = [int(mem) for mem in self.allocations]

    def execute(self, requests):
        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input_0")
            input_data = input_tensor.as_numpy().ravel()

            # Transfer input data to the GPU.
            cuda.memcpy_htod(self.inputs[0], input_data)

            # Execute the model.
            self.context.execute_v2(self.bindings)

            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh(self.outputs[0], self.allocations[1])

            output_data = np.array(self.outputs[0]).reshape(self.context.get_binding_shape(1))
            output_tensor = pb_utils.Tensor("output_0", output_data)

            responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses

    def finalize(self):
        print('Cleaning up...')
