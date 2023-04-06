import onnxruntime
import torch

# Path to the ONNX model file
model_path = "centroid-reid_resnet50_256_128_epoch_120.onnx"

# Create an inference session for the ONNX model
session = onnxruntime.InferenceSession(model_path)

input_sample=torch.randn((1,3,256,128))

# Run the model
output_data = session.run(['output'], input_sample)
# Get the names of the output nodes
print(output_data)
