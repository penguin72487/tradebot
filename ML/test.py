import torch

# 檢查 CUDA 是否可用
print("CUDA Available:", torch.cuda.is_available())

# 如果 CUDA 可用，則打印出當前使用的 GPU 裝置
if torch.cuda.is_available():
    print("Current CUDA Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available, running on CPU")
