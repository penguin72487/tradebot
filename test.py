import torch, torchvision, time
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    x = torch.randn(8192, 8192, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    y = x @ x
    torch.cuda.synchronize()
    print("mm(8192) time(s):", time.time() - t0)
