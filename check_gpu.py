# If GPU is not available
# delete line model.to(f"cuda:{args.device}") from predict.py
# otherwise it would throw an error RuntimeError: No CUDA GPUs are available
import torch
import os

if not torch.cuda.is_available():
    ret_code = os.system("sed -i '40d' multivers/multivers/predict.py")
    print(f"exit code = {ret_code}")
