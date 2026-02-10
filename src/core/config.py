import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "liver_unet.pth")

ROI_SIZE = (64, 64, 64)
SPACING = (1.5, 1.5, 2.0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
