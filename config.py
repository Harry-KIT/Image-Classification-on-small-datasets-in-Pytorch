import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224

USE_GPU = torch.cuda.is_available()

MODEL_SAVE_FILE = 'results/wild-animal-dataset.pth'

