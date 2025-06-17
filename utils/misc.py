import cv2
import numpy as np
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F

def batch_erode(binary_masks, kernel_size=9, num_round=3):
    '''
    binary_masks: [B, H, W]
    '''
    binary_masks = binary_masks.unsqueeze(1)
    kernel = torch.ones(1, 1, kernel_size, kernel_size)
    for _ in range(0, num_round):
        conved_mask = F.conv2d(binary_masks.double(), kernel.to(binary_masks.device).double(), padding=int((kernel_size-1)/2))[:, 0, :, :]
        binary_masks = torch.where(conved_mask>=kernel_size*kernel_size, 1, 0)
        binary_masks = binary_masks.unsqueeze(1)
    return binary_masks.squeeze(1)

import json
import numpy as np

class NpEncoder(json.JSONEncoder):
   """ Custom encoder for numpy data types """
   def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)
