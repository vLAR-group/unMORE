import numpy as np
import cv2
import torchvision
from matplotlib.colors import hsv_to_rgb
from torchvision import transforms
import matplotlib.pyplot as plt 
import torch.nn.functional as F



def get_mask_plot_colors(nr_colors):
  """Get nr_colors uniformly spaced hues to plot mask values."""
  hsv_colors = np.ones((nr_colors, 3), dtype=np.float32)
  hsv_colors[:, 0] = np.linspace(0, 1, nr_colors, endpoint=False)
  color_conv = hsv_to_rgb(hsv_colors)
  return color_conv



def vis_GT_gray(data):
    data = np.array(data)
    labels = np.unique(data)
    labels.sort()
    cmap = get_mask_plot_colors(np.count_nonzero(labels))
    color_image = np.zeros([data.shape[0],data.shape[1],3])
    color_idx = 0
    for idx, label in enumerate(labels):
        if label == 0:
            continue
        obj_mask = data==label
        color_image[:,:,0][obj_mask] = cmap[color_idx][0]
        color_image[:,:,1][obj_mask] = cmap[color_idx][1]
        color_image[:,:,2][obj_mask] = cmap[color_idx][2]
        color_idx += 1
    return color_image * 255

