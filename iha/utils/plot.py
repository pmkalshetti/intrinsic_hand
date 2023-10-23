import cv2 as cv
import numpy as np

def draw_pts_on_img(img, list_pt2d, radius=5, color=(255, 0, 0), thickness=2):
    img_plot = img.copy()
    for pt2d in list_pt2d:
        img_plot = cv.circle(img_plot, pt2d.astype(int), radius, color, thickness)
    return img_plot

def alpha_composite(bg_rgba, fg_rgba, alpha):
    composite_img = bg_rgba.copy()
    composite_img[:, :, :3] = bg_rgba[:, :, :3] * (1 - fg_rgba[:, :, 3:4]*(1-alpha)) + fg_rgba[:, :, :3] * fg_rgba[:, :, 3:4]*alpha
    composite_img[:, :, :3] = np.clip(composite_img[:, :, :3], 0, 1)
    composite_img[:, :, 3] = np.maximum(composite_img[:, :, 3], fg_rgba[:, :, 3])
    return composite_img
