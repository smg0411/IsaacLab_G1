import cv2
import numpy as np
import os.path as osp
import os
import glob
from natsort import natsorted

folder_path = r"C:\IsaacLab\scripts\reinforcement_learning\rsl_rl\output\camera"
imgs_path = natsorted(glob.glob(osp.join(folder_path, "*.png")))

frame = cv2.imread(imgs_path[0])
h, w, layers = frame.shape
fps = 15

out = cv2.VideoWriter(osp.join(osp.dirname(folder_path), "output_video_G1_Lift_v10_15.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

for img_path in imgs_path:
    img = cv2.imread(img_path)
    out.write(img)

out.release()