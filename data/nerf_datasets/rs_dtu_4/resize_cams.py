import os
import argparse
import numpy as np
from tqdm import tqdm
import cv2

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    default="DTU",
    help="Data directory",
)
args = parser.parse_args()

cam_paths = [os.path.join(args.data_dir, x, 'cameras.npz') for x in os.listdir(args.data_dir)]

scale_fact = 4

for cam_path in tqdm(cam_paths):
    if not os.path.exists(cam_path):
        continue
    z = dict(np.load(cam_path))
    for k in z.keys():
        if k.startswith("camera_mat_inv_"):
            pass
        elif k.startswith("world_mat_inv_"):
            pass
        elif k.startswith("camera_mat_"):
            z[k][:3, :3] = z[k][:3, :3] * scale_fact
        elif k.startswith("world_mat_"):
            #  K, R, t = cv2.decomposeProjectionMatrix(z[k][:3])[:3]
            #  print('FROM')
            #  print(K)
            #  print(R)
            #  print(t)
            z[k][:2] = z[k][:2] / scale_fact
            #  K, R, t = cv2.decomposeProjectionMatrix(z[k][:3])[:3]
            #  print('TO')
            #  print(K)
            #  print(R)
            #  print(t)

    for k in z.keys():
        if k.startswith("camera_mat_inv_"):
            noninv = "camera_mat_" + k[k.rindex('_') + 1:]
            z[k] = np.linalg.inv(z[noninv])
        elif k.startswith("world_mat_inv_"):
            noninv = "world_mat_" + k[k.rindex('_') + 1:]
            z[k] = np.linalg.inv(z[noninv])
    np.savez(cam_path, **z)
