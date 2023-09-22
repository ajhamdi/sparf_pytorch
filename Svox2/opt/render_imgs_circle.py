# Copyright 2021 Alex Yu
# Render 360 circle path

import torch
import svox2
import json
import svox2.utils
import math
import argparse
import numpy as np
import os
from os import path
from util.dataset import datasets
from util.util import Timing, compute_ssim, viridis_cmap, pose_spherical
from util import config_util

import imageio
import cv2
from tqdm import tqdm
from reflect import SparseRadianceFields

parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)

config_util.define_common_args(parser)

parser.add_argument('--n_eval', '-n', type=int, default=100000, help='images to evaluate (equal interval), at most evals every image')
# parser.add_argument('--rf_variant', type=int, default=0,
#                     help='the variant of the rf used in rendering')
parser.add_argument('--traj_type',
                    choices=['spiral', 'circle',"zoom","vertical"],
                    default='spiral',
                    help="Render a spiral (doubles length, using 2 elevations), or just a cirle")
parser.add_argument('--fps',
                    type=int,
                    default=30,
                    help="FPS of video")
parser.add_argument(
                "--width", "-W", type=float, default=None, help="Rendering image width (only if not --traj)"
                        )
parser.add_argument(
                    "--height", "-H", type=float, default=None, help="Rendering image height (only if not --traj)"
                            )
parser.add_argument(
	"--num_views", "-N", type=int, default=300,
    help="Number of frames to render"
)

# Path adjustment
parser.add_argument(
    "--offset", type=str, default="0,0,0", help="Center point to rotate around (only if not --traj)"
)
parser.add_argument("--radius", type=float, default=0.85, help="Radius of orbit (only if not --traj)")
parser.add_argument("--closeup_factor", type=float, default=0.5,
                    help="smaller Radius of orbit (only if --traj_type == `zoom`)")
parser.add_argument("--density_threshold", type=float, default=-10000.0,
                    help="smaller Radius of orbit (only if --traj_type == `zoom`)")
parser.add_argument(
    "--elevation",
    type=float,
    default=-45.0,
    help="Elevation of orbit in deg, negative is above",
)
parser.add_argument(
    "--elevation2",
    type=float,
    default=20.0,
    help="Max elevation, only for spiral",
)
parser.add_argument(
    "--vec_up",
    type=str,
    default=None,
    help="up axis for camera views (only if not --traj);"
    "3 floats separated by ','; if not given automatically determined",
)
parser.add_argument(
    "--vert_shift",
    type=float,
    default=0.0,
    help="vertical shift by up axis"
)

# Camera adjustment
parser.add_argument('--crop',
                    type=float,
                    default=1.0,
                    help="Crop (0, 1], 1.0 = full image")

# Foreground/background only
parser.add_argument('--nofg',
                    action='store_true',
                    default=False,
                    help="Do not render foreground (if using BG model)")
parser.add_argument('--nobg',
                    action='store_true',
                    default=False,
                    help="Do not render background (if using BG model)")

# Random debugging features
parser.add_argument('--blackbg',
                    action='store_true',
                    default=False,
                    help="Force a black BG (behind BG model) color; useful for debugging 'clouds'")

args = parser.parse_args()
config_util.maybe_merge_config_file(args, allow_invalid=True)
device = 'cuda:0'


dset = datasets[args.dataset_type](args.data_dir, split="test",
                                    **config_util.build_data_options(args))

if args.vec_up is None:
    up_rot = dset.c2w[:, :3, :3].cpu().numpy()
    ups = np.matmul(up_rot, np.array([0, -1.0, 0])[None, :, None])[..., 0]
    args.vec_up = np.mean(ups, axis=0)
    args.vec_up /= np.linalg.norm(args.vec_up)
    print('  Auto vec_up', args.vec_up)
else:
    args.vec_up = np.array(list(map(float, args.vec_up.split(","))))


args.offset = np.array(list(map(float, args.offset.split(","))))
if args.traj_type == 'spiral':
    angles = np.linspace(-180, 180, args.num_views + 1)[:-1]
    elevations = np.linspace(args.elevation, args.elevation2, args.num_views)
    c2ws = [
        pose_spherical(
            angle,
            ele,
            args.radius,
            args.offset,
            vec_up=args.vec_up,
        )
        for ele, angle in zip(elevations, angles)
    ]
    c2ws += [
        pose_spherical(
            angle,
            ele,
            args.radius,
            args.offset,
            vec_up=args.vec_up,
        )
        for ele, angle in zip(reversed(elevations), angles)
    ]
elif args.traj_type == 'zoom':
    angles = np.linspace(-180, 180, args.num_views + 1)[:-1]
    elevations = np.linspace(args.elevation, args.elevation2, args.num_views)
    distances = np.linspace(args.radius, args.radius * args.closeup_factor, args.num_views)
    c2ws = [
        pose_spherical(
            angle,
            ele,
            dist,
            args.offset,
            vec_up=args.vec_up,
        )
        for ele, angle, dist in zip(elevations, angles, distances)
    ]
    c2ws += [
        pose_spherical(
            angle,
            ele,
            dist,
            args.offset,
            vec_up=args.vec_up,
        )
        for ele, angle, dist in zip(reversed(elevations), angles, reversed(distances))
    ]

elif args.traj_type == 'circle':
    c2ws = [
        pose_spherical(
            angle,
            args.elevation,
            args.radius,
            args.offset,
            vec_up=args.vec_up,
        )
        for angle in np.linspace(-180, 180, args.num_views + 1)[:-1]
    ]
elif args.traj_type == 'vertical':
    c2ws = [
        pose_spherical(
            0,
            angle,
            args.radius,
            args.offset,
            vec_up=args.vec_up,
        )
        for angle in np.linspace(-90, 90, args.num_views + 1)[:-1]
    ]
c2ws = np.stack(c2ws, axis=0)
if args.vert_shift != 0.0:
    c2ws[:, :3, 3] += np.array(args.vec_up) * args.vert_shift
c2ws = torch.from_numpy(c2ws).to(device=device)

config_file_name = os.path.join(args.ckpt, "meta.json")
with open(config_file_name, "r") as config_file:
    old_configs = json.load(config_file)
if not path.isfile(args.ckpt):
    args.ckpt = path.join(args.ckpt, "data_{}.npz".format(old_configs["rf_variant"]))

render_out_path = path.join(path.dirname(args.ckpt), "{}_renders".format(args.traj_type))

# Handle various image transforms
if args.crop != 1.0:
    render_out_path += f'_crop{args.crop}'
if args.vert_shift != 0.0:
    render_out_path += f'_vshift{args.vert_shift}'

# grid = svox2.SparseGrid.load(args.ckpt, device=device)
# print(grid.center, grid.radius)
partial_alias = path.split(path.dirname(args.ckpt))[1]
vox_resolution = json.loads(old_configs["reso"])[-1][0]
srf = SparseRadianceFields(vox_res=vox_resolution, sh_dim=old_configs["sh_dim"], partial_alias=partial_alias, normalize="none", dataset_type=args.dataset_type)
coords, feats = srf.load_coords_and_feats(path.join(path.dirname(args.ckpt), "data_{}.npz".format(old_configs["rf_variant"])), device=device)
if args.dataset_type != "co3d":
    grid = srf.construct_grid(args.data_dir, coords, feats)
else:
    grid = srf.construct_grid(path.split(path.split(path.split(path.split(config_file_name)[0])[0])[0])[0], coords, feats)


##############################################################
# coords, feats = coords_and_feats_from_grid(grid)
# density_threshold = args.density_threshold  # -10_000.0 # 0.0
# # print(coords.shape[0])
# s_alias = "p" if density_threshold >=0 else "n"
# rf_alias = s_alias + str(int(abs(density_threshold)))
# render_out_path += rf_alias
# coords, feats = prune_sparse_voxels(coords, feats, density_threshold)
# save_coords_and_feats(path.join(path.dirname(args.ckpt),
#                                 "data_{}.npz".format(rf_alias)), coords, feats)
# # extract_mesh_from_sparse_voxels(coords, feats[:, 0], path.join(path.dirname(args.ckpt), "rf_{}_mesh.obj".format(rf_alias)), vox_res=512,
# #                                 smooth=False, level_set=density_threshold, clean=False)
# coords, feats = load_coords_and_feats(path.join(path.dirname(
#     args.ckpt), "data_{}.npz".format(rf_alias)), device=device)
# # print(coords.shape[0])
# grid = construct_grid(os.path.split(path.dirname(args.ckpt))[0], coords, feats,
#                       resolution=511, denormalize=False, device=device,sh_dim=1)


#######################################################
# DEBUG
#  grid.background_data.data[:, 32:, -1] = 0.0
#  render_out_path += '_front'

if grid.use_background:
    if args.nobg:
        grid.background_data.data[..., -1] = 0.0
        render_out_path += '_nobg'
    if args.nofg:
        grid.density_data.data[:] = 0.0
        #  grid.sh_data.data[..., 0] = 1.0 / svox2.utils.SH_C0
        #  grid.sh_data.data[..., 9] = 1.0 / svox2.utils.SH_C0
        #  grid.sh_data.data[..., 18] = 1.0 / svox2.utils.SH_C0
        render_out_path += '_nofg'

    #  # DEBUG
    #  grid.background_data.data[..., -1] = 100.0
    #  a1 = torch.linspace(0, 1, grid.background_data.size(0) // 2, dtype=torch.float32, device=device)[:, None]
    #  a2 = torch.linspace(1, 0, (grid.background_data.size(0) - 1) // 2 + 1, dtype=torch.float32, device=device)[:, None]
    #  a = torch.cat([a1, a2], dim=0)
    #  c = torch.stack([a, 1-a, torch.zeros_like(a)], dim=-1)
    #  grid.background_data.data[..., :-1] = c
    #  render_out_path += "_gradient"

config_util.setup_render_opts(grid.opt, args)

if args.blackbg:
    print('Forcing black bg')
    render_out_path += '_blackbg'
    grid.opt.background_brightness = 0.0

render_out_path += '.mp4'
print('Writing to', render_out_path)

# NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
# other backends will manually generate rays per frame (slow)
with torch.no_grad():
    n_images = c2ws.size(0)
    img_eval_interval = max(n_images // args.n_eval, 1)
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    n_images_gen = 0
    frames = []
    #  if args.near_clip >= 0.0:
    grid.opt.near_clip = 0.0 #args.near_clip
    if args.width is None:
        args.width = dset.get_image_size(0)[1]
    if args.height is None:
        args.height = dset.get_image_size(0)[0]

    for img_id in tqdm(range(0, n_images, img_eval_interval)):
        dset_h, dset_w = args.height, args.width
        im_size = dset_h * dset_w
        w = dset_w if args.crop == 1.0 else int(dset_w * args.crop)
        h = dset_h if args.crop == 1.0 else int(dset_h * args.crop)

        cam = svox2.Camera(c2ws[img_id],
                           dset.intrins.get('fx', 0),
                           dset.intrins.get('fy', 0),
                           w * 0.5,
                           h * 0.5,
                           w, h,
                           ndc_coeffs=(-1.0, -1.0))
        torch.cuda.synchronize()
        im = grid.volume_render_image(cam, use_kernel=True)
        torch.cuda.synchronize()
        im.clamp_(0.0, 1.0)

        im = im.cpu().numpy()
        im = (im * 255).astype(np.uint8)
        frames.append(im)
        im = None
        n_images_gen += 1
    if len(frames):
        vid_path = render_out_path
        imageio.mimwrite(vid_path, frames, fps=args.fps, macro_block_size=8)  # pip install imageio-ffmpeg


