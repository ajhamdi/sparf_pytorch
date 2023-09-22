import sys
import os
from matplotlib.pyplot import axis

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "pixel_nerf", "src")))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
# import torch.nn.functional as F
import numpy as np
import imageio
import pixel_nerf.src.util as util
from pathlib import Path

from extra_utils import check_folder
import warnings
from util.util import compute_ssim 
# from data import get_split_dataset
from render import NeRFRenderer
from model import make_model
# from scipy.interpolate import CubicSpline
import tqdm
import math
from pyhocon import ConfigFactory


def vidoe_distance(frames1, frames2, no_lpips=False,number_of_chunks=1):
    """
    a util function to measure the diastance between two video frames ( as torch tensors ``N*H*W*C` ) in psnr, sssim and lpips (optional)
    """
    avg_psnr, avg_ssim, avg_lpips = 0.0, 0.0, 0.0
    # assert frames1.shape == frames2.shape ," the two videos are not equal in size"
    if frames1.shape != frames2.shape:
        return {"PSNR": avg_psnr, "SSIM": avg_ssim, "LPIPS": avg_lpips}

    if not no_lpips:
        import lpips
        lpips_vgg = lpips.LPIPS(
            net="vgg", verbose=False).eval().to(frames1.device)
    mse = (frames1 - frames2) ** 2
    mse_num: float = mse.mean().item()
    try:
        psnr = -10.0 * math.log10(mse_num)
    except:
        psnr = 0.0

    avg_psnr += psnr
    ssim = compute_ssim(frames1, frames2).mean().item()
    avg_ssim += ssim
    if not no_lpips:
        chunk_frames = int(frames1.shape[0]/number_of_chunks)
        for ii in range(number_of_chunks):
            lpips_i = lpips_vgg(frames1[ii*chunk_frames:ii*chunk_frames+chunk_frames].permute([0, 3, 1, 2]).contiguous(),
                                frames2[ii*chunk_frames:ii*chunk_frames+chunk_frames].permute([0, 3, 1, 2]).contiguous(), normalize=True).mean().item()
            avg_lpips += lpips_i
        passed = chunk_frames * (number_of_chunks-1)
        if passed != frames1.shape[0]:
            lpips_i = lpips_vgg(frames1[passed::].permute([0, 3, 1, 2]).contiguous(),
                                frames2[passed::].permute([0, 3, 1, 2]).contiguous(), normalize=True).mean().item()
            avg_lpips += lpips_i

    return {"PSNR": avg_psnr, "SSIM": avg_ssim, "LPIPS": avg_lpips}


def evaluate_pixel_images(d_dir, eval_frames, traj_type="zoom", srf=None, device=None):
    if traj_type != "hard" and  traj_type != "test":
        gt_vid = os.path.join(d_dir, "SRF", "vox512", "full", "{}_renders.mp4".format(traj_type))
        gt_frames = imageio.mimread(gt_vid)
        gt_frames = torch.from_numpy(np.concatenate([x[None,...] for x in gt_frames]))/255.0
    else :
        _, gts, masks = srf.load_c2ws_images(data_dir=d_dir, device=device, split=traj_type, c_rf_variant=0, randomized_views=False, all_views=True)
        gt_frames = torch.Tensor(gts).permute(0, 3, 1, 2)[None, ...]

    eval_frames = torch.from_numpy(np.concatenate([x[None, ...] for x in eval_frames]))/255.0

    metrics = vidoe_distance(gt_frames, eval_frames)
    return metrics

def find_split_and_indx(shape_id,lists_dir):
    shape_id = os.path.split(shape_id)[1]
    for lbl in ["train","val","test"]:
        file1 = open(os.path.join(lists_dir,"snr_{}.txt".format(lbl)), 'r')
        Lines = file1.read().splitlines()
        if shape_id in Lines:
            return lbl, Lines.index(shape_id)
    return None , None


def visualize_pixel_nerf2(data_dict, batch_indx, render_dir, num_views=200, vizualization_id=0, gif=False, traj_type="zoom", setup=None, device="cuda:0", srf=None):
    shape_id = data_dict["labels"][batch_indx]
    views = '2' if setup["nb_views"] == 1 else '2 6 10'
    fps = 30 if num_views == 200 else int(30*num_views/200.0)
    found_split, found_indx = find_split_and_indx(shape_id, setup["data_dir"])
    c2ws = "None" if traj_type not in ["hard", "test"] else srf.load_c2ws_images(data_dir=shape_id, device=device, split=traj_type, c_rf_variant=0, randomized_views=False, all_views=True)[0].tolist()
    # print("$$$$$$$$$$$$$$$$$$$",len(c2ws))
    command = "python pixel_nerf/eval/gen_video.py -n sn64 --gpu_id=0 --split {} -P '{}' -D data/nerf_datasets/NMR_Dataset -S {} --conf pixel_nerf/conf/exp/sn64.conf --checkpoints_path pixel_nerf/checkpoints --visual_path {} --radius 0.0 --num_views {} --traj_type {} --vizualization_id {} --new_res {} --fps {} --c2ws '{}'".format(
        found_split, views, found_indx, render_dir, num_views, traj_type, vizualization_id, setup["img_res"], fps,str(c2ws))
    # print(command)
    os.system(command)
    vid_file = os.path.join(render_dir, str(vizualization_id)+".mp4")
    frames = imageio.mimread(vid_file)
    # print("$$$$$$$$$$$", len(frames), frames[0].shape, frames[0].max())
    return frames


def visualize_vision_nerf(data_dict, batch_indx, render_dir, num_views=200, vizualization_id=0, gif=False, traj_type="zoom", setup=None,device="cuda:0",srf=None):
    shape_id = data_dict["labels"][batch_indx]
    print("$$$$$$$$$$$$$$", vizualization_id," : ", shape_id)
    views = '2'# if setup["nb_views"] == 1 else '2 6 10'
    fps = 30 if num_views == 200 else int(30*num_views/200.0)
    found_split, found_indx = find_split_and_indx(shape_id, setup["data_dir"])
    c2ws = "None" if traj_type not in ["hard", "test"] else np.transpose(srf.load_c2ws_images(data_dir=shape_id, device=device, split=traj_type, c_rf_variant=0, randomized_views=False, all_views=True)[0],(0,1,2)).tolist()

    command = "python vision_nerf/eval_nmr.py --config vision_nerf/configs/render_nmr.txt --use_data_index --data_indices {} --mode {} --new_res {} --fps {} --num_views {} --traj_type {} --outdir {} --vizualization_id {} --pose_index {} --c2ws '{}'".format(
        found_indx, found_split, setup["img_res"], fps, num_views, traj_type, render_dir, vizualization_id, views, str(c2ws))
    os.system(command)
    vid_file = os.path.join(render_dir, str(vizualization_id),"{}_renders_{}.mp4".format(traj_type,str(vizualization_id)))
    frames = imageio.mimread(vid_file)
    # print("$$$$$$$$$$$", len(frames), frames[0].shape, frames[0].max())
    return frames

def visualize_pixel_nerf(data_dict,batch_indx, net, render_dir, num_views=200, vizualization_id=0, gif=False, traj_type="zoom", setup=None, conf=None,device=None):
    elevation = -10
    elevation2 = 20
    radius = 0.0  # 0.85
    focal = torch.tensor(482.842712474619, dtype=torch.float32)[None]
    lindisp = False
    z_near = 1.2
    split = "test"
    z_far = 4.0
    ray_batch_size = 50000
    scale = 1.0
    fps = 30
    num_views = setup["nb_frames"]
    c = None # torch.tensor((setup["img_res"]/2, setup["img_res"]/2),dtype=torch.float32).to(device=device)
    source = torch.tensor(list(range(setup["nb_views"])), dtype=torch.long)
    data_path = data_dict["labels"][batch_indx]
    print("Data instance loaded:", data_path)

    images = data_dict["imgs"][batch_indx]  # (NV, 3, H, W)

    poses = data_dict["c2ws"][batch_indx]


    # c = data.get("c")
    # if c is not None:
    #     c = c.to(device=device).unsqueeze(0)

    NV, _, H, W = images.shape

    if scale != 1.0:
        Ht = int(H * scale)
        Wt = int(W * scale)
        if abs(Ht / scale - H) > 1e-10 or abs(Wt / scale - W) > 1e-10:
            warnings.warn(
                "Inexact scaling, please check {} times ({}, {}) is integral".format(
                    scale, H, W
                )
            )
        H, W = Ht, Wt


    renderer = NeRFRenderer.from_conf(
        conf["renderer"], lindisp=lindisp, eval_batch_size=ray_batch_size,
    ).to(device=device)

    render_par = renderer.bind_parallel(net, "0", simple_output=True).eval()

    # Get the distance from camera to origin
    # z_near = dset.z_near
    # z_far = dset.z_far

    print("Generating rays")

    # dtu_format = hasattr(dset, "sub_format") and dset.sub_format == "dtu"

    print("Using default (360 loop) camera trajectory")
    if radius == 0.0:
        radius = (z_near + z_far) * 0.5
        print("> Using default camera radius", radius)
    else:
        radius = radius

    # Use 360 pose sequence from NeRF
    render_poses = torch.stack(
        [
            util.pose_spherical(angle, elevation, radius)
            for angle in np.linspace(-180, 180, num_views + 1)[:-1]
        ],
        0,
    )  # (NV, 4, 4)

    render_rays = util.gen_rays(
        render_poses,
        W,
        H,
        focal * scale,
        z_near,
        z_far,
        c=c * scale if c is not None else None,
    ).to(device=device)
    # (NV, H, W, 8)

    focal = focal.to(device=device)

    # source = torch.tensor(list(map(int, args.source.split())), dtype=torch.long)
    
    NS = len(source)
    print("$$$$$$$$$$$$$", focal)

    random_source = NS == 1 and source[0] == -1
    assert not (source >= NV).any()

    if renderer.n_coarse < 64:
        # Ensure decent sampling resolution
        renderer.n_coarse = 64
        renderer.n_fine = 128

    with torch.no_grad():
        print("Encoding source view(s)")
        if random_source:
            src_view = torch.randint(0, NV, (1,))
        else:
            src_view = source

        net.encode(
            images[src_view].unsqueeze(0),
            poses[src_view].unsqueeze(0).to(device=device),
            focal,
            c=c,
        )

        print("Rendering", num_views * H * W, "rays")
        all_rgb_fine = []
        for rays in tqdm.tqdm(torch.split(render_rays.view(-1, 8), ray_batch_size, dim=0) ):
            rgb, _depth = render_par(rays[None])
            all_rgb_fine.append(rgb[0])
        _depth = None
        rgb_fine = torch.cat(all_rgb_fine)
        # rgb_fine (V*H*W, 3)

        frames = rgb_fine.view(-1, H, W, 3)

    print("Writing video")
    vid_name = "{}".format(vizualization_id)
    vid_path = os.path.join(render_dir, vid_name + ".mp4")
    # viewimg_path = os.path.join(render_dir, args.name, "video" + vid_name + "_view.jpg")
    imageio.mimwrite(vid_path, (frames.cpu().numpy() * 255).astype(np.uint8), fps=fps, quality=8
    )


    return frames


def evaluate_pixel_nerf(val_loader, device, srf, setup):
    setup["pixel_dir"] = os.path.join(setup["root_dir"], "pixel_nerf")

    # model_path = os.path.join(setup["pixel_dir"], "checkpoints","sn64","pixel_nerf_latest" )
    # conf = ConfigFactory.parse_file(os.path.join(setup["pixel_dir"],"conf","exp","sn64.conf"))
    # net = make_model(conf["model"]).to(device=device)
    # net.my_load_weights(model_path)
    val_ssim, val_psnr, val_lpips = [], [], []
    losses = []

    # args, conf = util.args.parse_args(extra_args)
    # args.resume = True


    for i, data_dict in enumerate(val_loader):
        short_list_cond = i * setup["batch_size"] in range(0, 0+setup["visualizations_nb"])
        if not short_list_cond :
            continue 

        for ii, d_dir in enumerate(data_dict["labels"]):
            render_dir = os.path.join(setup["baseline_dir"],setup["run"], "view"+str(setup["nb_views"]), "vids")
            Path(render_dir).mkdir(parents=True, exist_ok=True)
            # frames =  visualize_pixel_nerf(data_dict, batch_indx=ii, net=net, render_dir=render_dir, num_views=setup["nb_frames"], vizualization_id=setup["batch_size"]*i + ii, gif=setup["gif"], traj_type=setup["traj_type"], setup=setup, conf=conf, device=device)
            if setup["run"] == "pixel":
                frames = visualize_pixel_nerf2(data_dict, batch_indx=ii, render_dir=render_dir, num_views=setup["nb_frames"], vizualization_id=setup["batch_size"]*i + ii, gif=setup["gif"], traj_type=setup["traj_type"], setup=setup, device=device, srf=srf)
            elif setup["run"] == "vision":
                frames = visualize_vision_nerf(data_dict, batch_indx=ii, render_dir=render_dir,num_views=setup["nb_frames"], vizualization_id=setup["batch_size"]*i + ii, gif=setup["gif"], traj_type=setup["traj_type"], setup=setup, device=device, srf=srf )

            pred_metrics = evaluate_pixel_images(d_dir, frames, traj_type=setup["traj_type"], srf=srf, device=device)


            # if setup["gif"]:
            #     wandb.log({"renderings/{}".format(setup["batch_size"]*i + ii): wandb.Video(np.transpose(np.concatenate([fr[None, ...] for fr in frames], axis=0), (
            #         0, 3, 1, 2)), fps=2, format="gif"), "epoch": str(epoch)}, commit=False)
            # if setup["concat_gt_output"] and not setup["gif"] and setup["visualize_gt"]:
            #     concat_render_dir = os.path.join(
            #         setup["output_dir"], "comparisons", str(epoch))
            #     os.makedirs(concat_render_dir, exist_ok=True)
            #     out_vid = os.path.join(render_dir, "{}_renders_{}.mp4".format(
            #         setup["traj_type"], str(setup["batch_size"]*i + ii)))
            #     gt_vid = os.path.join(gt_render_dir, "{}_renders_{}.mp4".format(
            #         setup["traj_type"], str(setup["batch_size"]*i + ii)))
            #     concat_vid = os.path.join(concat_render_dir, "{}_renders_{}.mp4".format(
            #         setup["traj_type"], str(setup["batch_size"]*i + ii)))
            #     concat_horizontal_videos(source_videos_list=[out_vid, gt_vid], output_file=concat_vid)

        val_ssim.append(pred_metrics["SSIM"])
        val_psnr.append(pred_metrics["PSNR"])
        val_lpips.append(pred_metrics["LPIPS"])
        torch.cuda.empty_cache()

    return {"loss": np.mean(losses), "ssim": np.mean(val_ssim), "psnr": np.mean(val_psnr), "lpips": np.mean(val_lpips)}
