from gc import callbacks
from tkinter import EXCEPTION
from typing import List
from unittest import result
import shutil

# import clip
from tqdm import tqdm
# import kaolin.ops.mesh
# import kaolin as kal
import torch
# from neural_style_field import NeuralStyleField
# from utils import device
# from mesh import Mesh , LODMesh
# from render import Renderer

# from Normalization import MeshNormalizer
# from utils import add_vertices, sample_bary ,get_2dembedding_model , inception_score
from models import SRFNet ,ResidualUNet3D
import numpy as np
import random
import pandas as pd
import math
import copy
import torchvision
import os
from PIL import Image
import trimesh
from ops import train, validate, make_data_loader, record_metrics, fully_profile_network
from extra_utils import random_id , ListDict , save_results , reconstruct_mesh_from_points , save_trimesh_scene, check_folder , gif_folder , unit_spherical_grid , merge_two_dicts ,run_command_with_time_measure , unit_spherical_random
import argparse
from pathlib import Path
from torchvision import transforms
from datasets import ShapeNetCore , ShapeNetRend
from torch.utils.tensorboard import SummaryWriter 
from Svox2.opt.reflect import SparseRadianceFields
import wandb
import cv2 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(torch.cuda.current_device())
# device = torch.device("cuda:{}".format(torch.cuda.current_device()))
ShapeNet_cat_ids = { "watercraft": "04530566", "rifle": "04090263", "display": "03211117", "lamp": "03636649", "speaker": "03691459", "cabinet": "02933112", "chair": "03001627", "bench": "02828884", "car": "02958343", "airplane": "02691156", "sofa": "04256520", "table": "04379243", "phone": "04401088" }
USER = "userA" # pick your own wandb user name here
PROJECT = "SPARF" # pick your own wandb project name here

def prepare_setup(setup):
    if setup["exp_id"] == "random":
        setup["exp_id"] = random_id()
    setup["root_dir"] = os.getcwd()
    setup["results_dir"] = os.path.join(setup["root_dir"], "results")
    setup["gt_dir"] = os.path.join(setup["results_dir"], "GT", setup["object_class"])
    setup["baseline_dir"] = os.path.join(setup["results_dir"], "baselines", setup["object_class"])
    setup["output_dir"] = os.path.join(setup["results_dir"],setup["exp_set"],setup["exp_id"])
    setup["results_file"] = os.path.join(setup["output_dir"] , "results_{}.csv".format(setup["exp_id"]))
    Path(setup["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(setup["gt_dir"]).mkdir(parents=True, exist_ok=True)
    Path(setup["baseline_dir"]).mkdir(parents=True, exist_ok=True)

    setup["weights"] = os.path.join(setup["output_dir"],setup["weights"])
    setup["ae_weights"] = os.path.join("checkpoints","ae_voxnet_{}.pth".format(setup["vox_res"]))

    setup["log_dir"] = os.path.join(setup["output_dir"],"logs")
    check_folder(setup["log_dir"])

    setup["render_dir"] = os.path.join(setup["output_dir"],"renderings")
    setup["outmesh_dir"] = os.path.join(setup["output_dir"], "meshes")
    setup["mesh_render_dir"] = os.path.join(setup["output_dir"], "mesh_renderings")

    check_folder(setup["render_dir"])
    check_folder(setup["outmesh_dir"])
    check_folder(setup["mesh_render_dir"])
    setup["rf_alias"] = "STF" if setup["sh_dim"] == 1 else "SRF"
    setup["partial_alias"] = "full" if setup["nb_views"] == 400 else "view{}".format(setup["nb_views"])
    # check_folder(setup["data_dir"])
    for k, v in setup.items():
        if isinstance(v, bool):
            setup[k] = int(v)
    wandb.init(project=PROJECT, entity=USER,id=setup["exp_set"]+"_"+setup["exp_id"],config = setup ,sync_tensorboard=False)

    setup["res_alias"] = str(setup["vox_res"])

    setup["in_nchannel"] = 3*setup["input_sh_dim"]+1
    setup["cam_in_size"] = int(setup["encode_cameras"]) * (3 + setup["num_encoding_functions"]* 6) * setup["nb_views"]  # for position encoding
    setup["img_in_size"] = int(setup["encode_imgs"]) * setup["img_res"]
    setup["time_in_size"] = int(setup["encode_time"]) * (1 + setup["num_encoding_functions"] * 2)

    setup["added_cam_latent_channels"] = int(setup["encode_cameras"]) * setup["cam_in_size"] + int(setup["encode_cameras"]) * int(setup["learn_cam_embed"]) * (setup["cam_latent_channels"]-setup["cam_in_size"])    # for position encoding
    setup["added_img_latent_channels"] = setup["img_latent_channels"] * int(setup["encode_imgs"])
    setup["out_nchannel"] = 3*setup["sh_dim"]+1
    setup["added_time_latent_channels"] = setup["time_in_size"] * int(setup["encode_time"]) + int(
        setup["encode_time"]) * int(setup["learn_time_embed"]) * (setup["time_latent_channels"]-setup["time_in_size"])
        
    # Constrain all sources of randomness
    torch.manual_seed(setup["seed"])
    torch.cuda.manual_seed(setup["seed"])
    torch.cuda.manual_seed_all(setup["seed"])
    random.seed(setup["seed"])
    np.random.seed(setup["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    return setup

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm
def density_threshold_from_nbpoints(nb_points):
    """
    a util function to heuristcly pick the threshold for poission mesh reincstruction based on the nb_points used 
    """
    if nb_points > 50000:
        density_threshold = 0.01
    if nb_points > 5000:
         density_threshold = 500.0/nb_points
    else :
        density_threshold = 0.1
    
    return density_threshold
import json
def create_camera_dict(scene,aabb_scale=1):
    """
    creaet a camera JSON dictionary frindly to the NERF setup from a Trimesh.Scene ( see example at https://github.com/NVlabs/instant-ngp/blob/master/data/nerf/fox/transforms.json )
    """
    w = float(scene.camera.resolution[0])
    h = float(scene.camera.resolution[1])
    fl_x = float(scene.camera.focal[0])
    fl_y = float(scene.camera.focal[1])
    k1 = 0
    k2 = 0
    p1 = 0
    p2 = 0
    cx = w / 2
    cy = h / 2
    angle_x = math.atan(w / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2
    out = {
    "camera_angle_x": angle_x,
    "camera_angle_y": angle_y,
    "fl_x": fl_x,
    "fl_y": fl_y,
    "k1": k1,
    "k2": k2,
    "p1": p1,
    "p2": p2,
    "cx": cx,
    "cy": cy,
    "w": w,
    "h": h,
    "aabb_scale": aabb_scale,
    "frames": [],
    }
    return out 
def append_frame_to_camera_dict(scene,camera_dict,image_name,scale=1.0):
    """
    update a camera JSON dictionary frindly to the NERF setup from a Trimesh.Scene and the rendered image under the name image_name
    """
    b = sharpness(image_name)
    bname = os.path.split(image_name)
    name = os.path.join(os.path.split(bname[0])[-1],bname[-1])
    c2w = scene.camera_transform
    # frame={"file_path":name,"sharpness":b,"transform_matrix": c2w}
    frame={"file_path":name[0:-4],"sharpness":b,"transform_matrix": c2w}

    camera_dict["frames"].append(frame)
    camera_dict = fix_camera_dict(scene,camera_dict,scale=scale)
    return camera_dict
def fix_camera_dict(scene,camera_dict,scale=1.0):
    """
    fix a camera JSON dictionary frindly to the NERF setup from a Trimesh.Scene
    """
    c2w = camera_dict["frames"][-1]["transform_matrix"] # np.linalg.inv(m)
    # c2w[0:3,3] -= centroid
    c2w[0:3,3] *= scale
    # #print(name,c2w)
    # c2w[0:3,2] *= -1 # flip the y and z axis
    # c2w[0:3,1] *= -1
    # c2w = c2w[[0,2,1,3],:] # swap y and z 012 201 102
    # c2w[2,:] *= -1 # flip whole world upside down
    camera_dict["frames"][-1]["transform_matrix"] = c2w
    return camera_dict
def save_camera_dict(camera_dict,save_name):
    """
    fix a camera JSON dictionary frindly to the NERF setup from a Trimesh.Scene
    """
    for f in camera_dict["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    with open(save_name, "w") as outfile:
        json.dump(camera_dict, outfile, indent=2)

def run_branched(args):
    setup = vars(args)
    setup = prepare_setup(setup)

    if setup["cluster_run"]:
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1400, 900))
        display.start()
        trimesh.util.attach_to_log()
    else : 
        import pyglet
        pyglet.options["headless"] = True

        
    if setup["run"] == "base":
        run_base(setup,)
    elif setup["run"] == "render":
        run_render(setup,)
    elif setup["run"] == "extract":
        run_extract(setup,)
    elif setup["run"] == "visualize":
        run_visualize(setup,)
    elif setup["run"] in ["train","eval"]:
        run_train(setup,)
    elif setup["run"] == "preprocess":
        run_preprocess(setup)
    elif setup["run"] in ["pixel","vision"]:
        run_pixel_nerf(setup)
def run_pixel_nerf(setup): 
    # root_dir = os.path.join(setup["data_dir"],"ShapeNetRend")
    from nerf_ops import evaluate_pixel_nerf

    root_dir = setup["data_dir"]

    writer = SummaryWriter(log_dir=setup["log_dir"])

    print("working and data direcoties: ",root_dir,os.getcwd())
    srf = SparseRadianceFields(vox_res=setup["vox_res"], sh_dim=setup["sh_dim"], reflect_type=setup["reflect_type"], nb_views=setup["nb_views"], input_sh_dim=setup["input_sh_dim"], nb_rf_variants=setup["nb_rf_variants"], normalize=setup["normalize_input"], input_quantization_size=setup["input_quantization_size"],
                               encode_cameras=bool(setup["encode_cameras"]),
                               num_encoding_functions=setup["num_encoding_functions"], 
                               prune_input=setup["prune_input"], density_threshold=setup["density_threshold"],
                               encode_imgs=bool(setup["encode_imgs"]),
                                density_normilzation_factor=setup["density_normilzation_factor"], 
                                colors_normilzation_factor=setup["colors_normilzation_factor"],
                               encode_time=bool(setup["encode_time"]),
                               time_steps=setup["time_steps"],diffusion_type=setup["diffusion_type"],
                               randomized_views=setup["randomized_views"], online_nb_views=setup["online_nb_views"],
                               kernel_std=setup["kernel_std"], diffusion_kernel_size=setup["diffusion_kernel_size"], ignore_input=["ignore_input"],
                               dataset_type=setup["dataset_type"]
                               )
    # dataloader = make_data_loader("train", root_dir, setup["batch_size"], shuffle=True, num_workers=setup["num_workers"], repeat=False, object_class=setup["object_class"],augment_data=False, drop_last=True, dset_partition=setup["dset_partition"],srf=srf,use_lower_res=setup["use_lower_res"])
    val_loader = make_data_loader("test", root_dir, setup["batch_size"], shuffle=False, num_workers=setup["num_workers"], repeat=False,object_class=setup["object_class"], augment_data=False, drop_last=True, dset_partition=setup["dset_partition"], srf=srf, use_lower_res=setup["use_lower_res"])

    metrics = evaluate_pixel_nerf(val_loader, device,srf, setup)
    val_metrics = {"val_"+k:v for k,v in metrics.items()}
    record_metrics(val_metrics,writer,0)

    c_Results = {'best_loss':0,'best_ssim':val_metrics["val_ssim"],'best_psnr':val_metrics["val_psnr"],'best_lispip':val_metrics["val_lpips"],'best_acc':0,"best_diou":0,"c_epoch":0,}     
    print(c_Results)
    results = ListDict(merge_two_dicts(setup, c_Results))
    save_results(setup["results_file"],results)



def run_train(setup): 
    # root_dir = os.path.join(setup["data_dir"],"ShapeNetRend")
    root_dir = setup["data_dir"]
    srf = SparseRadianceFields(vox_res=setup["vox_res"], sh_dim=setup["sh_dim"], reflect_type=setup["reflect_type"], nb_views=setup["nb_views"], input_sh_dim=setup["input_sh_dim"], nb_rf_variants=setup["nb_rf_variants"], normalize=setup["normalize_input"], input_quantization_size=setup["input_quantization_size"],
                               encode_cameras=bool(setup["encode_cameras"]),
                               num_encoding_functions=setup["num_encoding_functions"], 
                               prune_input=setup["prune_input"], density_threshold=setup["density_threshold"],
                               encode_imgs=bool(setup["encode_imgs"]),
                                density_normilzation_factor=setup["density_normilzation_factor"], 
                                colors_normilzation_factor=setup["colors_normilzation_factor"],
                               encode_time=bool(setup["encode_time"]),
                               time_steps=setup["time_steps"],diffusion_type=setup["diffusion_type"],
                               randomized_views=setup["randomized_views"], online_nb_views=setup["online_nb_views"],
                               kernel_std=setup["kernel_std"], diffusion_kernel_size=setup["diffusion_kernel_size"], ignore_input=["ignore_input"],
                               dataset_type=setup["dataset_type"]
                               )

    dataloader = make_data_loader("train", root_dir, setup["batch_size"], shuffle=True, num_workers=setup["num_workers"], repeat=False, object_class=setup["object_class"],augment_data=False, drop_last=True, dset_partition=setup["dset_partition"],srf=srf,use_lower_res=setup["use_lower_res"])
    val_loader = make_data_loader("test", root_dir, setup["batch_size"], shuffle=False, num_workers=setup["num_workers"], repeat=False,object_class=setup["object_class"], augment_data=False, drop_last=True, dset_partition=setup["dset_partition"], srf=srf, use_lower_res=setup["use_lower_res"])

    # channels_dict = {"sh": 3*setup["sh_dim"] +1, "3d_texture": 4, "mlp": 9}

    writer = SummaryWriter(log_dir=setup["log_dir"])
    # if not setup["dense_pipeline"]:
    net = SRFNet(setup["time_in_size"], setup["cam_in_size"], setup["img_in_size"], setup["learn_cam_embed"], setup["learn_time_embed"], setup["network_depth"], setup["support_network_depth"], setup["use_adv_loss"], setup["add_input_late"], resolution=setup["vox_res"], in_nchannel=setup["in_nchannel"], out_nchannel=setup["out_nchannel"], batch_size=setup["batch_size"],
                    enable_pruning=setup["enable_pruning"], prune_last_layer=setup["prune_last_layer"], strides=setup[
                        "strides"], normalize=setup["normalize_input"], added_cam_latent_channels=setup["added_cam_latent_channels"],
                 added_img_latent_channels=setup["added_img_latent_channels"], added_time_latent_channels=setup["added_time_latent_channels"], pooling_method=setup["pooling_method"], joint_heads=setup["joint_heads"])
    # gflops, params, used_time = fully_profile_network(net, input_size=(3, 224, 224))
    # else:
    #     net = ResidualUNet3D(in_channels=setup["in_nchannel"], out_channels=setup["in_nchannel"], final_sigmoid=False, f_maps=64, layer_order='gcr',
    #              num_groups=1, num_levels=5, is_segmentation=False, conv_padding=1,)
    print("number of parameters = ",  sum(p.numel() for p in net.parameters() if p.requires_grad))
    if setup["devices"] > 0:
        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import WandbLogger
        from pytorch_lightning.callbacks import LearningRateMonitor , ModelCheckpoint
        import MinkowskiEngine as ME
        from plt_ops import SRF_PLT, MyPrintingCallback

        net = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(net)
        pl_module = SRF_PLT(net, optimizer_name="adamw", lr=setup["lr"], weight_decay=setup["weight_decay"],
                            voxel_size=1.0, batch_size=setup["batch_size"], val_batch_size=setup["batch_size"],
                            train_num_workers=setup["num_workers"], val_num_workers=setup["num_workers"], root_dir=root_dir,
                            srf=srf, setup=setup)
        wandb_logger = WandbLogger(project=PROJECT, entity=USER, id=setup["exp_set"]+"_"+setup["exp_id"], config=setup,reuse="allow")
        checkpoint_callback = ModelCheckpoint(save_top_k=int(setup["enable_checkpointing"]),monitor="val/acc",mode="max",dirpath=setup["output_dir"],filename="checkpoint",save_weights_only=False)
        callbacks = [checkpoint_callback, MyPrintingCallback(), LearningRateMonitor(logging_interval="epoch")]
        trainer = Trainer(max_epochs=setup["max_epochs"], devices=setup["devices"], num_nodes=setup["num_nodes"],
                          check_val_every_n_epoch=setup["val_freq"], accelerator="gpu", strategy=setup["strategy"],
                          default_root_dir=setup["output_dir"], enable_checkpointing=True, precision=setup["precision"],
                          profiler=None, logger=wandb_logger, enable_progress_bar=False, callbacks=callbacks)
        ckpt_path = os.path.join(setup["results_dir"], setup["exp_set"], setup["resume"],"checkpoint.ckpt") if setup["resume"] else None
        if setup["run"] == "train":
            trainer.fit(pl_module, ckpt_path=ckpt_path)
        else :
            trainer.validate(pl_module, ckpt_path=ckpt_path)

        return None
    net.to(device)

    if setup["run"] == "train":
        results = ListDict(setup)
        save_results(setup["results_file"],results)

        if setup["ae_pretraining"]:
            checkpoint = torch.load(setup["ae_weights"])
            net.load_state_dict(checkpoint["state_dict"])    

        train(net, dataloader, val_loader, srf, writer, device, setup)
    else:
        epoch= 0
        checkpoint = torch.load(setup["weights"])
        net.load_state_dict(checkpoint["state_dict"])
        val_metrics,hard_metrics = validate(net,val_loader,epoch,srf,device, setup,visualize=True)
        val_metrics = {"val_"+k:v for k,v in val_metrics.items()}
        record_metrics(val_metrics,writer,epoch)
        c_Results = {'best_loss':val_metrics["val_loss"],'best_ssim':val_metrics["val_ssim"],'best_psnr':val_metrics["val_psnr"],'best_lispip':val_metrics["val_lpips"],'best_acc':val_metrics["val_acc"],"best_diou":val_metrics["val_diou"],"c_epoch":epoch,}     
        print(c_Results)
        results = ListDict(merge_two_dicts(setup, c_Results))
        save_results(setup["results_file"],results)


def run_preprocess(setup): 
    sizes = (int(0.5*setup["vox_res"]),int(0.5*setup["vox_res"]),int(0.5*setup["vox_res"]),setup["vox_res"],setup["vox_res"],setup["vox_res"] )
    resolution = '"[[{}, {}, {}], [{}, {}, {}]]"'.format(*sizes)
    channels_dict = {"sh":setup["sh_dim"],"3d_texture":4,"mlp":9}
    class_root_dir = os.path.join(setup["data_dir"],setup["object_class"])
    if setup["dset_partition"] == -1 :
        all_objects_list = sorted(list(os.listdir(class_root_dir)))
    else:
        partitions = pd.read_csv(os.path.join(
            setup["data_dir"], "SNRL_splits.csv"), sep=",", dtype=str)
        all_objects_list = list(partitions[partitions.partition.isin(
            [str(setup["dset_partition"])]) & partitions.classlabel.isin([str(setup["object_class"])])]["file"])
    for ii, object_name in enumerate(all_objects_list):
        data_dir = os.path.join(class_root_dir,object_name)
        if not os.path.isdir(data_dir):
            continue
        features_dir = os.path.join(data_dir, setup["rf_alias"], "vox{}".format(str(setup["vox_res"])), setup["partial_alias"])
        if os.path.isfile(os.path.join(features_dir, "data_{}.npz".format(setup["rf_variant"]))):
            continue

        # extract feaytres 
        command = "python Svox2/opt/opt.py {} -t {} --basis_type {} --sh_dim {} --basis_reso {} --reso {} --n_train {} --postprune_density_threshold {} --rf_variant {} ".format(
            data_dir, features_dir, setup["basis_type"], channels_dict[setup["basis_type"]], setup["vox_res"], resolution, setup["nb_views"], setup["postprune_density_threshold"], setup["rf_variant"])
        if setup["randomized_views"]:
            command += " --randomized_views"
        run_command_with_time_measure(command)

        if setup["evaluate"]:
            command = "python Svox2/opt/render_imgs.py {} {} --fps 3 --no_imsave --no_vid".format(features_dir, data_dir)
            run_command_with_time_measure(command)
            command = "python Svox2/opt/render_imgs.py {} {} --fps 3 --no_imsave --no_vid --hard".format(
                features_dir, data_dir)
            run_command_with_time_measure(command)

        # visualize
        if setup["visualize"]:
            command = "python Svox2/opt/render_imgs_circle.py {} {} --fps 30 --traj_type {} --radius {} --vec_up {} --num_views {}".format(
                features_dir, data_dir, setup["traj_type"], 0.9, "0.0,1.0,0.0", setup["nb_frames"])
            run_command_with_time_measure(command)


def run_visualize(setup):
    class_root_dir = os.path.join(setup["data_dir"], setup["object_class"])
    if setup["dset_partition"] == -1:
        all_objects_list = sorted(list(os.listdir(class_root_dir)))
    else:
        partitions = pd.read_csv(os.path.join(
            setup["data_dir"], "SNRL_splits.csv"), sep=",", dtype=str)
        all_objects_list = list(partitions[partitions.partition.isin(
            [str(setup["dset_partition"])]) & partitions.classlabel.isin([str(setup["object_class"])])]["file"])

    for ii, object_name in enumerate(all_objects_list):
        if setup["debug_type"] == "plenoxels" and ii not in [0, 2, 3, 7]:
            continue

        data_dir = os.path.join(class_root_dir, object_name)
        if not os.path.isdir(data_dir):
            continue
        features_dir = os.path.join(data_dir, setup["rf_alias"], "vox{}".format(str(setup["vox_res"])), setup["partial_alias"])
        if not os.path.isdir(features_dir) :
            continue

        # evaluate
        if setup["dataset_type"] != "co3d": # TODO make support for co3d evaluation 
            command = "python Svox2/opt/render_imgs.py {} {} --fps 3 --no_imsave --dataset_type {} ".format(
                features_dir, data_dir.setup["dataset_type"])
            run_command_with_time_measure(command)
            command = "python Svox2/opt/render_imgs.py {} {} --fps 3 --no_imsave --hard --dataset_type {} ".format(
                features_dir, data_dir, setup["dataset_type"])
            run_command_with_time_measure(command)
            command = "python Svox2/opt/render_imgs_circle.py {} {} --fps 30 --traj_type {} --radius {} --vec_up {} --num_views {}".format(
                features_dir, data_dir, setup["traj_type"], 0.9, "0.0,1.0,0.0", setup["nb_frames"])
            run_command_with_time_measure(command)
        else:
            seq_id = ii+1  # TODO current error in saving , make it seq_id=ii instead of seq_id=ii+1
            data_dir = setup["data_dir"]
            command = "python Svox2/opt/render_imgs_circle.py {} {} --fps 30 --traj_type {} --radius {} --vec_up {} --num_views {} --dataset_type co3d --config Svox2/opt/configs/co3d.json --seq_id {} --elevation -15 --elevation2 -10 --vert_shift 0.08".format(
                features_dir, data_dir, setup["traj_type"], 1.0, "0.0,-1.0,0.0", setup["nb_frames"], seq_id)
            run_command_with_time_measure(command)

def run_extract(setup):                                                 
    sizes = (int(0.5*setup["vox_res"]),int(0.5*setup["vox_res"]),int(0.5*setup["vox_res"]),setup["vox_res"],setup["vox_res"],setup["vox_res"] )
    resolution = '"[[{}, {}, {}], [{}, {}, {}]]"'.format(*sizes)
    channels_dict = {"sh": setup["sh_dim"], "3d_texture": 4, "mlp": 9}
    class_root_dir = os.path.join(setup["data_dir"],setup["object_class"])
    if setup["dset_partition"] == -1 :
        all_objects_list = sorted(list(os.listdir(class_root_dir)))
    else:
        partitions = pd.read_csv(os.path.join(
            setup["data_dir"], "SNRL_splits.csv"), sep=",", dtype=str)
        all_objects_list = list(partitions[partitions.partition.isin(
            [str(setup["dset_partition"])]) & partitions.classlabel.isin([str(setup["object_class"])])]["file"])
    for ii, object_name in enumerate(all_objects_list):
        if setup["debug_type"] == "plenoxels" and ii not in [0,2,3]:
            continue

        data_dir = os.path.join(class_root_dir,object_name)
        if not os.path.isdir(data_dir):
            continue
        features_dir = os.path.join(data_dir, setup["rf_alias"], "vox{}".format(str(setup["vox_res"])), setup["partial_alias"])
        if os.path.isdir(features_dir) and os.path.isfile(os.path.join(features_dir,"data_{}.npz".format(setup["rf_variant"]))):
            continue
        
        # extract features 
        if setup["dataset_type"] == 'co3d':
            data_dir = setup["data_dir"]
            seq_id = ii+1
        command = "python Svox2/opt/opt.py {} -t {} --basis_type {} --sh_dim {} --basis_reso {} --reso {} --rf_variant {}".format(
            data_dir, features_dir, setup["basis_type"], channels_dict[setup["basis_type"]], setup["vox_res"], resolution, setup["rf_variant"])
        if setup["randomized_views"]:
            command += " --randomized_views"
        if setup["dataset_type"] == 'co3d':
            command += " --dataset_type co3d --config Svox2/opt/configs/co3d_sparf.json --seq_id {}".format(
                seq_id)

        run_command_with_time_measure(command)


        # evaluate
        if setup["evaluate"]:
            command = "python Svox2/opt/render_imgs.py {} {} --fps 3 --no_imsave --no_vid".format(features_dir, data_dir)
            run_command_with_time_measure(command)
            command = "python Svox2/opt/render_imgs.py {} {} --fps 3 --no_imsave --no_vid --hard".format(
                features_dir, data_dir)
            run_command_with_time_measure(command)

        # visualize
        if setup["visualize"]:
            command = "python Svox2/opt/render_imgs_circle.py {} {} --fps 30 --traj_type {} --radius {} --vec_up {} --num_views {}".format(
                features_dir, data_dir, setup["traj_type"], 0.9, "0.0,1.0,0.0", setup["nb_frames"])
            run_command_with_time_measure(command)



def run_render(setup): 
    nb_views = setup["nb_views"] -1 
    t_nb_views = setup["test_nb_views"]
    h_nb_views = setup["hard_nb_views"]
    image_size = setup["img_res"]
    random_views = setup["randomized_views"]
    scale = setup["object_scale"]

    aabb_scale = 1
    train_random_factor = 0.0 # variance of the distance for the train set under `train` directory 
    test_random_factor = 0.0 # variance of the distance for the test set under `test` directory 
    hard_random_factor= 0.4 # variance of the distance for the robustness set under `hard` directory 

    hemisphere = False
    elev = 0
    pitch = 0
    azim  = 0
    # light_color = 100
    # light_intensity = 1.0
    smooth = True
    background = (0, 255, 0, 255)  # None  #
    fov = (45.0, 45.0)

    mesh_root_dir = os.path.join("data","ShapeNetCore.v2",ShapeNet_cat_ids[setup["object_class"]])
    # object_name = "1a6f615e8b1b5ae4dbbc9440457e303e"
    if setup["dset_partition"] == -1 :
        all_objects_list = sorted(list(os.listdir(mesh_root_dir)))
    else:
        partitions = pd.read_csv(os.path.join(
            setup["data_dir"], "SNRL_splits.csv"), sep=",", dtype=str)
        all_objects_list = list(partitions[partitions.partition.isin(
            [str(setup["dset_partition"])]) & partitions.classlabel.isin([str(setup["object_class"])])]["file"])

    for object_name in all_objects_list:
        print("rendering : ", object_name)
        img_nbr = 0
        class_root_dir = os.path.join(setup["data_dir"], setup["object_class"])
        distances = setup["distance"] - np.abs(train_random_factor*np.random.randn(nb_views))
        t_distances = setup["distance"] - np.abs(test_random_factor*np.random.randn(t_nb_views))
        h_distances = setup["distance"] - np.abs(hard_random_factor*np.random.randn(h_nb_views))
        render_root_dir = os.path.join(class_root_dir, object_name)
        rendering_dir = os.path.join(render_root_dir, "train")
        t_rendering_dir = os.path.join(render_root_dir, "test")
        h_rendering_dir = os.path.join(render_root_dir, "hard")
        save_name = os.path.join(render_root_dir, "transforms_train.json")
        t_save_name = os.path.join(render_root_dir, "transforms_test.json")
        h_save_name = os.path.join(render_root_dir, "transforms_hard.json")
        if os.path.isdir(render_root_dir) and os.path.isfile(save_name) and os.path.isfile(t_save_name) and os.path.isfile(h_save_name):
            continue 
        check_folder(class_root_dir)
        check_folder(render_root_dir)
        check_folder(rendering_dir)
        check_folder(t_rendering_dir)
        check_folder(h_rendering_dir)

        if not random_views:
            azims,elevs = unit_spherical_grid(nb_views)

        else:
            azims,elevs = unit_spherical_random(nb_views, hemisphere=hemisphere)

        t_azims,t_elevs = unit_spherical_random(t_nb_views, hemisphere=hemisphere)
        h_azims,h_elevs = unit_spherical_random(h_nb_views, hemisphere=hemisphere)


        if not os.path.isfile(os.path.join(mesh_root_dir,object_name,"models/model_normalized.obj")):
            shutil.rmtree(render_root_dir)
            continue

        mesh = trimesh.load(os.path.join(mesh_root_dir,object_name,"models/model_normalized.obj"))
        mesh.apply_translation(-mesh.centroid)
        mesh.apply_scale(scale)
        # scene.lights[0].color = (light_color,light_color,light_color,255) ;  scene.lights[1].color = (light_color,light_color,light_color,255)
        # scene.lights[0].intensity = scene.lights[0].intensity * light_intensity ;  scene.lights[1].intensity = scene.lights[1].intensity * light_intensity
        # del scene.lights[1]

        scene = trimesh.scene.Scene(mesh)
        valid_mesh = True
        image_name = os.path.join(rendering_dir,str(img_nbr).zfill(3)+".png")
        try:
            save_trimesh_scene(scene, image_name, resolution=(image_size, image_size), distance=setup["distance"], angles=(-np.deg2rad(
                elev), np.deg2rad(azim), np.deg2rad(pitch)), center=(0, 0, 0), show=False, background=background, smooth=smooth, fov=fov)
        except:
            valid_mesh = False
        if not valid_mesh:
            shutil.rmtree(render_root_dir)
            continue
       
        c_dict = create_camera_dict(scene,aabb_scale=aabb_scale)
        c_dict = append_frame_to_camera_dict(scene,c_dict,image_name,scale=scale)

        for ii , (azim,elev,distance) in enumerate(zip(azims.tolist(),elevs.tolist(),distances.tolist())):
            img_nbr = ii +1 
            image_name = os.path.join(rendering_dir,str(img_nbr).zfill(3)+".png")
            save_trimesh_scene(scene,image_name,resolution=(image_size,image_size),distance=distance,angles=(-np.deg2rad(elev),np.deg2rad(azim),np.deg2rad(pitch)),center=(0,0,0),show=False,background=background,smooth=smooth,fov=fov)
            c_dict = append_frame_to_camera_dict(scene,c_dict,image_name,scale=scale)
        save_camera_dict(c_dict,save_name)
        # test images 
        c_dict = create_camera_dict(scene,aabb_scale=aabb_scale)
        for ii , (azim,elev,distance) in enumerate(zip(t_azims.tolist(),t_elevs.tolist(),t_distances.tolist())):
            img_nbr = ii +1 
            image_name = os.path.join(t_rendering_dir,str(img_nbr).zfill(3)+".png")
            save_trimesh_scene(scene,image_name,resolution=(image_size,image_size),distance=distance,angles=(-np.deg2rad(elev),np.deg2rad(azim),np.deg2rad(pitch)),center=(0,0,0),show=False,background=background,smooth=smooth,fov=fov)
            c_dict = append_frame_to_camera_dict(scene,c_dict,image_name,scale=scale)
        save_camera_dict(copy.deepcopy(c_dict),t_save_name)

        # robustness images 
        c_dict = create_camera_dict(scene,aabb_scale=aabb_scale)
        for ii , (azim,elev,distance) in enumerate(zip(h_azims.tolist(),h_elevs.tolist(),h_distances.tolist())):
            img_nbr = ii +1 
            image_name = os.path.join(h_rendering_dir,str(img_nbr).zfill(3)+".png")
            save_trimesh_scene(scene,image_name,resolution=(image_size,image_size),distance=distance,angles=(-np.deg2rad(elev),np.deg2rad(azim),np.deg2rad(pitch)),center=(0,0,0),show=False,background=background,smooth=smooth,fov=fov)
            c_dict = append_frame_to_camera_dict(scene,c_dict,image_name,scale=scale)
        save_camera_dict(c_dict,h_save_name)
 
def run_base(setup): 
    dset = ShapeNetCore(setup["data_dir"], ("train",), 100000, load_textures=False,dset_norm=False,)
    classes = dset.classes
    results = ListDict(["label","shp_id","shp_indx","nb_points","dist","depth"])
    print("classes nb:", len(classes), "number of models: ", len(dset), classes)
    for ii,shp_nbr in enumerate([12,19,18,29,33,75,36,154,57,62,170,66]): # [12,19,18,29,33]
        shapes_dir = os.path.join(setup["output_dir"],str(ii))
        rendering_dir = os.path.join(shapes_dir,"renderings")
        Path(rendering_dir).mkdir(parents=True, exist_ok=True)
        lbl, mesh, GT_POINTS = dset[shp_nbr]
        # mesh.show()
        for img_nbr, nb_points in enumerate([500000]): #  
            print("V:",mesh.vertices.shape[0],"F:",mesh.faces.shape[0])
            points = mesh.sample(nb_points)
            density_threshold=density_threshold_from_nbpoints(nb_points)
            depth = 10
            vertices, faces =  reconstruct_mesh_from_points(points,depth=depth,normals_correction=100,density_threshold=density_threshold,target_faces=20000)
            r_mesh = trimesh.Trimesh(vertices=vertices,faces=faces) 
            # r_mesh.show()
            dist =  np.mean(trimesh.proximity.closest_point(r_mesh, GT_POINTS)[1]) + np.mean(trimesh.proximity.closest_point(mesh,r_mesh.sample(100000) )[1])          
            print("\n\n##### Label:",classes[lbl],"V:",vertices.shape[0],"F:",faces.shape[0],"dis:", dist)
            scene = trimesh.scene.Scene(r_mesh.apply_scale(1.0))
            save_scene_frame(scene ,rendering_dir,img_nbr)
            r_mesh.export(os.path.join(shapes_dir,str(nb_points)+".obj"))
            c_resuls = {"label":classes[lbl],"shp_id":shp_nbr,"shp_indx":ii,"nb_points":nb_points,"dist":100*dist,"depth":depth}
            results.append(c_resuls)
            save_results(os.path.join(setup["output_dir"],"results.csv"),results)
        gif_folder(rendering_dir,"png",duration=0.5)


def save_scene_frame(scene ,rendering_dir,img_nbr,resolution=(400,400),distance=1.3,angles=(-0.6,-1.9,0)):
    image_name = os.path.join(rendering_dir,str(img_nbr).zfill(3)+".png")
    save_trimesh_scene(scene,image_name,resolution=resolution,distance=distance,angles=angles,center=(0,0,0),show=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--obj_path', type=str, default='meshes/mesh1.obj') # the init mesh file 
    parser.add_argument('--data_dir', type=str, default='data/ShapeNetRend') # the init mesh file 
    parser.add_argument('--dset_partition', type=int, default=-1) # partition -1 means all the dataaset

    # PL utilties 
    parser.add_argument('--devices', type=int, default=1) # nb of gpus to run the model
    parser.add_argument('--num_nodes', type=int, default=1) # nb of nodes in a cluster
    parser.add_argument('--precision', type=int, default=32) # amp precision
    parser.add_argument('--accelerator', type=str, default='gpu') # amp precision
    parser.add_argument('--strategy', type=str, default='ddp') # amp precision
    # parser.add_argument('--prompt', nargs="+", default='a pig with pants') # the text to geenrate
    # parser.add_argument('--normprompt', nargs="+", default=None) # ?????????????????
    # parser.add_argument('--promptlist', nargs="+", default=None) # not used ( multiple prompts)
    # parser.add_argument('--normpromptlist', nargs="+", default=None)
    # parser.add_argument('--image', type=str, default=None) # use the image as the condition for image geernation 
    # # parser.add_argument('--output_dir', type=str, default='round2/alpha5') # the output directery for all the results
    # parser.add_argument('--traintype', type=str, default="shared") #????????????
    # parser.add_argument('--sigma', type=float, default=10.0) # hyper parameter for the neural style network
    # parser.add_argument('--normsigma', type=float, default=10.0) # ???????????????
    # parser.add_argument('--depth', type=int, default=4) # the network depth
    # parser.add_argument('--width', type=int, default=256) # the network width 
    # parser.add_argument('--colordepth', type=int, default=2) # network
    # parser.add_argument('--normdepth', type=int, default=2) # network
    # parser.add_argument('--normwidth', type=int, default=256) # network
    # parser.add_argument('--learning_rate', type=float, default=0.0005)
    # parser.add_argument('--normal_learning_rate', type=float, default=0.0005) # ?????????
    # parser.add_argument('--decay', type=float, default=0) # lr
    # parser.add_argument('--lr_decay', type=float, default=1) # lr
    # parser.add_argument('--lr_plateau', action='store_true') # lr
    # parser.add_argument('--no_pe', dest='pe', default=True, action='store_false') # no pregressive encoding 
    # parser.add_argument('--decay_step', type=int, default=100) # lr
    parser.add_argument('--nb_views', type=int, default=400)
    parser.add_argument('--test_nb_views', type=int, default=20)
    parser.add_argument('--hard_nb_views', type=int, default=10)

    # parser.add_argument('--n_augs', type=int, default=0) # augmentations 
    # parser.add_argument('--n_normaugs', type=int, default=0) # augmentation with normlaization  
    # parser.add_argument('--n_iter', type=int, default=3000)
    # parser.add_argument('--encoding', type=str, default='gaussian') # position encoding 
    # parser.add_argument('--normencoding', type=str, default='xyz') 
    # parser.add_argument('--layernorm', action="store_true") # network 
    parser.add_argument('--run', type=str, default=None)
    # parser.add_argument('--gen', action='store_true') # ?????????????
    # parser.add_argument('--clamp', type=str, default="tanh")
    # parser.add_argument('--normclamp', type=str, default="tanh")
    # parser.add_argument('--normratio', type=float, default=0.1)
    # parser.add_argument('--frontview', action='store_true')
    # parser.add_argument('--no_prompt', default=False, action='store_true')
    # parser.add_argument('--exclude', type=int, default=0) # network 

    # parser.add_argument('--frontview_std', type=float, default=8) # for fromt view visualization 
    # parser.add_argument('--frontview_center', nargs=2, type=float, default=[0., 0.])
    # parser.add_argument('--clipavg', type=str, default=None) # use the average of clip or "view" 
    # parser.add_argument('--geoloss', action="store_true") # ????????????????????????????????????????????
    # parser.add_argument('--samplebary', action="store_true") # ??????????????????
    # parser.add_argument('--promptviews', nargs="+", default=None) #???????????????????????
    # parser.add_argument('--mincrop', type=float, default=1)
    # parser.add_argument('--maxcrop', type=float, default=1)
    # parser.add_argument('--normmincrop', type=float, default=0.1)
    # parser.add_argument('--normmaxcrop', type=float, default=0.1)
    # parser.add_argument('--splitnormloss', action="store_true")
    # parser.add_argument('--splitcolorloss', action="store_true")
    # parser.add_argument("--nonorm", action="store_true")   
    # parser.add_argument('--cropsteps', type=int, default=0)
    # parser.add_argument('--cropforward', action='store_true')
    # parser.add_argument('--cropdecay', type=float, default=1.0)
    # parser.add_argument('--decayfreq', type=int, default=None)
    # parser.add_argument('--overwrite', action='store_true')
    # parser.add_argument('--show', action='store_true')
    # parser.add_argument('--background', nargs=3, type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--save_render', action="store_true")
    # parser.add_argument('--input_normals', default=False, action='store_true') #use normals from the mesh 
    # parser.add_argument('--symmetry', default=False, action='store_true')
    # parser.add_argument('--only_z', default=False, action='store_true') # use 1 dim images 
    # parser.add_argument('--standardize', default=False, action='store_true') # normalize the input 

    # my addiitions 
    parser.add_argument('--dataset_type',choices=['nerf', 'co3d',],default='nerf',help='dset type to load posed images')

    parser.add_argument('--exp_set', type=str, default="00") # use the image as the condition for image geernation 
    parser.add_argument('--exp_id', type=str, default="random") # use the image as the condition for image geernation 
    # parser.add_argument('--clipsize', type=str, default='B')
    parser.add_argument('--basis_type',choices=['sh', '3d_texture', 'mlp'],default='sh',help='Basis function type')
    parser.add_argument('--traj_type',
                        choices=['spiral', 'circle', "zoom", "spiralzoom","vertical","test", "hard"],
                    default='zoom',
                    help="Render a spiral (doubles length, using 2 elevations), or just a cirle")
    parser.add_argument("--distance", type=float, default=1.5)
    parser.add_argument("--object_scale", type=float, default=1.0)


    parser.add_argument('--visualize', action='store_true',default=False) # viisualize the output 
    parser.add_argument('--evaluate', action='store_true',default=False) # evaluate the optimized radiance fields 
    parser.add_argument('--visualize_input', action='store_true',default=False) # viisualize the output 
    parser.add_argument('--evaluate_robustness', action='store_true',default=False) # viisualize the output 
    parser.add_argument('--evaluate_input', action='store_true',default=False) # viisualize the output 


    parser.add_argument('--validate_training', action='store_true',default=False) # run a vlidation loop on the trainign set and validation set at val_freq
    parser.add_argument('--visualize_training', action='store_true',default=False) # viisualize the output 
    parser.add_argument('--visualize_input_training', action='store_true',default=False) # viisualize the output 
    parser.add_argument('--visualize_gt', action='store_true',default=False) # viisualize the GT 
    parser.add_argument('--visualize_input_imgs', action='store_true',default=False)  # viisualize the input imgs
    parser.add_argument('--concat_gt_output', action='store_true',
                        default=False)  # viisualize the GT
    

    parser.add_argument('--short_test', action='store_true',default=False) # run the validation on a shorter validation set for speed 
    parser.add_argument('--visualizations_nb', type=int, default=50) # nb of visulaiations and test examples if `short_test` is used 

    parser.add_argument('--dense_pipeline', action='store_true',default=False) # lr
    parser.add_argument('--normalize_input', type=str, default="const",
                        choices=["const", "none", "tanh","sigmoid"])  # lr
    parser.add_argument('--encode_cameras', action='store_true',default=False) # encode the cameras to the input
    parser.add_argument('--encode_imgs', action='store_true',default=False) # encode the imgs to the input
    parser.add_argument('--encode_time', action='store_true',default=False) # encode the time to the input
    parser.add_argument('--time_steps', type=int, default=1) # total nb of 
    parser.add_argument("--diffusion_type", type=str, default="none",choices=["none","none","quantize","squantize","gaussian","gaussianmix","egaussian"])
    parser.add_argument("--kernel_std", type=float, default=0.444)
    parser.add_argument('--diffusion_kernel_size', type=int, default=100_000) # nb of visulaiations and test examples if `short_test` is used 
    parser.add_argument('--ignore_input',action='store_true',default=False) # nb of visulaiations and test examples if `short_test` is used 
    parser.add_argument('--uniform_loss',action='store_true',default=False) # uniform cls points in the output voxel space 

    parser.add_argument('--num_encoding_functions', type=int, default=0) # positional emcoding of the cameras if `encode_cameras` == True
    parser.add_argument("--strides", type=int, default=2) # num,ber of strides in Minkoskynet conv and upconv
    parser.add_argument("--network_depth", type=int, default=0) # number of subbmodules in Minkoskynet , 1 is default large U-Net
    parser.add_argument("--support_network_depth", type=int, default=1) # number of subbmodules in the support 
    parser.add_argument("--pooling_method", type=str, default="max",choices=["max","mean","meanmax"])

    parser.add_argument("--cam_latent_channels", type=int, default=32) # num,ber of strides in Minkoskynet conv and upconv
    parser.add_argument("--img_latent_channels", type=int, default=128) # num,ber of strides in Minkoskynet conv and upconv
    parser.add_argument("--time_latent_channels", type=int, default=32)


    parser.add_argument('--cluster_run', action='store_true',default=False) # lr

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--img_res', type=int, default=400)
    parser.add_argument('--vox_res', type=int, default=512)
    parser.add_argument('--object_class', type=str, default="chair") # the object class  to perform the exp 
    # parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--pre_optim_steps", type=int, default=1 * 12800)
    parser.add_argument("--post_optim_steps", type=int, default=1 * 12800)
    parser.add_argument('--record_freq', type=int, default=1000)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--visualize_freq", type=int, default=50)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr_decay", type=float, default=0.99)
    parser.add_argument("--l1_beta", type=float, default=1.0)
    parser.add_argument('--use_adv_loss', action='store_true',default=False) # allow using a lower resolution in supervising the training 
    parser.add_argument("--loss_type", type=str, default="l1",choices=["l1","l2","bcel1","bcel2","ce"])
    parser.add_argument("--lambda_reg", type=float, default=0.0) # the regualarizer L1 of the output of the network 
    parser.add_argument("--lambda_cls", type=float, default=1.0) # the regualarizer BCE loss of the output density of the network 
    parser.add_argument('--ignore_loss_mask', action='store_true',default=False) # ignore the mask when defning the loss 
    parser.add_argument("--lambda_main", type=float, default=1.0) # the factor of the main loss 
    parser.add_argument("--lambda_adv", type=float, default=0.0) # the factor of the main loss 
    parser.add_argument("--lambda_2d", type=float, default=0.0) # the factor of the main loss 
    parser.add_argument("--sh_dim", type=int, default=4) # the spherical harmonics nb of channels per color of RGB ( max 10)
    parser.add_argument("--input_sh_dim", type=int, default=1) # the spherical harmonics nb of channels per color of RGB ( max 10) for the input of the network
    parser.add_argument("--density_normilzation_factor",type=float, default=10_000.0)  # the factor of the main loss
    parser.add_argument("--colors_normilzation_factor", type=float, default=10.0) # the factor of the main loss 

    parser.add_argument("--rf_variant", type=int, default=0) # The variant number of the optimized radiance field 
    parser.add_argument("--nb_rf_variants", type=int, default=1) # The number of variants of NErfs used in training  


    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weights", type=str, default="checkpoint.ckpt")
    parser.add_argument("--load_optimizer", type=str, default="true")
    parser.add_argument('--enable_pruning', action='store_true',default=False) # pruning for ME Net
    parser.add_argument('--prune_last_layer', action='store_true',default=False) # pruning for ME Net
    parser.add_argument('--learn_cam_embed', action='store_true',default=False) # 
    parser.add_argument('--learn_time_embed',action='store_true', default=False)
    parser.add_argument('--enable_checkpointing', action='store_true',default=False) # default  not save checkppoints to save space 
    parser.add_argument('--resume', type=str, default="")
    parser.add_argument('--ae_pretraining', action='store_true',default=False) # 
    parser.add_argument('--train_ae', action='store_true',default=False) # tain an AE of the sparse voxels data 
    parser.add_argument('--use_lower_res', action='store_true',default=False) # allow using a lower resolution in supervising the training 
    parser.add_argument('--use_multi_stage_loss', action='store_true',default=False) # allow using a lower resolution in supervising the training 
    parser.add_argument('--use_2d_masks', action='store_true',default=False) # allow using a lower resolution in supervising the training 
    parser.add_argument("--mask_2dbg_alpha", type=float, default=0.0)
    parser.add_argument('--kill_density_grads',action='store_true',default=False) # kill grads of density when training wiht 2d Loss 
    parser.add_argument("--online_nb_views", type=int, default=3)
    parser.add_argument('--joint_heads',action='store_true',default=False) # seperate heads of the SRFNet 
    parser.add_argument('--quantize_online_rendering',action='store_true',default=False) # quantize online rendering ( slow down a bit but prevent some errors with repeated indices predictions )
    parser.add_argument("--delay_2d_epochs", type=int, default=0)

    parser.add_argument('--randomized_views', action='store_true',default=False) # 
    parser.add_argument("--reflect_type", type=str, default="preprocessed",choices=["standard","fast","faster","preprocessed","volume","depth"])
    parser.add_argument("--rays_batch_size", type=int, default=2000)
    # parser.add_argument('--use_upconvs', action='store_true',default=False) # use upconvolution in ME network 

    parser.add_argument("--input_quantization_size", type=float, default=1.0) # quantize the input to reduce the size of input ( especially for large voxels) default no quantization with 1.0 , 2.0 is alot 
    parser.add_argument('--post_optimize', action='store_true',default=False) # use upconvolution in ME network 
    parser.add_argument('--extract_mesh', action='store_true',default=False) # extract 
    parser.add_argument('--render_extracted_mesh',
                        action='store_true', default=False)  # extract
    parser.add_argument('--nb_frames', type=int, default=200) # the number of frames if 
    parser.add_argument('--gif', action='store_true',default=False) # save frames as gif instead of mp4


##  augmentations 
    parser.add_argument("--augment_type", type=str, default="none",choices=["none","axis_rotate","free_rotate","random_color","axis_rotate_random_color","free_rotate_random_color","rotate_color","primary_color"])
    parser.add_argument("--augment_prop", type=float, default=0.5)
    parser.add_argument("--clean_type", type=str, default="none",choices=["none","clean"])
    parser.add_argument("--lambda_clean", type=float, default=0.0)
    parser.add_argument("--clean_tau", type=float, default=0.05)
    parser.add_argument("--clean_annealing_Factor", type=float, default=0.95)
    parser.add_argument("--mask_type", type=str, default="nonempty",choices=["nonempty","full","dense","weighted","sliced","points","densepoints"])
    parser.add_argument("--density_threshold", type=float, default=0.0) # the density threshold that dtermine solid from air (for evlauations)
    parser.add_argument("--loss_density_threshold", type=float, default=0.0) # the density threshold that dtermine solid from air (for loss)
    # the density threshold that dtermine solid from air (for loss)
    parser.add_argument("--loss_points_proportion", type=float, default=40.0)
    parser.add_argument('--prune_input', action='store_true',default=False) # pruning for ME Net
    parser.add_argument('--add_input_late', action='store_true',default=False) # add the RF input at multiple stages 



    parser.add_argument("--postprune_density_threshold",
                        type=float, default=-10000.0)

    parser.add_argument("--density_mask_tightness", type=float, default=1.0) # the density weights tightness if mask_type == weighted

    parser.add_argument("--debug_type", type=str, default="none",choices=["none","features","density","fast","plenoxels"])
    # parser.add_argument('--small_data', action='store_true',default=False) # using a small portion of the dataset


    # parser.add_argument('--output_variant', type=str, default='buzz') # the output directery for all the results
    # parser.add_argument('--lods', type=int, default=1) # level of details model number , default no == 1 level 
    


    args = parser.parse_args()

    run_branched(args)
