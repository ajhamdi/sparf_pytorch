# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#coords, feats = ME.utils.sparse_collate(
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
from contextlib import redirect_stdout
import matplotlib.pyplot as plt

from multiprocessing import reduction
import os
from shutil import Error, ExecError
import sys
import imageio
import logging
import numpy as np
from time import time
import urllib
import copy
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import quantile
import wandb

# Must be imported before large libs
try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from datasets import ShapeNetRend
from Svox2.opt.reflect import get_timestep_embedding


import MinkowskiEngine as ME
from extra_utils import check_folder , merge_two_dicts ,ListDict, save_results , rotate3d_tensor , EmptyModule , torch_random_unit_vector , rotation_matrix ,  torch_random_angles , torch_color , save_trimesh_scene , concat_horizontal_videos , profile_op
from torch.utils.data.sampler import Sampler
from Svox2.svox2.utils import eval_sh_bases
import trimesh

M = np.array(
    [
        [0.80656762, -0.5868724, -0.07091862],
        [0.3770505, 0.418344, 0.82632997],
        [-0.45528188, -0.6932309, 0.55870326],
    ]
)

assert (
    int(o3d.__version__.split(".")[1]) >= 8
), f"Requires open3d version >= 0.8, the current version is {o3d.__version__}"

# if not os.path.exists("ModelNet40"):
#     logging.info("Downloading the pruned ModelNet40 dataset...")
#     subprocess.run(["sh", "./examples/download_modelnet40.sh"])


###############################################################################
# Utility functions
###############################################################################


class InfSampler(Sampler):
    """Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)

class CollationAndTransformation:
    def __init__(self, resolution, augment_data, use_lower_res):
        self.resolution = resolution
        self.augment_data = augment_data
        self.use_lower_res = use_lower_res

    def random_crop(self, coords_list):
        crop_coords_list = []
        for coords in coords_list:
            sel = coords[:, 0] < self.resolution / 3
            crop_coords_list.append(coords[sel])
        return crop_coords_list

    def __call__(self, list_data):
        coords, feats, labels,  in_coords_, in_feat_, c2ws,cam_embed, imgs,masks, l_coords, l_feats, t, t_embed, in_rf_variant = list(
            zip(*list_data))
        # coords, feats, labels = list(zip(*list_data))

        if self.augment_data:
            coords = self.random_crop(coords)

        # Concatenate all lists
        # print("@@@@@@@@@@@@@@@@@@@@@@@@",len(coords),len(feats),coords[0].shape,feats[0].shape)
        
        coords, feats = ME.utils.sparse_collate(coords=coords, feats=[feat.float() for feat in feats])
        # try:
        #     print("$$$$$$$$$$$", in_coords_[0].shape)
        # except:
        #     print("########## FOUND", t)
        #     raise Exception
        in_coords_, in_feat_,  = ME.utils.sparse_collate(coords=in_coords_, feats=[feat.float() for feat in in_feat_])

        if self.use_lower_res :
            l_coords, l_feats,  = ME.utils.sparse_collate(coords=l_coords, feats=[feat.float() for feat in l_feats])

        # feats = [feat.float() for feat in feats]

        return {
            "coords": coords,
            "feats": feats,
            "cropped_coords": coords,
            "labels": labels, #torch.LongTensor(labels),
            "in_coords": in_coords_,
            "in_feats": in_feat_,
            "l_coords": l_coords,
            "l_feats": l_feats,
            "c2ws": torch.cat(c2ws,dim=0) , 
            "cam_embed": torch.cat(cam_embed, dim=0),
            "imgs": torch.cat(imgs, dim=0),
            "masks": torch.cat(masks, dim=0),
            "t": torch.cat(t, dim=0),
            "in_rf_variant": torch.cat(in_rf_variant, dim=0),
            "t_embed": torch.cat(t_embed, dim=0)
        }





def make_data_loader(phase,data_dir, batch_size, shuffle, num_workers, repeat, object_class="chair",augment_data=False,drop_last=True,dset_partition=-1,srf=None,use_lower_res=False):
    dset = ShapeNetRend(data_dir, phase, object_class=object_class,
                        dset_partition=dset_partition, srf=copy.deepcopy(srf), use_lower_res=use_lower_res)

    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": CollationAndTransformation(srf.vox_res, augment_data=augment_data, use_lower_res=use_lower_res),
        "pin_memory": False,
        "drop_last": drop_last,
    }

    if repeat:
        args["sampler"] = InfSampler(dset, shuffle)
    else:
        args["shuffle"] = shuffle

    loader = torch.utils.data.DataLoader(dset, **args)

    return loader


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d %H:%M:%S",
    handlers=[ch],
)

def record_metrics(metrics,writer,epoch):
    for k,v in metrics.items():
        kk = k.replace("_","/")
        writer.add_scalar(kk,v,epoch)
        wandb.log({kk:v,"epoch":epoch})

def train(net, train_loader,val_loader,srf,writer, device, setup):
    c_Results = {'best_loss':float("inf"),'best_ssim':0,'best_psnr':0,'best_lispip':float("inf"),'best_acc':0,"best_diou":0,"c_epoch":0,'training_ssim':0,'training_psnr':0,'training_loss':float("inf"),"training_acc":0,"training_lpips":0}     

    rf_augmenter = RadianceFieldAugmenter(augment_type=setup["augment_type"], augment_prop=setup["augment_prop"],vox_res=setup["vox_res"], feat_size=setup["in_nchannel"], batch_size=setup["batch_size"], sh_dim=setup["sh_dim"]).to(device)
    rf_cleaner =  RadianceFieldCleaner(clean_type=setup["clean_type"], lambda_clean=setup["lambda_clean"],clean_tau = setup["clean_tau"] ,clean_annealing_Factor=setup["clean_annealing_Factor"] )

    net.train()


    optimizer = optim.AdamW(
        net.parameters(),
        lr=setup["lr"],
        betas=(setup["momentum"],0.999),
        weight_decay=setup["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, setup["lr_decay"])

    BCEcrit = nn.BCEWithLogitsLoss()
    MSEcrit = nn.L1Loss(reduction="none") if "l1" in setup["loss_type"] else nn.MSELoss(reduction="none")

    # MSEcrit = nn.SmoothL1Loss(reduction="sum",beta=setup["l1_beta"])



    for epoch in range(setup["max_epochs"]):
        print("######################" ,epoch)
        c_lambda_2d = setup["lambda_2d"] if epoch >= setup["delay_2d_epochs"] else 0.0
        losses = []
        tr_ssim , tr_psnr , tr_lpips , tr_acc, dious = [],[],[],[],[]
        all_feats = []


        # val_iter = iter(val_dataloader)
        logging.info(f"LR: {scheduler.get_lr()}")
        for i,data_dict in enumerate(train_loader):
            if i >= 0  and setup["debug_type"] == "fast":
                continue 

            # s = time()
            size_tensor = torch.empty((len(data_dict["labels"]),setup["out_nchannel"],setup["vox_res"],setup["vox_res"],setup["vox_res"])).size()
            # sizes = (int(0.5*setup["vox_res"]),int(0.5*setup["vox_res"]),int(0.5*setup["vox_res"]),setup["vox_res"],setup["vox_res"],setup["vox_res"] )
            starget = ME.SparseTensor(
                features=data_dict["feats"],
                coordinates=data_dict["coords"],
                device=device,
                requires_grad=False
            )
            # print("$$$$$$$$$$$", data_dict["coords"].max(),data_dict["coords"].min(),data_dict["feats"].shape)

            optimizer.zero_grad()
            if not setup["train_ae"]:
                sin = ME.SparseTensor(
                    features=data_dict["in_feats"],
                    coordinates=data_dict["in_coords"],
                    device=device,
                    requires_grad=False
                )

            else :
                sin = starget #############################################################################################################
            # sin = srf.batched_forward_diffusion(sin, data_dict["t"])
            sin, starget = rf_augmenter(sin, starget)
            if setup["debug_type"] == "density":
                sin.F[:,0][sin.F[:,0] > setup["density_threshold"]] = 1.0
                sin.F[:,0][sin.F[:,0] < setup["density_threshold"]] = - 1.0
                starget.F[:,0][starget.F[:,0] > setup["density_threshold"]] = 1.0
                starget.F[:,0][starget.F[:,0] < setup["density_threshold"]] = -1.0
            if setup["debug_type"] == "features":
                all_feats.extend(starget.F.detach().cpu().numpy().tolist())

            # oppp = ME.MinkowskiPoolingTranspose(4, stride=1, dilation=1, dimension=3)
            # sin = ME.SparseTensor(features=data_dict["l_feats"],
            #                 coordinates=data_dict["l_coords"], device=device, requires_grad=False)
#####################3
            if setup["visualize_input_training"]:
                srf.construct_and_visualize_batch_stensor(sin, data_dict["labels"], render_dir=setup["render_dir"], vizualization_id=setup["batch_size"]*i,
                                                                      traj_type=setup["traj_type"], input_construct=True, gif=setup["gif"], nb_frames=setup["nb_frames"])

                # sin = sin + ME.SparseTensor(features=points_feats, coordinates=points,
                #                             coordinate_manager=sin.coordinate_manager, device=device, requires_grad=False)
                # for ii,d_dir in enumerate(data_dict["labels"]):
                    # T = setup["time_steps"]
                    # for t in range(T):
                    # coords = sin.coordinates_at(ii)
                    # feats = sin.features_at(ii)
                        # print("$$$$$$$$$$$", coords.max(),coords.min(), feats.shape,feats.max(),feats.min())
                        # coords, feats = srf.forward_diffusion(coords, feats, t=t)
                        # grid = srf.construct_grid(d_dir, coords, feats, device=device, input_construct=True)
                        # srf.visualize_grid(grid, d_dir, setup["render_dir"], num_views=setup["nb_frames"], vizualization_id=t + 100*(
                        #     setup["batch_size"]*i + ii), device=device, gif=setup["gif"], traj_type=setup["traj_type"])
                    # srf.construct_and_visualize_grid(coords, feats, d_dir, render_dir=setup["render_dir"], vizualization_id=setup["batch_size"]*i + ii,
                    #                             traj_type=setup["traj_type"], input_construct=True, gif=setup["gif"], nb_frames=setup["nb_frames"])
#############################33
            # Generate target sparse tensor
            # try :
            loss ,adv_loss, sout, density_iou, sparsity = define_loss(net,sin, starget,data_dict, setup, BCEcrit, MSEcrit, size_tensor, srf=srf,c_lambda_2d=c_lambda_2d)            
            # except:
            #     print(data_dict["labels"])
            optimizer.zero_grad()

            loss.backward()
            grads = [p.grad.cpu().norm() for p in net.parameters() if p.grad is not None]       
            # grads = [p.grad.cpu().norm() for p in net.cam_encoder.parameters() if p.grad is not None]
            # grads = [p.grad.cpu().norm() for p in net.img_encoder.parameters() if p.grad is not None]
            # grads = [p.grad.cpu().norm() for p in net.time_encoder.parameters() if p.grad is not None]
            optimizer.step()




            losses.append(loss.item())
            dious.append(density_iou)
            del density_iou , loss

            # if not bool(setup["lambda_main"]):
            #     sout.F[:,1::] = dout[:,1::,...] = 0.0 
            # t = time() - s
            torch.cuda.empty_cache()
            writer.add_scalar('train/loss',losses[-1],epoch*len(train_loader)+i)
            writer.add_scalar('train/grads',np.mean(grads),epoch*len(train_loader)+i)
            writer.add_scalar('train/diou',dious[-1],epoch*len(train_loader)+i)
            writer.add_scalar('train/sparsity', sparsity,epoch*len(train_loader)+i)
            writer.add_scalar('train/adv_loss', adv_loss,epoch*len(train_loader)+i)

            wandb.log({'train/loss':losses[-1]},step=epoch*len(train_loader)+i)
            wandb.log({'train/grads':np.mean(grads)},step=epoch*len(train_loader)+i)
            wandb.log({'train/diou':dious[-1]},step=epoch*len(train_loader)+i)
            wandb.log({'train/sparsity': sparsity},step=epoch*len(train_loader)+i)
            wandb.log({'train/adv_loss': adv_loss},step=epoch*len(train_loader)+i)





            short_list_cond = i*setup["batch_size"] in range(setup["visualizations_nb"])
            if setup["validate_training"] and (epoch+1) % setup["val_freq"] == 0 and  short_list_cond:
                # print(coords.shape, feats.shape)
                # coords, feats = starget.coordinates.data, starget.features.data

                for ii,d_dir in enumerate(data_dict["labels"]):
                    coords_ = sout.coordinates_at(ii)
                    feats_ = sout.features_at(ii)

                    # print("$$$$$$$$$$$$$$$$$$,", coords_.shape)
                    coords_, feats_ = srf.crop_sparse_voxels(coords_,feats_) # to limit coords outside the vox reolution 

                    # print("$$$$$$$$$$$$$$$$$$,", coords_.max(),feats_.shape, coords_[0, :2], feats_[0, :2], feats_.max())
                    gt_metrics = get_GT_metrics(os.path.join(d_dir,srf.relative_features_dir,"test_metrics"))
                    grid = srf.construct_grid(d_dir, coords_, feats_)
                    pred_metrics = srf.evaluate_grid(grid, d_dir)
                    if ((epoch+1) % setup["visualize_freq"] == 0) and setup["visualize_training"]:
                        render_dir = os.path.join(setup["render_dir"],"tr_"+str(epoch))
                        check_folder(render_dir)
                        grid = srf.construct_grid(d_dir, coords_, feats_)
                        srf.visualize_grid(grid, d_dir, render_dir, num_views=setup["nb_frames"], vizualization_id=setup[
                                       "batch_size"]*i + ii, gif=setup["gif"], traj_type=setup["traj_type"])
                    tr_ssim.append(pred_metrics["SSIM"])
                    tr_psnr.append(pred_metrics["PSNR"])
                    tr_lpips.append(pred_metrics["LPIPS"])
                    tr_acc.append(100.0*pred_metrics["PSNR"]/gt_metrics["PSNR"])
                    del coords_ , feats_ , grid
                del sout
                torch.cuda.empty_cache()
        if setup["debug_type"] == "features":
            np.save(os.path.join(setup["output_dir"],"features.npz"),np.array(all_feats))
        scheduler.step()
        logging.info(f"LR: {scheduler.get_lr()}")

        if (epoch+1) % setup["record_freq"] == 0:
            logging.info(
                f"Iter: {i}, Loss: {np.mean(losses):.3e}"
            )
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "curr_iter": i,
                },
                setup["weights"],
            )

        if (epoch+1) % setup["val_freq"] == 0:
            val_metrics,hard_metrics = validate(net, val_loader, epoch, srf, device, setup, visualize=setup["visualize"] and (
                (epoch+1) % setup["visualize_freq"] == 0))
            val_metrics = {"val_"+k: v for k, v in val_metrics.items()}
            record_metrics(val_metrics,writer,epoch)
            if setup["evaluate_robustness"]:
                hard_metrics = {"hard_"+k: v for k, v in hard_metrics.items()}
                record_metrics(hard_metrics, writer, epoch)

            if val_metrics["val_psnr"] > c_Results['best_psnr']:
                c_Results = {'best_loss':val_metrics["val_loss"],'best_ssim':val_metrics["val_ssim"],'best_psnr':val_metrics["val_psnr"],'best_lispip':val_metrics["val_lpips"],'best_acc':val_metrics["val_acc"],"best_diou":val_metrics["val_diou"],"c_epoch":epoch,'training_ssim':0,'training_psnr':0,'training_loss':np.mean(losses),"training_acc":0,"training_lpips":0}     
            results = ListDict(merge_two_dicts(setup, c_Results))
            save_results(setup["results_file"],results)
            if setup["validate_training"]:
                metrics = {"loss":np.mean(losses) ,"ssim": np.mean(tr_ssim) ,"psnr": np.mean(tr_psnr),"lpips": np.mean(tr_lpips),"acc": np.mean(tr_acc),"diou":np.mean(dious)}
                training_metrics = {"training_"+k:v for k,v in metrics.items()}
                record_metrics(training_metrics,writer,epoch)
                c_Results.update(training_metrics)
                results = ListDict(merge_two_dicts(setup, c_Results))
                save_results(setup["results_file"],results)

            # visualize(net, val_loader, device, setup)        
            net.train()



def validate(net,val_loader,epoch,srf,device, setup,visualize=False):
    net.eval()
    losses = []
    val_ssim , val_psnr, val_lpips, val_acc, dious = [],[],[],[],[]
    hard_ssim, hard_psnr, hard_lpips, hard_acc = [], [], [], []

    BCEcrit = nn.BCEWithLogitsLoss()
    MSEcrit = nn.L1Loss(reduction="none") if "l1" in setup["loss_type"] else nn.MSELoss(reduction="none")

    for i,data_dict in enumerate(val_loader):
        # print("@@@@@@@@@@@@@@@@@@@@" ,i)
        # if i > 2 and setup["debug_type"] == "fast":
        #     continue 

        short_list_cond = i*setup["batch_size"] in range(0,0+setup["visualizations_nb"])
        if setup["short_test"] and not short_list_cond :
            continue 
        # s = time()
        size_tensor = torch.empty((len(data_dict["labels"]),setup["out_nchannel"],setup["vox_res"],setup["vox_res"],setup["vox_res"])).size()
        starget = ME.SparseTensor(
            features=data_dict["feats"],
            coordinates=data_dict["coords"],
            device=device,
            requires_grad=False
        )

        if not setup["train_ae"]:
            sin = ME.SparseTensor(
                features=data_dict["in_feats"],
                coordinates=data_dict["in_coords"],
                device=device,
                requires_grad=False
            )
        else : 
            sin = starget #############################################################################################################

        if setup["visualize_gt"]:
            gt_render_dir = os.path.join(setup["gt_dir"], srf.relative_features_dir,"vids")
            os.makedirs(gt_render_dir, exist_ok=True)
            if not os.path.isfile(os.path.join(gt_render_dir, "{}_renders_{}.mp4".format(setup["traj_type"],setup["batch_size"]*(i+1)-1))):
                srf.construct_and_visualize_batch_stensor(starget, data_dict["labels"], render_dir=gt_render_dir, vizualization_id=setup["batch_size"]*i,
                                                      traj_type=setup["traj_type"], input_construct=False, gif=setup["gif"], nb_frames=setup["nb_frames"])


        if setup["visualize_input"]:
            in_render_dir_vids = os.path.join(setup["gt_dir"], srf.relative_reflect_dir,"vids")
            in_render_dir_imgs = os.path.join(setup["gt_dir"], srf.relative_reflect_dir,"imgs")
            os.makedirs(in_render_dir_vids, exist_ok=True)
            os.makedirs(in_render_dir_imgs, exist_ok=True)
            srf.construct_and_visualize_imgs_batch_stensor(sin, data_dict, render_dir_vids=in_render_dir_vids, render_dir_imgs=in_render_dir_imgs, vizualization_id=setup["batch_size"]*i,
                                                      traj_type=setup["traj_type"], input_construct=True, gif=setup["gif"], nb_frames=setup["nb_frames"])
        if setup["evaluate_input"]:
            sout = sin
            loss = 0
            density_iou = 0
            losses.append(loss)

        else:
            with torch.no_grad():
                loss ,_ ,  sout, density_iou, _  = define_loss(net,sin, starget,data_dict, setup, BCEcrit, MSEcrit, size_tensor,srf=srf,c_lambda_2d=0.0)

            losses.append(loss.item())
        dious.append(density_iou)
        # if not bool(setup["lambda_main"]):
        #     sout.F[:,1::] = dout[:,1::,...] = 0.0 

        del loss 

        for ii,d_dir in enumerate(data_dict["labels"]):
            coords_ = sout.coordinates_at(ii)

            feats_ = sout.features_at(ii)
            coords_, feats_ = srf.crop_sparse_voxels(coords_,feats_) # to limit coords outside the vox reolution 

            # print("$$$$$$$$$$$$$$$$$$,",coords_.max(),feats_.shape)
            gt_metrics = get_GT_metrics(os.path.join(d_dir,srf.relative_features_dir,"test_metrics"))
            grid = srf.construct_grid(d_dir, coords_, feats_,for_training=bool(setup["post_optimize"]),input_construct=setup["evaluate_input"])
            if setup["post_optimize"]:
                grid = srf.post_optimize(d_dir,grid,rays_batch_size=setup["rays_batch_size"],n_iters=setup["post_optim_steps"],randomization=setup["randomized_views"],split="test")
            pred_metrics = srf.evaluate_grid(grid, d_dir)
            if setup["evaluate_robustness"]:
                hard_metrics = srf.evaluate_grid(grid, d_dir,split="hard")
                hard_gt_metrics = get_GT_metrics(os.path.join(d_dir, srf.relative_features_dir, "hard_metrics"))

            if visualize:
                render_dir = os.path.join(setup["render_dir"],str(epoch))
                check_folder(render_dir)
                frames = srf.visualize_grid(grid, d_dir, render_dir, num_views=setup["nb_frames"], vizualization_id=setup["batch_size"]*i + ii, gif=setup["gif"], traj_type=setup["traj_type"])
                if setup["gif"]:
                    wandb.log({"renderings/{}".format(setup["batch_size"]*i + ii): wandb.Video(np.transpose(np.concatenate([fr[None, ...] for fr in frames], axis=0), (
                        0, 3, 1, 2)), fps=2, format="gif"), "epoch": str(epoch)}, commit=False)
                if setup["concat_gt_output"] and not setup["gif"] and setup["visualize_gt"]:
                    concat_render_dir = os.path.join(setup["output_dir"],"comparisons",str(epoch))
                    os.makedirs(concat_render_dir, exist_ok=True)
                    out_vid = os.path.join(render_dir, "{}_renders_{}.mp4".format(setup["traj_type"], str(setup["batch_size"]*i + ii)))
                    gt_vid = os.path.join(gt_render_dir, "{}_renders_{}.mp4".format(setup["traj_type"], str(setup["batch_size"]*i + ii)))
                    concat_vid = os.path.join(concat_render_dir, "{}_renders_{}.mp4".format(setup["traj_type"], str(setup["batch_size"]*i + ii)))
                    concat_horizontal_videos(source_videos_list=[out_vid, gt_vid], output_file=concat_vid)
                if setup["extract_mesh"]:
                    out_mesh_dir = os.path.join(setup["outmesh_dir"], str(epoch))
                    check_folder(out_mesh_dir)
                    out_mesh_render_dir = os.path.join(setup["mesh_render_dir"], str(epoch))
                    check_folder(out_mesh_render_dir)
                    srf.extract_and_visualize_mesh(coords_, feats_, mesh_dir=out_mesh_dir, mesh_render_dir=out_mesh_render_dir, vizualization_id=setup["batch_size"]*i + ii,
                                                smooth=False, density_threshold=setup["density_threshold"], clean=True, render=setup["render_extracted_mesh"], img_res=setup["img_res"])

            val_ssim.append(pred_metrics["SSIM"])
            val_psnr.append(pred_metrics["PSNR"])
            val_lpips.append(pred_metrics["LPIPS"])
            val_acc.append(100.0*pred_metrics["PSNR"]/gt_metrics["PSNR"])
            del coords_ , feats_
            if setup["evaluate_robustness"]:
                hard_ssim.append(hard_metrics["SSIM"])
                hard_psnr.append(hard_metrics["PSNR"])
                hard_lpips.append(hard_metrics["LPIPS"])
                hard_acc.append(100.0*hard_metrics["PSNR"]/hard_gt_metrics["PSNR"])

            # torch.cuda.empty_cache()
        # except:
        #     print(data_dict["labels"])
        del sout , grid
        torch.cuda.empty_cache()

    return {"loss":np.mean(losses) ,"ssim": np.mean(val_ssim) ,"psnr": np.mean(val_psnr),"lpips": np.mean(val_lpips),"acc": np.mean(val_acc),"diou":np.mean(dious)} , {"ssim": np.mean(hard_ssim) ,"psnr": np.mean(hard_psnr),"lpips": np.mean(hard_lpips),"acc": np.mean(hard_acc)}
            

def get_GT_metrics(data_dir):
    with open(os.path.join(data_dir,"psnr.txt"), "r") as f:
        avg_psnr = float(f.read())
    with open(os.path.join(data_dir,"lpips.txt"), "r") as f:
        avg_lpips = float(f.read())
    with open(os.path.join(data_dir,"ssim.txt"), "r") as f:
        avg_ssim = float(f.read())

    return {'PSNR': avg_psnr,'SSIM': avg_ssim , 'LPIPS': avg_lpips}
class RadianceFieldAugmenter(nn.Module):

    def __init__(self,augment_type="none",augment_prop=0.5,vox_res=32,feat_size=512 ,batch_size=1, sh_dim=4):
        super().__init__()
        self.augment_type= augment_type
        self.augment_prop = augment_prop
        self.augmenter = nn.ModuleList()
        if self.augment_type == "none":
            self.augmenter.append(EmptyModule(2))
        elif "rotate" in self.augment_type :
            self.augmenter.append(RadianceFieldRotateAugmenter(vox_res=vox_res,feat_size=feat_size ,batch_size=batch_size,augment_type=augment_type) )
        elif "color" in self.augment_type :
            self.augmenter.append(RadianceFieldColorAugmenter(vox_res=vox_res,feat_size=feat_size ,batch_size=batch_size,augment_type=augment_type,sh_dim=sh_dim))
        
    def forward(self,sparse_tensor_in,sparse_tensor_target):
        with torch.no_grad():
            for ii , aug in enumerate(self.augmenter):
                if torch.rand(1) <= self.augment_prop:
                   sparse_tensor_in,sparse_tensor_target =  aug(sparse_tensor_in,sparse_tensor_target)
            return sparse_tensor_in,sparse_tensor_target



class RadianceFieldRotateAugmenter(nn.Module):

    def __init__(self,vox_res,feat_size ,batch_size,augment_type="axis_rotate"):
        super().__init__()
        self.augment_type = augment_type
        self.vox_res = vox_res     
        self.batch_size = batch_size
        self.interpolation_mode = 'nearest' # [ "bilinear" , "nearest" ]
        self.padding_mode ="reflection" # "zeros"
        self.size_tensor = torch.empty((batch_size,feat_size,self.vox_res,self.vox_res,self.vox_res)).size()
    def forward(self,sparse_tensor_in,sparse_tensor_target):
        dense_tensor_in  = sparse_tensor_in.dense(self.size_tensor)[0].to(sparse_tensor_in.device) 
        dense_tensor_target  =sparse_tensor_target.dense(self.size_tensor)[0].to(sparse_tensor_target.device) 

        for ii in range(self.batch_size):
            angle = torch_random_angles(1,in_degrees=True,full_plane=False)
            axis = (0.0,1.0,0.0) if "axis" in self.augment_type else torch_random_unit_vector(3)
            rot_mat = torch.from_numpy(rotation_matrix(axis, angle, in_degrees=True)).to(torch.float).to(sparse_tensor_target.device)    
            dense_tensor_in[ii]  = rotate3d_tensor(dense_tensor_in[ii][None,...], rot_mat,mode=self.interpolation_mode,padding_mode=self.padding_mode)[0]
            dense_tensor_target[ii] = rotate3d_tensor(dense_tensor_target[ii][None,...], rot_mat,mode=self.interpolation_mode,padding_mode=self.padding_mode)[0]

        sparse_tensor_in = ME.to_sparse(dense_tensor_in.data)
        sparse_tensor_target = ME.to_sparse(dense_tensor_target.data)
        # sparse_tensor_in = ME.to_sparse(rotate3d_tensor(sparse_tensor_in.dense(self.size_tensor)[0], rot_mat))
        # sparse_tensor_target = ME.to_sparse(rotate3d_tensor(sparse_tensor_target.dense(self.size_tensor)[0], rot_mat))

        return sparse_tensor_in,sparse_tensor_target

class RadianceFieldColorAugmenter(nn.Module):

    def __init__(self,vox_res,feat_size ,batch_size,augment_type="random_color",sh_dim=4):
        super().__init__()
        self.augment_type = augment_type
        self.vox_res = vox_res     
        self.batch_size = batch_size
        self.sh_dim = sh_dim
        self.size_tensor = torch.empty((batch_size,feat_size,self.vox_res,self.vox_res,self.vox_res)).size()
    def forward(self,sparse_tensor_in,sparse_tensor_target):
        flip_color = torch_color("random",max_lightness=False).to(sparse_tensor_in.device)
        color_indices = (1, 1+self.sh_dim, 1+2*self.sh_dim,)
        if "random" in self.augment_type:
            sparse_tensor_in.F[:,color_indices] = 0.5* sparse_tensor_in.F[:,color_indices] + 0.5* (flip_color - 0.5)
            sparse_tensor_target.F[:,color_indices] = 0.5* sparse_tensor_target.F[:,color_indices] + 0.5* (flip_color - 0.5)
        elif "primary" in self.augment_type:
            sparse_tensor_in.F[:,color_indices] = torch.cat((extract_primary_color(sparse_tensor_in.F[:,1:1+self.sh_dim],sh_dim=self.sh_dim),extract_primary_color(sparse_tensor_in.F[:,1+self.sh_dim:1+2*self.sh_dim],sh_dim=self.sh_dim),extract_primary_color(sparse_tensor_in.F[:,1+2*self.sh_dim:1+3*self.sh_dim],sh_dim=self.sh_dim)),dim=1)
            sparse_tensor_in.F[:,2:1+self.sh_dim] = sparse_tensor_in.F[:,2+self.sh_dim:1+2*self.sh_dim] = sparse_tensor_in.F[:,2+self.sh_dim:1+3*self.sh_dim] =  0 
            sparse_tensor_target.F[:,(1,1+self.sh_dim,1+2*self.sh_dim,)] = torch.cat((extract_primary_color(sparse_tensor_target.F[:,1:1+self.sh_dim],sh_dim=self.sh_dim),extract_primary_color(sparse_tensor_target.F[:,1+self.sh_dim:1+2*self.sh_dim]),extract_primary_color(sparse_tensor_target.F[:,1+2*self.sh_dim:1+3*self.sh_dim],sh_dim=self.sh_dim)),dim=1)
            sparse_tensor_target.F[:,2:1+self.sh_dim] = sparse_tensor_target.F[:,2+self.sh_dim:1+2*self.sh_dim] = sparse_tensor_target.F[:,2+self.sh_dim:1+3*self.sh_dim] =  0 

        return sparse_tensor_in,sparse_tensor_target

def extract_primary_color(sh_colors,n_dirs=10,pref_dirs=None,sh_dim=4):
    """
    perform simple integration to obtain the primary color out of SH color factors, if pref_dir is given use it instead of integration on all
    """
    bs = sh_colors.shape[0]
    r_dirs  = torch.randn((bs,n_dirs,3))
    r_dirs =  r_dirs/(torch.norm(r_dirs,p=2,dim=-1,keepdim=True) + 1e-5)
    p_color =  torch.mean(torch.sum(eval_sh_bases(sh_dim,r_dirs.to(sh_colors.device)).to(sh_colors.device) * sh_colors[:,None,...].repeat(1,n_dirs,1),dim=2) ,dim=1) 

    return p_color[...,None]

class RadianceFieldCleaner(nn.Module):

    def __init__(self,clean_type="none",lambda_clean=0.0, clean_tau = 0.05 ,clean_annealing_Factor = 1.0):
        super().__init__()
        self.clean_type= clean_type
        self.lambda_clean = lambda_clean
        self.tau = torch.Tensor([clean_tau])
        self.clean_annealing_Factor = clean_annealing_Factor
    def forward(self,sparse_tensor_in,epoch=0):
        if self.clean_type == "none":
            return 0
        c_tau = - self.tau.to(sparse_tensor_in.device) + self.tau.to(sparse_tensor_in.device) *  (self.clean_annealing_Factor ** (epoch+1)) # maybe introduce annealing to the cleaning loss later TODO
        # print(torch.mean(sparse_tensor_in.F[:,0],dim=-1))
        clean_loss = torch.maximum(c_tau, torch.mean(sparse_tensor_in.F[:,0],dim=-1),)
        return self.lambda_clean * clean_loss


def points_to_coordinates(points,batch_size):
    avg_nb = int(points.shape[0] / batch_size)
    batch_vec = (batch_size-1.0) * torch.ones_like(points)[:,0].to(points.device)
    batch_ind = torch.tensor(list(range(batch_size))).to(torch.float).to(points.device)
    batch_vec[0:avg_nb*batch_size] = torch.repeat_interleave(batch_ind, avg_nb)
    return torch.cat((batch_vec[...,None],points),dim=1)


def define_loss(net, sin, starget,data_dict, setup, BCEcrit, MSEcrit, size_tensor,srf=None,c_lambda_2d=0.0):
    device = sin.device
    density_iou, loss, adv_loss, grads = 0.0, 0.0, 0.0 , 0.0
    stargets = [starget]
    # DIFFUSION 
    # sin = sin+srf.batched_diffusion_kernel(sin) if setup["diffusion_type"] != "none" and not net.training else sin # diffusion kernel at test timne 
    souts, fake_encodes = net(sin, data_dict["cam_embed"].to(device), data_dict["imgs"].to(device), data_dict["t_embed"].to(device), skip_main=False)
    if setup["use_lower_res"] and net.network_depth > 1:
        stargets.append(ME.SparseTensor(features=data_dict["l_feats"],coordinates=data_dict["l_coords"],device=device,requires_grad=False) )
    
    # DIFFUSION 
    # if setup["diffusion_type"] != "none" and not net.training:
    #     with torch.no_grad():
    #         for tt in range(1, setup["time_steps"]-1):
    #             t = data_dict["t"] - tt
    #             data_dict["t_embed"] = get_timestep_embedding(t, 1+2*setup["num_encoding_functions"]) if setup["encode_time"] else t
    #             sin_t = srf.batched_forward_diffusion(souts[-1], t)
    #             souts, fake_encodes = net(sin_t+sin, data_dict["cam_embed"].to(device), data_dict["imgs"].to(
    #                 device), data_dict["t_embed"].to(device), skip_main=False)

    loss += define_srf_loss(souts[-1], stargets[0], setup, BCEcrit, MSEcrit, size_tensor, device)
    for ii in range(1, net.network_depth):
        if setup["use_multi_stage_loss"]:
            loss += define_srf_loss(souts[-1-ii], stargets[-1],setup, BCEcrit, MSEcrit, size_tensor, device)
    sparsity = 1.0 - (souts[-1].F.shape[0] / (setup["batch_size"] * float(setup["vox_res"])**3))
    if c_lambda_2d != 0:
        adv_loss += define_perceptual_loss(souts[-1], data_dict, MSEcrit, device, srf, for_training=net.training, setup=setup,quantize=setup["quantize_online_rendering"])

        loss += c_lambda_2d * adv_loss
        adv_loss = adv_loss.item()
    
    ## ADV LOSS 
    # if net.training:
    #     if net.use_adv_loss:
    #         freeze_model(net.srf_encoder,False)
    #         loss += setup["lambda_adv"] * BCEcrit(fake_encodes, torch.ones_like(fake_encodes).to(device))
        # loss.backward()
        # grads = [p.grad.cpu().norm()
        #             for p in net.parameters() if p.grad is not None]
        # # grads = [p.grad.cpu().norm() for p in net.cam_encoder.parameters() if p.grad is not None]
        # # grads = [p.grad.cpu().norm() for p in net.img_encoder.parameters() if p.grad is not None]
        # # grads = [p.grad.cpu().norm() for p in net.time_encoder.parameters() if p.grad is not None]
        # optimizer.step()
        # optimizer.zero_grad()
        # if net.use_adv_loss:
        #     freeze_model(net.srf_encoder, True)
        #     adv_loss += setup["lambda_adv"] *define_adv_loss(net, starget, BCEcrit,label=1.0, device=device)
        #     adv_loss += setup["lambda_adv"] * define_adv_loss(net, souts[-1].detach(),BCEcrit, label=0.0, device=device)
        #     adv_loss.backward()
        #     optimizer.step()
        #     adv_loss = adv_loss.item()

    return loss, adv_loss, souts[-1], density_iou, sparsity #, grads


def define_adv_loss(net, stensor, BCEcrit,label=0.0, device=None):
    adv_loss = 0.0
    _, real_encodes = net(stensor, None, None,None, skip_main=True)
    adv_loss += BCEcrit(real_encodes,torch.full(real_encodes.shape, label).to(device))
    return adv_loss


def define_perceptual_loss(sout, data_dict, MSEcrit, device, srf, for_training=True, setup=None, quantize=False):
    loss_2d = 0.0
    gt_imgs = data_dict["imgs"].to(device) 
    gt_c2ws = data_dict["c2ws"].to(device)
    gt_masks = data_dict["masks"].to(device) * (1.0 - setup["mask_2dbg_alpha"]) + setup["mask_2dbg_alpha"]
     
    c_data_dirs = data_dict["labels"]
    b_coords, b_feats = sout.decomposed_coordinates_and_features
    bs = len(b_coords)
    for jj in range(bs):
        out_img = srf.forward_rendering(c_data_dirs[jj], b_coords[jj], b_feats[jj], gt_c2ws[jj], device=device,for_training=for_training, input_rendering=False, kill_density_grads=setup["kill_density_grads"],quantize=quantize)
        # [imageio.imwrite(os.path.join(setup["output_dir"], "{}_{}.jpg".format(jj, str(kk))), gt_masks[jj][kk][...,None].cpu().detach().numpy()*out_img[kk].cpu().detach().numpy()) for kk in range(out_img.shape[0])]
        # [imageio.imwrite(os.path.join(setup["output_dir"], "GT{}_{}.jpg".format(jj, str(kk))), gt_masks[jj][kk][..., None].cpu().detach().numpy()*gt_imgs[jj].permute(0, 2, 3, 1)[kk].cpu().detach().numpy()) for kk in range(out_img.shape[0])]
        if setup["mask_2dbg_alpha"] == 1:
            loss_2d += MSEcrit(out_img.permute(0, 3, 1, 2), gt_imgs[jj]).mean()
        else:
            loss_2d += MSEcrit((out_img*gt_masks[jj][..., None]).permute(0, 3, 1, 2), gt_imgs[jj]*gt_masks[jj][..., None].permute(0, 3, 1, 2)).mean()
    # raise Exception
    return loss_2d/bs
def define_srf_loss(sout, starget,setup, BCEcrit, MSEcrit, size_tensor, device):
    if setup["mask_type"] not in ["sliced","points","densepoints"]:
        dout = sout.dense(size_tensor)[0]
        if setup["mask_type"] == "nonempty":
            loss_mask = (starget.dense(size_tensor)[0].to(device) != 0)[:,0,...][:,None,...].repeat(1,starget.F.shape[-1],1,1,1) 
        elif setup["mask_type"] == "full":
            loss_mask = torch.ones_like(dout).to(device)
        elif setup["mask_type"] == "dense":
            loss_mask = (starget.dense(size_tensor)[0].to(device) > setup["loss_density_threshold"])[:,0,...][:,None,...].repeat(1,starget.F.shape[-1],1,1,1) 
        elif setup["mask_type"] == "weighted":
            loss_mask = torch.sigmoid(setup["density_mask_tightness"]*(starget.dense(size_tensor)[0].to(device)[:,0,...] - setup["loss_density_threshold"] ))[:,None,...].repeat(1,starget.F.shape[-1],1,1,1) 
            loss_mask = loss_mask * (starget.dense(size_tensor)[0].to(device) != 0)[:,0,...][:,None,...].repeat(1,starget.F.shape[-1],1,1,1) 
        if setup["ignore_loss_mask"]:
            loss_mask[:,0,...] = 0
        loss_cls = BCEcrit(dout[:, 0, ...], (starget.dense(size_tensor)[0][:, 0, ...].to(device) > setup["loss_density_threshold"]).to(torch.float))
        loss_reg = torch.norm((dout+1.0) * (starget.dense(size_tensor)[0].to(device) == 0).to(torch.float), p=1)
        density_adj = 0 if setup["mask_type"] != "full" else (starget.dense(size_tensor)[0].to(device) ==0).to(torch.float).to(device)
        loss_mse = (MSEcrit(dout , starget.dense(size_tensor)[0].to(device) - density_adj)* loss_mask.to(torch.float) ).sum()
        density_iou = 100 * (((starget.dense(size_tensor)[0][:, 0, ...].to(device) > setup["density_threshold"]).to(torch.bool) & (dout[:, 0, ...] > setup["density_threshold"])).to(torch.float).sum() / ((starget.dense(size_tensor)[0][:, 0, ...].to(device) > setup["density_threshold"]).to(torch.bool) | (dout[:, 0, ...] > setup["density_threshold"])).to(torch.float).sum()).item()
    else :
        sliced = sout.features_at_coordinates(starget.C.to(torch.float))
        if setup["mask_type"] == "densepoints":
            d_mask = (starget.F[:, 0] >  setup["loss_density_threshold"]).to(torch.float)[...,None]
            loss_mse = MSEcrit(d_mask*sliced[:,1::], d_mask* starget.F[:,1::]).mean()
        else :
            loss_mse = MSEcrit(sliced[:,1::], starget.F[:,1::]).mean()
        loss_reg = torch.norm(sout.F, p=1)
        if setup["mask_type"] == "sliced":
            dout = sout.dense(size_tensor)[0]
            loss_cls = BCEcrit(dout[:, 0, ...], (starget.dense(size_tensor)[0][:, 0, ...].to(device) > setup["loss_density_threshold"]).to(torch.float)) ##############
            # loss_cls = BCEcrit(sliced[:, 0], (starget.F[:,0] > setup["loss_density_threshold"]).to(torch.float)) #############
            density_iou = 100 * (((starget.dense(size_tensor)[0][:, 0, ...].to(device) > setup["density_threshold"]).to(torch.bool) & (dout[:, 0, ...] > setup["density_threshold"])).to(torch.float).sum() / ((starget.dense(size_tensor)[0][:, 0, ...].to(device) > setup["density_threshold"]).to(torch.bool) | (dout[:, 0, ...] > setup["density_threshold"])).to(torch.float).sum()).item()

        elif setup["mask_type"] in ["points", "densepoints"]:
            if setup["uniform_loss"]:
                points = torch.FloatTensor(int(setup["loss_points_proportion"]*sliced.shape[0]), 3).uniform_(0.0, float(setup["vox_res"]-1)).to(device)
            else:
                vox_limit = 0.5*float(setup["vox_res"]-1)
                points = torch.FloatTensor(int(setup["loss_points_proportion"]*sliced.shape[0]), 3).normal_(mean=vox_limit, std=setup["kernel_std"]*vox_limit).to(device)

            # points = torch.cat((starget.C.to(torch.float)[:, 0][..., None], points), dim=1)
            points = points_to_coordinates(points,batch_size=setup["batch_size"])
            points_feats = sout.features_at_coordinates(points)
            sliced = torch.cat((sliced,points_feats), dim=0)[:,0]
            points_feats_target = starget.features_at_coordinates(points)
            target = (torch.cat((starget.F[:, 0], points_feats_target[:, 0]), dim=0) > setup["loss_density_threshold"]).to(torch.float)
            loss_cls = BCEcrit(sliced, target)
            density_iou = 0  # TODO find a nefficient way to calcualte the DioU since the following line cause a memory isue in the GPU   #############
    loss = setup["lambda_main"] * loss_mse + setup["lambda_reg"] * loss_reg + setup["lambda_cls"] * loss_cls           
    return loss #, density_iou


def freeze_model(model,allow_grads=False):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = allow_grads
        freeze_model(child, allow_grads=allow_grads)


def fully_profile_network(model, input_size=(3, 224, 224), MAX_ITER=10000, verbose=False):
    """
    fully characterize a Pytorch model in terms of speed (ms), number of parameters (M), and, number of operations (GFLOPS)
    """
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False, verbose=verbose)
    inp = torch.rand((1, *input_size)).cuda()
    avg_time = profile_op(MAX_ITER, model.cuda(), inp)
    if verbose:
        print(model, "\n\n\n" "\t", macs, "\t",
              params, "\t", "{}".format(avg_time*1e3))
    return 2 * macs * 1e-9, params*1e-6, avg_time*1e3
