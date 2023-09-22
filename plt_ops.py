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

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from datasets import ShapeNetRend
from Svox2.opt.reflect import get_timestep_embedding


import MinkowskiEngine as ME
from extra_utils import check_folder , merge_two_dicts ,ListDict, save_results , rotate3d_tensor , EmptyModule , torch_random_unit_vector , rotation_matrix ,  torch_random_angles , torch_color , save_trimesh_scene , concat_horizontal_videos
from torch.utils.data.sampler import Sampler
from Svox2.svox2.utils import eval_sh_bases
from ops import make_data_loader, record_metrics, get_GT_metrics , RadianceFieldAugmenter, RadianceFieldCleaner , points_to_coordinates , define_loss , define_adv_loss ,define_perceptual_loss ,define_srf_loss,freeze_model
from pytorch_lightning.core import LightningModule
from pytorch_lightning.callbacks import Callback

M = np.array(
    [
        [0.80656762, -0.5868724, -0.07091862],
        [0.3770505, 0.418344, 0.82632997],
        [-0.45528188, -0.6932309, 0.55870326],
    ]
)


# if not os.path.exists("ModelNet40"):
#     logging.info("Downloading the pruned ModelNet40 dataset...")
#     subprocess.run(["sh", "./examples/download_modelnet40.sh"])


###############################################################################
# Utility functions
###############################################################################


class SRF_PLT(LightningModule):
    r"""
    SRF PLT Modeuke.
    """

    def __init__(
        self,
        model,
        optimizer_name="adamw",
        lr=1e-2,
        weight_decay=1e-5,
        voxel_size=1.0,
        batch_size=12,
        val_batch_size=6,
        train_num_workers=4,
        val_num_workers=2,
        root_dir=None,
        srf=None,
        setup=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model","srf","setup"])
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)
        self.BCEcrit = torch.nn.BCEWithLogitsLoss()
        self.MSEcrit = torch.nn.L1Loss(reduction="none") if "l1" in setup["loss_type"] else torch.nn.MSELoss(reduction="none")
        self.rf_augmenter = RadianceFieldAugmenter(augment_type=setup["augment_type"], augment_prop=setup["augment_prop"], vox_res=setup["vox_res"],
                                          feat_size=setup["in_nchannel"], batch_size=setup["batch_size"], sh_dim=setup["sh_dim"])


    def train_dataloader(self):
        return make_data_loader("train", self.root_dir, self.batch_size, shuffle=True, num_workers=self.train_num_workers, repeat=False, object_class=self.setup["object_class"],
                                augment_data=False, drop_last=True, dset_partition=self.setup["dset_partition"], srf=self.srf, use_lower_res=self.setup["use_lower_res"])

    def val_dataloader(self):
        return make_data_loader("test", self.root_dir, self.val_batch_size, shuffle=False, num_workers=self.val_num_workers, repeat=False, object_class=self.setup["object_class"],
                                augment_data=False, drop_last=True, dset_partition=self.setup["dset_partition"], srf=self.srf, use_lower_res=self.setup["use_lower_res"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data_dict = batch
        i = batch_idx
        c_lambda_2d = self.setup["lambda_2d"] if self.current_epoch >= self.setup["delay_2d_epochs"] else 0.0

        size_tensor = torch.empty((self.batch_size, self.setup["out_nchannel"], self.setup["vox_res"], self.setup["vox_res"], self.setup["vox_res"])).size()

        starget = ME.SparseTensor(features=data_dict["feats"],
                coordinates=data_dict["coords"],
                requires_grad=False
            )
        if not self.setup["train_ae"]:
            sin = ME.SparseTensor(
                features=data_dict["in_feats"],
                coordinates=data_dict["in_coords"],
                requires_grad=False
            )

        else:
            sin = starget
        sin, starget = self.rf_augmenter(sin, starget)
        if self.setup["debug_type"] == "density":
            sin.F[:, 0][sin.F[:, 0] > self.setup["density_threshold"]] = 1.0
            sin.F[:, 0][sin.F[:, 0] < self.setup["density_threshold"]] = - 1.0
            starget.F[:, 0][starget.F[:, 0] >
                            self.setup["density_threshold"]] = 1.0
            starget.F[:, 0][starget.F[:, 0] <
                            self.setup["density_threshold"]] = -1.0
        if self.setup["visualize_input_training"]:
            self.srf.construct_and_visualize_batch_stensor(sin, data_dict["labels"], render_dir=self.setup["render_dir"], vizualization_id=self.setup["batch_size"]*i,
                                                        traj_type=self.setup["traj_type"], input_construct=True, gif=self.setup["gif"], nb_frames=self.setup["nb_frames"])

        loss, adv_loss, sout, density_iou, sparsity = define_loss(self.model, sin, starget, data_dict, self.setup, self.BCEcrit, self.MSEcrit, size_tensor, srf=self.srf,c_lambda_2d=c_lambda_2d)
        grads = [p.grad.cpu().norm() for p in self.model.parameters() if p.grad is not None]       


        # loss.backward()

        # if self.global_step % 10 == 0:
        torch.cuda.empty_cache()
        # self.log("training/loss", loss.item(), on_step=True,on_epoch=True, prog_bar=True, logger=True,batch_size=self.batch_size)
        # self.log('train/grads', np.mean(grads),on_step=True,on_epoch=False, prog_bar=False,batch_size=self.batch_size)
        # self.log('train/diou', density_iou,on_step=True,on_epoch=False, prog_bar=False,batch_size=self.batch_size)
        # self.log('train/sparsity', sparsity,on_step=True,on_epoch=False, prog_bar=False,batch_size=self.batch_size)
        # self.log('train/adv_loss', adv_loss, on_step=True,on_epoch=False, prog_bar=False, batch_size=self.batch_size)
        wandb.log({'train/loss': loss.item()}, step=self.global_step)
        wandb.log({'train/grads': np.mean(grads)}, step = self.global_step)
        wandb.log({'train/diou': density_iou}, step=self.global_step)
        wandb.log({'train/sparsity': sparsity},step=self.global_step)
        wandb.log({'train/adv_loss': adv_loss}, step=self.global_step)

        short_list_cond = batch_idx * self.setup["batch_size"] in range(self.setup["visualizations_nb"])
        if self.setup["validate_training"] and (self.current_epoch+1) % self.setup["val_freq"] == 0 and short_list_cond:
            for ii, d_dir in enumerate(data_dict["labels"]):
                coords_ = sout.coordinates_at(ii)
                feats_ = sout.features_at(ii)

                coords_, feats_ = self.srf.crop_sparse_voxels(coords_, feats_)
                gt_metrics = get_GT_metrics(os.path.join(d_dir, self.srf.relative_features_dir, "test_metrics"))
                grid = self.srf.construct_grid(d_dir, coords_, feats_)
                pred_metrics = self.srf.evaluate_grid(grid, d_dir)
                if ((self.current_epoch+1) % self.setup["visualize_freq"] == 0) and self.setup["visualize_training"]:
                    render_dir = os.path.join(self.setup["render_dir"], "tr_"+str(self.current_epoch))
                    check_folder(render_dir)
                    grid = self.srf.construct_grid(d_dir, coords_, feats_)
                    self.srf.visualize_grid(grid, d_dir, render_dir, num_views=self.setup["nb_frames"], vizualization_id=self.setup["batch_size"]*i + ii,  gif=self.setup["gif"], traj_type=self.setup["traj_type"])
                self.log("training/ssim", pred_metrics["SSIM"], on_step=False, on_epoch=True, prog_bar=True, logger=True,batch_size=self.batch_size )
                self.log("training/lpips", pred_metrics["LPIPS"], on_step=False,on_epoch=True, prog_bar=True, logger=True,batch_size=self.batch_size )
                self.log("training/psnr", pred_metrics["PSNR"], on_step=False,on_epoch=True, prog_bar=True, logger=True,batch_size=self.batch_size )
                self.log("training/acc", 100.0*pred_metrics["PSNR"]/gt_metrics["PSNR"], on_step=False,on_epoch=True, prog_bar=True, logger=True,batch_size=self.batch_size )
                del coords_, feats_, grid
            del sout
            torch.cuda.empty_cache()
        self.log("training/loss", loss.item(), on_step=False,on_epoch=True, prog_bar=True, logger=True,batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        size_tensor = torch.empty((self.val_batch_size, self.setup["out_nchannel"], self.setup["vox_res"], self.setup["vox_res"], self.setup["vox_res"])).size()
        data_dict = batch
        i = batch_idx
        starget = ME.SparseTensor(features=data_dict["feats"],coordinates=data_dict["coords"],requires_grad=False)
        short_list_cond = batch_idx * self.setup["batch_size"] in range(self.setup["visualizations_nb"])
        if self.setup["short_test"] and not short_list_cond:
            return 0

        if not self.setup["train_ae"]:
            sin = ME.SparseTensor(
                features=data_dict["in_feats"],
                coordinates=data_dict["in_coords"],
                requires_grad=False
            )
        else:
            sin = starget

        if self.setup["visualize_gt"]:
            gt_render_dir = os.path.join(
                self.setup["gt_dir"], self.srf.relative_features_dir, "vids")
            os.makedirs(gt_render_dir, exist_ok=True)
            if not os.path.isfile(os.path.join(gt_render_dir, "{}_renders_{}.mp4".format(self.setup["traj_type"], self.setup["batch_size"]*(i+1)-1))):
                self.srf.construct_and_visualize_batch_stensor(starget, data_dict["labels"], render_dir=gt_render_dir, vizualization_id=self.setup["batch_size"]*i,
                                                          traj_type=self.setup["traj_type"], input_construct=False, gif=self.setup["gif"], nb_frames=self.setup["nb_frames"])

        if self.setup["visualize_input"]:
            in_render_dir_vids = os.path.join(
                self.setup["gt_dir"], self.srf.relative_reflect_dir, "vids")
            in_render_dir_imgs = os.path.join(
                self.setup["gt_dir"], self.srf.relative_reflect_dir, "imgs")
            os.makedirs(in_render_dir_vids, exist_ok=True)
            os.makedirs(in_render_dir_imgs, exist_ok=True)
            self.srf.construct_and_visualize_imgs_batch_stensor(sin, data_dict, render_dir_vids=in_render_dir_vids, render_dir_imgs=in_render_dir_imgs, vizualization_id=self.setup["batch_size"]*i,
                                                           traj_type=self.setup["traj_type"], input_construct=True, gif=self.setup["gif"], nb_frames=self.setup["nb_frames"])

        loss, _,  sout, density_iou, _= define_loss(self.model, sin, starget, data_dict, self.setup, self.BCEcrit, self.MSEcrit, size_tensor, srf=self.srf,c_lambda_2d=0.0)
        self.log("val/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.val_batch_size)
        if self.setup["evaluate_input"]:
            sout = sin
        for ii, d_dir in enumerate(data_dict["labels"]):
            coords_ = sout.coordinates_at(ii)
            feats_ = sout.features_at(ii)
            coords_, feats_ = self.srf.crop_sparse_voxels(coords_, feats_)
            gt_metrics = get_GT_metrics(os.path.join(d_dir, self.srf.relative_features_dir, "test_metrics"))
            grid = self.srf.construct_grid(d_dir, coords_, feats_,  for_training=bool(self.setup["post_optimize"]),input_construct=self.setup["evaluate_input"])
            if self.setup["post_optimize"]:
                grid = self.srf.post_optimize(d_dir, grid, rays_batch_size=self.setup["rays_batch_size"], n_iters=self.setup["post_optim_steps"], randomization=self.setup["randomized_views"], split="test")
            pred_metrics = self.srf.evaluate_grid(grid, d_dir)
            if self.setup["evaluate_robustness"]:
                hard_metrics = self.srf.evaluate_grid(grid, d_dir,split="hard")
                hard_gt_metrics = get_GT_metrics(os.path.join(d_dir, self.srf.relative_features_dir, "hard_metrics"))

            if (self.current_epoch+1) % self.setup["visualize_freq"] ==0: 
                render_dir = os.path.join(self.setup["render_dir"], str(self.current_epoch))
                check_folder(render_dir)
                frames = self.srf.visualize_grid(grid, d_dir, render_dir, num_views=self.setup["nb_frames"], vizualization_id=self.setup["batch_size"]*i + ii,  gif=self.setup["gif"], traj_type=self.setup["traj_type"])
                if self.setup["gif"]:
                    wandb.log({"renderings/{}".format(self.setup["batch_size"]*i + ii): wandb.Video(np.transpose(np.concatenate([fr[None, ...] for fr in frames], axis=0), (
                        0, 3, 1, 2)), fps=2, format="gif"), "epoch": str(self.current_epoch)}, commit=False)
                if self.setup["concat_gt_output"] and not self.setup["gif"] and self.setup["visualize_gt"]:
                    concat_render_dir = os.path.join(
                        self.setup["output_dir"], "comparisons", str(self.current_epoch))
                    os.makedirs(concat_render_dir, exist_ok=True)
                    out_vid = os.path.join(render_dir, "{}_renders_{}.mp4".format(
                        self.setup["traj_type"], str(self.setup["batch_size"]*i + ii)))
                    gt_vid = os.path.join(gt_render_dir, "{}_renders_{}.mp4".format(
                        self.setup["traj_type"], str(self.setup["batch_size"]*i + ii)))
                    concat_vid = os.path.join(concat_render_dir, "{}_renders_{}.mp4".format(
                        self.setup["traj_type"], str(self.setup["batch_size"]*i + ii)))
                    concat_horizontal_videos(
                        source_videos_list=[out_vid, gt_vid], output_file=concat_vid)
                if self.setup["extract_mesh"]:
                    out_mesh_dir = os.path.join(
                        self.setup["outmesh_dir"], str(self.current_epoch))
                    check_folder(out_mesh_dir)
                    out_mesh_render_dir = os.path.join(
                        self.setup["mesh_render_dir"], str(self.current_epoch))
                    check_folder(out_mesh_render_dir)
                    self.srf.extract_and_visualize_mesh(coords_, feats_, mesh_dir=out_mesh_dir, mesh_render_dir=out_mesh_render_dir, vizualization_id=self.setup["batch_size"]*i + ii,
                                                   smooth=False, density_threshold=self.setup["density_threshold"], clean=True, render=self.setup["render_extracted_mesh"], img_res=self.setup["img_res"])

            self.log("val/ssim", pred_metrics["SSIM"], on_step=False, on_epoch=True, prog_bar=True, logger=True,batch_size=self.val_batch_size )
            self.log("val/lpips", pred_metrics["LPIPS"], on_step=False, on_epoch=True, prog_bar=True, logger=True,batch_size=self.val_batch_size )
            self.log("val/psnr", pred_metrics["PSNR"], on_step=False, on_epoch=True, prog_bar=True, logger=True,batch_size=self.val_batch_size )
            self.log("val/acc", 100.0*pred_metrics["PSNR"]/gt_metrics["PSNR"],on_step=False, on_epoch=True, prog_bar=True, logger=True,batch_size=self.val_batch_size )
            if self.setup["evaluate_robustness"]:
                self.log("hard/ssim", hard_metrics["SSIM"], on_step=False, on_epoch=True, prog_bar=True, logger=True,batch_size=self.val_batch_size )
                self.log("hard/lpips", hard_metrics["LPIPS"], on_step=False, on_epoch=True, prog_bar=True, logger=True,batch_size=self.val_batch_size )
                self.log("hard/psnr", hard_metrics["PSNR"], on_step=False, on_epoch=True, prog_bar=True, logger=True,batch_size=self.val_batch_size )
                self.log("hard/acc", 100.0*hard_metrics["PSNR"]/hard_gt_metrics["PSNR"],on_step=False, on_epoch=True, prog_bar=True, logger=True,batch_size=self.val_batch_size )
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, betas=(self.setup["momentum"], 0.999), weight_decay=self.weight_decay,)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": optim.lr_scheduler.ExponentialLR(optimizer, self.setup["lr_decay"]),
                    # "monitor": "val/acc",
                    # "frequency": 1
                    # If "monitor" references validation metrics, then "frequency" should be set to a
                    # multiple of "trainer.check_val_every_n_epoch".
                },
        }


class MyPrintingCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print("Training epoch: {}".format(pl_module.current_epoch))

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
