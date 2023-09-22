# Copyright 2021 Alex Yu

# First, install svox2
# Then, python opt.py <path_to>/nerf_synthetic/<scene> -t ckpt/<some_name>
# or use launching script:   sh launch.sh <EXP_NAME> <GPU> <DATA_DIR>

from matplotlib.pyplot import imsave
import torch
import torch.cuda
import torch.optim
import torch.nn.functional as F
import svox2
import imageio
import json
import trimesh
import mcubes

# import imageio
import os
from os import path
# import shutil
import gc
import numpy as np
import sys
import math
from timeit import default_timer as timer

import MinkowskiEngine as ME

# import argparse
# import cv2
sys.path.append(os.path.dirname(__file__))
from util.dataset import datasets
from util.util import Timing, get_expon_lr_func, generate_dirs_equirect, viridis_cmap, compute_ssim , pose_spherical

from util import config_util

# from warnings import warn
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
class SparseRadianceFields():
    def __init__(self, vox_res=512, sh_dim=4, partial_alias="full", reflect_type="preprocessed", nb_views=1, input_sh_dim=1, nb_rf_variants=1, normalize="none", input_quantization_size=1.0, encode_cameras=False, num_encoding_functions=0, prune_input=False, density_threshold=0.0,
     encode_imgs=False, density_normilzation_factor=5_000.0,colors_normilzation_factor = 5.0,encode_time=False,time_steps=1, diffusion_type="none",randomized_views = False,online_nb_views=1,kernel_std=0.333,diffusion_kernel_size=1e5,ignore_input=False,dataset_type="nerf"):
        self.available_vox_res = [32,128,512]
        self.max_views = 400
        self.vox_res = vox_res
        self.sh_dim = sh_dim
        self.dataset_type = dataset_type
        self.time_steps = time_steps
        self.diffusion_type = diffusion_type
        self.diffusion_kernel_size  = diffusion_kernel_size
        self.input_sh_dim = input_sh_dim
        self.normalize = normalize
        self.prune_input = prune_input
        self.density_threshold = density_threshold
        self.nb_views = nb_views
        self.partial_alias = partial_alias
        self.input_quantization_size=input_quantization_size
        self.encode_cameras=encode_cameras
        self.encode_imgs=encode_imgs
        self.encode_time = encode_time
        self.num_encoding_functions = num_encoding_functions
        self.rf_alias = "STF" if self.sh_dim == 1 else "SRF"
        self.input_rf_alias = "STF" # if self.input_sh_dim == 1 else "SRF" # TODO when you need to load SRFs as input, AE has issues now 
        self.nb_rf_variants = nb_rf_variants
        self.rf_variant = 0
        self.randomized_views =  randomized_views
        self.online_nb_views = online_nb_views
        self.kernel_std = kernel_std
        self.ignore_input  = ignore_input

        self.density_normilzation_factor = density_normilzation_factor
        self.colors_normilzation_factor = colors_normilzation_factor

        self.relative_features_dir = os.path.join(self.rf_alias, "vox{}".format(str(self.vox_res)), self.partial_alias)
        

        self.relative_reflect_dir = os.path.join(self.input_rf_alias, "vox{}".format(str(self.vox_res)), "view{}".format(self.nb_views))
        self.reflect_type =  reflect_type
        self.reflection_function = {"preprocessed":self.load_reflect,"fast":self.fast_reflect1,"faster":self.fast_reflect ,"standard":self.obtain_reflection, "volume":self.color_volume_reflect }[self.reflect_type]
        self.normalize_rf = {"const": self.normalize_rf_const, "none": torch.nn.Sequential(), "tanh": self.normalize_rf_tanh, "sigmoid": self.normalize_rf_sigmoid}[self.normalize]
        self.denormalize_rf = {"const": self.denormalize_rf_const, "none": torch.nn.Sequential(), "tanh": self.denormalize_rf_tanh, "sigmoid": self.denormalize_rf_sigmoid}[self.normalize]




    def links_from_coords(self,coords,resolution=(128,128,128)):
        links = torch.sparse_coo_tensor(indices=coords.T, values=torch.arange(coords.shape[0]).to(coords.device),size=resolution).to_dense()
        # llinks = torch.arange(coords.shape[0]).to(coords.device)
        # print("####################\n",torch.unique(coords,dim=0).shape[0],links.min(), links.max(),len(torch.unique(links)))
        return links

    def coords_from_sgrid(self,sparse_grid):
        curr_reso = sparse_grid.links.shape
        dtype = torch.float32
        X = (torch.arange(curr_reso[0], dtype=dtype, device=sparse_grid.links.device) ) 
        Y = (torch.arange(curr_reso[1], dtype=dtype, device=sparse_grid.links.device) )
        Z = (torch.arange(curr_reso[2], dtype=dtype, device=sparse_grid.links.device) )
        X, Y, Z = torch.meshgrid(X, Y, Z)
        points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)

        mask = sparse_grid.links.view(-1) >= 0
        points = points[mask.to(device=sparse_grid.links.device)]

        return points   
    def coords_and_feats_from_grid(self,sparse_grid,):
        feats = torch.cat((sparse_grid.density_data.data,sparse_grid.sh_data.data),dim=1)
        feats = self.normalize_rf(feats)
        coords = self.coords_from_sgrid(sparse_grid)
        return coords , feats

    def prune_sparse_voxels(self,coords, feats,density_thresold=0.0):
        s_indx = torch.nonzero(feats[:, 0] > density_thresold, as_tuple=False)[:,0]
        return coords[s_indx, :], feats[s_indx,:]

    def varcrop_sparse_voxels(self, coords, feats,mincoord=0,maxcoord=128):
        return ME.utils.sparse_quantize(torch.clamp(coords, min=mincoord, max=maxcoord).to(torch.float), feats)
    def crop_sparse_voxels(self, c_coords, c_feats):
        c_coords = torch.clamp(c_coords, min=0, max=self.vox_res-1).to(torch.float)
        device = c_coords.device
        c_coords, c_feats = ME.utils.sparse_quantize(c_coords, c_feats)
        return c_coords.to(device), c_feats.to(device)
    def densify_sparse_voxels(self, coords, feats,factor=2):
        """
        densify sparse voxels by a factor `factor`         
        """
        if factor<=1:
            return coords, feats
        coords = torch.cat([ coords+torch.Tensor([1, 1, 1]), coords+torch.Tensor(
            [1, 1, -1]), coords+torch.Tensor([1, -1, 1]), coords+torch.Tensor([1, -1, -1]), coords+torch.Tensor([-1, 1, 1]), coords+torch.Tensor(
            [-1, 1, -1]), coords+torch.Tensor([-1, -1, 1]), coords+torch.Tensor([-1, -1, -1])], dim=0)
        feats = torch.cat(8*[feats], dim=0)
        coords, feats = self.densify_sparse_voxels(coords, feats, factor=factor-1)
        return self.crop_sparse_voxels(coords, feats)
    def combine_two_srfs(self,coords1, feats1,coords2, feats2):
        return torch.cat((coords1, coords2.to(coords1.device)), dim=0), torch.cat((feats1, feats2.to(feats1.device)), dim=0)

    def enlarge_sparse_voxels(self, coords, feats,factor=2):
        """
        increase voxel resolution by a factor of `factor` for sparse tensor withj coordinates `coords` and features `feats`         
        """
        if factor<=1:
            return coords, feats
        coords = torch.cat([ factor*coords+torch.Tensor([1, 1, 1]), factor*coords+torch.Tensor(
            [1, 1, -1]), factor*coords+torch.Tensor([1, -1, 1]), factor*coords+torch.Tensor([1, -1, -1]), factor*coords+torch.Tensor([-1, 1, 1]), factor*coords+torch.Tensor(
            [-1, 1, -1]), factor*coords+torch.Tensor([-1, -1, 1]), factor*coords+torch.Tensor([-1, -1, -1])], dim=0)
        feats = torch.cat(8*[feats], dim=0)
        return self.crop_sparse_voxels(coords, feats)

    def quantize_sparse_voxels(self, coords,feats, quantization_size=None):
        q_coords, q_feats = ME.utils.sparse_quantize(coords.to(torch.float), feats, quantization_size=quantization_size)
        q_coords = q_coords * quantization_size if quantization_size else q_coords
        return q_coords, q_feats

    def quantizediffuse_sparse_voxels(self, q_coords,q_feats, quantization_size=None,crop_size=0):
        q_coords, q_feats = self.quantize_sparse_voxels(q_coords.to(torch.float), q_feats, quantization_size=quantization_size)
        q_coords, q_feats = self.densify_sparse_voxels(q_coords, q_feats, factor=int(quantization_size))
        q_coords, q_feats = self.varcrop_sparse_voxels(q_coords, q_feats, mincoord=0+crop_size, maxcoord=self.vox_res-crop_size)
        return q_coords, q_feats

    def gaussian_sparse_voxel(self, coords, feats, std=1.0):
        vox_limit = 0.5*float(self.vox_res-1)
        g_coords = torch.normal(torch.ones(coords.shape[0], 3) * vox_limit, torch.ones(coords.shape[0], 3)*std*vox_limit)
        g_feats = std * torch.randn((g_coords.shape[0], feats.shape[1]))
        return g_coords, g_feats
    def add_gaussian_sparse_voxel(self, coords, feats, std=1.0,mix_collision=True,empty=False):
        g_coords, g_feats = self.gaussian_sparse_voxel(coords, feats,  std=std)
        if empty:
            g_feats[:,0] = 0.0
        if mix_collision:
            coords, feats = torch.cat((coords, g_coords.to(coords.device)), dim=0), torch.cat((feats, g_feats.to(feats.device)), dim=0)
            shuf_indx = torch.randperm(coords.shape[0])
            coords, feats = coords[shuf_indx, :], feats[shuf_indx, :]
            return self.crop_sparse_voxels(coords, feats)
        return self.crop_sparse_voxels(torch.cat((coords, g_coords.to(coords.device)), dim=0), torch.cat((feats, g_feats.to(feats.device)), dim=0))
    def batched_diffusion_kernel(self,batch_stensor):  # TODO do proper bathcing of the diffusion operations  to make it fast
        b_coords, b_feats = batch_stensor.decomposed_coordinates_and_features
        bs = len(b_coords)
        out_tensor =[] 
        for jj in range(bs):
            out_tensor.append(self.diffusion_kernel(b_coords[jj], b_feats[jj]))
        b_coords, b_feats = ME.utils.sparse_collate(coords=list(zip(*out_tensor))[0], feats=list(zip(*out_tensor))[1])
        return ME.SparseTensor(features=b_feats, coordinates=b_coords, device=batch_stensor.device, requires_grad=False, coordinate_manager=batch_stensor.coordinate_manager)

    def diffusion_kernel(self, coords, feats, std=0.333):
        T = self.time_steps
        stds = std * (T-1)/T
        if self.diffusion_type=="quantize":
            d_coords, d_feats = coords, feats  # TODO
        elif self.diffusion_type =="crop":
            d_coords, d_feats = coords, feats  # TODO
        elif self.diffusion_type == "squantize":
            d_coords, d_feats = coords, feats  # TODO
        elif self.diffusion_type == "gaussian":
            d_coords, d_feats = self.gaussian_sparse_voxel(coords, feats, std=stds)
        elif self.diffusion_type == "egaussian":
            d_coords, d_feats = self.gaussian_sparse_voxel( coords, feats, std=stds)
            d_feats[:,0] = 0.0
        elif self.diffusion_type == "gaussianmix":
            d_coords, d_feats = self.gaussian_sparse_voxel( coords, feats, std=stds)
        return d_coords, d_feats

    def batched_forward_diffusion(self,batch_stensor, ts): # TODO do proper bathcing of the diffusion operations  to make it fast 
        b_coords, b_feats = batch_stensor.decomposed_coordinates_and_features
        bs = len(b_coords)
        out_tensor =[] 
        for jj in range(bs):
            out_tensor.append(self.forward_diffusion(b_coords[jj], b_feats[jj], t=ts[jj]))
        b_coords, b_feats = ME.utils.sparse_collate(coords=list(zip(*out_tensor))[0], feats=list(zip(*out_tensor))[1])
        return ME.SparseTensor(features=b_feats,coordinates=b_coords,device=batch_stensor.device,requires_grad=False, coordinate_manager=batch_stensor.coordinate_manager)


    def forward_diffusion(self,coords, feats, t=0):
        T = self.time_steps
        if t == 0 :
            return coords, feats
        beta = 0.07 * (1 + t/T) * (20.0/T) * self.vox_res/128.0  # beta schedule
        qs = 1.0 + beta*t
        cs = min(int(t/T * (self.vox_res/1.9)), int(self.vox_res/2.0)-2)
        vs = 0.9 * beta*t
        stds = 0.333 *t/T
        if self.diffusion_type=="none":
            return coords, feats
        elif self.diffusion_type=="quantize":
            d_coords, d_feats = self.quantizediffuse_sparse_voxels(coords, feats, quantization_size=qs,crop_size=cs)
        elif self.diffusion_type =="crop":
            d_coords, d_feats = self.varcrop_sparse_voxels(coords, feats, mincoord=0+cs, maxcoord=128-cs)
        elif self.diffusion_type == "squantize":
            # d_coords, d_feats = self.prune_sparse_voxels(coords, feats) # TODO make pruning works 
            d_coords = coords.to(torch.float) + (vs**0.5) * torch.randn(coords.size(), device=coords.device)
            # feats = feats.to(torch.float) + 0.1*(vs**0.5) * torch.randn(feats.size(), device=feats.device) # TODO make it work
            d_coords, d_feats = self.quantizediffuse_sparse_voxels(d_coords, feats, quantization_size=qs, crop_size=cs)
        elif self.diffusion_type == "gaussian":
            d_coords, d_feats = self.add_gaussian_sparse_voxel(coords, feats, std=stds,mix_collision=False)
            d_coords, d_feats = self.quantize_sparse_voxels(d_coords, d_feats, quantization_size=self.input_quantization_size)
        elif self.diffusion_type == "egaussian":
            d_coords, d_feats = self.add_gaussian_sparse_voxel(coords, feats, std=stds,mix_collision=False,empty=True)
        elif self.diffusion_type == "gaussianmix":
            d_coords, d_feats = self.add_gaussian_sparse_voxel(coords, feats, std=stds,mix_collision=True)
        return d_coords, d_feats

    def save_coords_and_feats(self,saved_file, coords, feats):
        np.savez_compressed(saved_file, coords=coords.cpu().numpy().astype(np.int32), density=feats[:, 0].data.cpu().numpy(), sh_feats=feats[:, 1::].data.cpu().numpy().astype(np.float16))


    def extract_mesh_from_sparse_voxels(self,coords, densities, mesh_name, level_set=0.0, smooth=False, clean=False):
        dense_voxels = torch.sparse_coo_tensor(coords.T.cpu(), densities.cpu(), (self.vox_res, self.vox_res, self.vox_res)).to_dense()
        dense_voxels = dense_voxels.numpy()
        if smooth:
            dense_voxels = mcubes.smooth(dense_voxels)
        vertices, triangles = mcubes.marching_cubes(dense_voxels, level_set)
        mesh = trimesh.Trimesh(vertices, triangles)
        if clean:
            mask_mesh = np.abs(mesh.vertices - mesh.centroid) < 0.3 * mesh.extents
            mesh.update_vertices(mask_mesh.all(axis=1))
            mesh.remove_duplicate_faces()
        mesh.export(mesh_name)
        return mesh
    def load_coords_and_feats(self,saved_file,device="cpu"):
        z = np.load(saved_file)
        feats = torch.cat((torch.from_numpy(z.f.density)[...,None],
                        torch.from_numpy(z.f.sh_feats)), dim=1)
        coords = torch.from_numpy(z.f.coords)
        return coords.to(device), feats.to(device)

    def normalize_rf_const(self, feats):
        feats[:, 0] = feats[:, 0] / self.density_normilzation_factor
        feats[:, 1::] = feats[:, 1::] / self.colors_normilzation_factor
        return feats

    def denormalize_rf_const(self, feats):
        feats[:, 0] = feats[:, 0] * self.density_normilzation_factor
        feats[:, 1::] = feats[:, 1::] * self.colors_normilzation_factor
        return feats

    def normalize_rf_tanh(self, feats):
        return torch.tanh(feats)

    def denormalize_rf_tanh(self, feats):
        return 0.5 * torch.log((1.0+feats + 1e-10)/(1.0-feats + 1e-10))

    def normalize_rf_sigmoid(self, feats):
        return 2.0 * torch.sigmoid(feats) -1.0 

    def denormalize_rf_sigmoid(self, feats):
        return torch.log((1.0+feats + 1e-10)/(1.0-feats + 1e-10))


    def color_volume_reflect(self,data_dir,rays_batch_size=4000,device="cuda:0",n_iters=15 * 12800,randomization=False,split="train"):
        basis_dim = self.sh_dim
        feat_dim = 3*basis_dim+1
        def get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples, inv_scale, near=2, far=6, pad=0, lindisp=False):
            '''
                point_samples [N_rays N_sample 3]
            '''

            N_rays, N_samples = point_samples.shape[:2]
            point_samples = point_samples.reshape(-1, 3)

            # wrap to ref view
            if w2c_ref is not None:
                R = w2c_ref[:3, :3]  # (3, 3)
                T = w2c_ref[:3, 3:]  # (3, 1)
                point_samples = torch.matmul(point_samples, R.t()) + T.reshape(1,3)

            if intrinsic_ref is not None:
                # using projection
                point_samples_pixel =  point_samples @ intrinsic_ref.t()

                point_samples_pixel[:,:2] = (point_samples_pixel[:,:2] / point_samples_pixel[:,-1:] + 0.0) / inv_scale.reshape(1,2)  # normalize to 0~1
                if not lindisp:
                    point_samples_pixel[:,2] = (point_samples_pixel[:,2] - near) / (far - near)  # normalize to 0~1
                else:
                    point_samples_pixel[:,2] = (1.0/point_samples_pixel[:,2]-1.0/near)/(1.0/far - 1.0/near)
            else:
                # using bounding box
                near, far = near.view(1,3), far.view(1,3)
                point_samples_pixel = (point_samples - near) / (far - near)  # normalize to 0~1
            del point_samples

            if pad>0:
                W_feat, H_feat = (inv_scale+1)/4.0
                point_samples_pixel[:,1] = point_samples_pixel[:,1] * H_feat / (H_feat + pad * 2) + pad / (H_feat + pad * 2)
                point_samples_pixel[:,0] = point_samples_pixel[:,0] * W_feat / (W_feat + pad * 2) + pad / (W_feat + pad * 2)

            point_samples_pixel = point_samples_pixel.view(N_rays, N_samples, 3)
            return point_samples_pixel
        def build_color_volume(point_samples, pose_ref, imgs, img_feat=None, downscale=1.0, with_mask=False):
            '''
            point_world: [N_ray N_sample 3]
            imgs: [N V 3 H W]
            '''

            device = imgs.device
            N, V, C, H, W = imgs.shape
            inv_scale = torch.tensor([W - 1, H - 1]).to(device)


            C += with_mask
            C += 0 if img_feat is None else img_feat.shape[2]
            colors = torch.empty((*point_samples.shape[:2], V*C), device=imgs.device, dtype=torch.float)

            # idx = i =  0 # for i,idx in enumerate(range(V)):
            for i,idx in enumerate(range(V)):

                w2c_ref, intrinsic_ref = pose_ref['w2cs'][idx], pose_ref['intrinsics'].clone()  # assume camera 0 is reference
                point_samples_pixel = get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples, inv_scale)[None]
                grid = point_samples_pixel[...,:2]*2.0-1.0

                # img = F.interpolate(imgs[:, idx], scale_factor=downscale, align_corners=True, mode='bilinear',recompute_scale_factor=True) if downscale != 1.0 else imgs[:, idx]
                data = F.grid_sample(imgs[:, idx], grid, align_corners=True, mode='bilinear', padding_mode='border')
                if img_feat is not None:
                    data = torch.cat((data,F.grid_sample(img_feat[:,idx], grid, align_corners=True, mode='bilinear', padding_mode='zeros')),dim=1)

                if with_mask:
                    in_mask = ((grid >-1.0)*(grid < 1.0))
                    in_mask = (in_mask[...,0]*in_mask[...,1]).float()
                    data = torch.cat((data,in_mask.unsqueeze(1)), dim=1)

                colors[...,i*C:i*C+C] = data[0].permute(1, 2, 0)
                del grid, point_samples_pixel, data

            return colors


        def get_ptsvolume(H, W, D, pad, near_far, intrinsic, c2w):
            device = c2w.device
            near,far = near_far

            corners = torch.tensor([[-pad,-pad,1.0],[W+pad,-pad,1.0],[-pad,H+pad,1.0],[W+pad,H+pad,1.0]]).float().to(device)
            corners = torch.matmul(corners, torch.inverse(intrinsic).t())

            linspace_x = torch.linspace(corners[0, 0], corners[1, 0], W+2*pad)
            linspace_y = torch.linspace(corners[ 0, 1], corners[2, 1], H+2*pad)
            ys, xs = torch.meshgrid(linspace_y, linspace_x)  # HW
            near_plane = torch.stack((xs,ys,torch.ones_like(xs)),dim=-1).to(device)*near
            far_plane = torch.stack((xs,ys,torch.ones_like(xs)),dim=-1).to(device)*far

            linspace_z = torch.linspace(1.0, 0.0, D).view(D,1,1,1).to(device)
            pts = linspace_z*near_plane + (1.0-linspace_z)*far_plane
            pts = torch.matmul(pts.view(-1,3), c2w[:3,:3].t()) + c2w[:3,3].view(1,3)

            return pts.view(D*(H+pad*2),W+pad*2,3)
        # args = parser.parse_args()
        args = AttrDict({})
        args.config = os.path.join(data_dir,self.relative_features_dir,"meta.json")

        config_util.maybe_merge_config_file(args,allow_invalid=True)

        density_factor = 0.5
        init_sigma = 0.0
        pad = 10
        near_far = (0.5,3.5)



        factor = 1
        dset = datasets["nerf"](
                    data_dir,
                    split="train",
                    device=device,
                    factor=factor,
                    n_images=self.nb_views,
                    randomization=randomization,
                    data_split = split,
                    verbose=False,
                    **config_util.build_data_options(args)
                    )



        grid_reso=[self.vox_res,self.vox_res,self.vox_res]
        grid_center=dset.scene_center
        grid_radius=dset.scene_radius
        grid_use_sphere_bound=dset.use_sphere_bound
        grid_basis_dim=basis_dim
        grid_use_z_order=True
        grid_basis_reso=self.vox
        grid_mlp_posenc_size=4
        grid_mlp_width=32
        grid_background_nlayers=0
        grid_background_reso=[self.vox_res,self.vox_res,self.vox_res]

        grid = torch.zeros((*grid_reso,feat_dim))

        batch_origins = dset.rays.origins
        batch_dirs = dset.rays.dirs
        rgb_gt = dset.rays.gt
        imgs = dset.gt.permute(0, 3, 1, 2)[None,...]
        indx = 0

        inmatrix = torch.Tensor([[dset.intrins.get('fx', indx),0.0,dset.intrins.get('cx', indx)],[0.0,dset.intrins.get('fy', indx),dset.intrins.get('cy', indx)],[0.0,0.0,1.0]])
        D,H,W = grid.shape[:3]
        c2w = dset.c2w
        w2c =  torch.from_numpy(np.linalg.inv(c2w.numpy()))
        pose_source = {"intrinsics":inmatrix,"c2ws":c2w,"w2cs":w2c}
        intrinsic, c2w = pose_source['intrinsics'], pose_source['c2ws'][0]
        # intrinsic[:2] /= 4
        vox_pts = get_ptsvolume(H-2*pad,W-2*pad,D, pad,  near_far, intrinsic, c2w)

        c_grid = build_color_volume(vox_pts, pose_source, imgs, with_mask=True).view(D,H,W,-1).unsqueeze(0).permute(0, 4, 1, 2, 3)  # [N,D,H,W,C]
        b_mask = c_grid[0].permute(1,2,3,0)[...,-1]
        c_grid = c_grid[0].permute(1,2,3,0)[...,0:-1]

        # for viz_indx in range(32):
        #     imageio.imsave(os.path.join("/home/hamdi/notebooks/learning_torch/data/meshgen/results/01/0027/renderings","relection{}.png".format(viz_indx)),c_grid[:,viz_indx,:,:].numpy())
        #     imageio.imsave(os.path.join("/home/hamdi/notebooks/learning_torch/data/meshgen/results/01/0027/renderings","mask{}.png".format(viz_indx)),b_mask[:,viz_indx,:][...,None].numpy())
        # print("$$$$$$$$$$$$$$$$$$$",c_grid.shape,c_grid.max(),c_grid.min())
        # raise Exception

        grid[:,:,:,0] = b_mask * density_factor
        grid[:, :, :, (1, 1+self.sh_dim, 1+2*self.sh_dim,)] = c_grid - 0.5
        del dset  , c_grid
        sout = grid.to_sparse(3)

        return sout.indices().T.data.to(device), sout.values().data.to(device), dset.c2w, dset.gt
    def load_reflect(self,data_dir,rays_batch_size=4000,device="cuda:0",n_iters=15 * 12800,randomization=False,split="train"):
        c_rf_variant = np.random.randint(self.nb_rf_variants) if split=="train" else 0
        model_path = os.path.join(data_dir, self.relative_reflect_dir, "data_{}.npz".format(c_rf_variant))
        coords, feats = self.load_coords_and_feats(model_path, device=device)
        feats = self.normalize_rf(feats)  
        c2ws, gts, masks = self.load_c2ws_images(data_dir=data_dir, device=device, split="train", c_rf_variant=c_rf_variant, randomized_views=self.randomized_views)

        return coords, feats , c2ws, gts ,masks, c_rf_variant


    def fast_reflect(self,data_dir,rays_batch_size=4000,device="cuda:0",n_iters=15 * 12800,randomization=False,split="train"):
        basis_dim = self.sh_dim
        feat_dim = 3*basis_dim+1



        # args = parser.parse_args()
        args = AttrDict({})
        args.config = os.path.join(data_dir,self.relative_features_dir,"meta.json")

        config_util.maybe_merge_config_file(args,allow_invalid=True)

        density_factor = 1.0
        init_sigma = 0.0



        factor = 1
        dset = datasets["nerf"](
                    data_dir,
                    split="train",
                    device=device,
                    factor=factor,
                    n_images=self.nb_views,
                    randomization=randomization,
                    data_split = split,
                    verbose=False,
                    **config_util.build_data_options(args)
                    )



        grid_reso=[self.vox_res,self.vox_res,self.vox_res]
        grid_center=dset.scene_center
        grid_radius=dset.scene_radius
        grid_use_sphere_bound=dset.use_sphere_bound
        grid_basis_dim=basis_dim
        grid_use_z_order=True
        grid_basis_reso=self.vox
        grid_mlp_posenc_size=4
        grid_mlp_width=32
        grid_background_nlayers=0
        grid_background_reso=[self.vox_res,self.vox_res,self.vox_res]

        grid = torch.zeros((grid_reso[0],grid_reso[1],feat_dim))
        grid[:,:,0] = init_sigma

        batch_origins = dset.rays.origins
        batch_dirs = dset.rays.dirs
        rgb_gt = dset.rays.gt
        # print("$$$$$$$$$$$$$$$$",batch_origins.mean(dim=0),batch_dirs.mean(dim=0),rgb_gt.min(dim=0),rgb_gt.max(dim=0))
        # raise Exception
        
        # grid.requires_grad_(True)


        # gstep_id_base = 0

        # resample_cameras = [
        #         svox2.Camera(c2w.to(device=device),
        #                     dset.intrins.get('fx', i),
        #                     dset.intrins.get('fy', i),
        #                     dset.intrins.get('cx', i),
        #                     dset.intrins.get('cy', i),
        #                     width=dset.get_image_size(i)[1],
        #                     height=dset.get_image_size(i)[0],
        #                     ndc_coeffs=dset.ndc_coeffs) for i, c2w in enumerate(dset.c2w)
        #     ]


        # last_upsamp_step = 0


        # epoch_id = -1
        # while gstep_id_base < n_iters:
        #     dset.shuffle_rays()
        #     epoch_id += 1

        #     def train_step():
        #         # print('Train step')
        #         batch_origins = dset.rays.origins
        #         batch_dirs = dset.rays.dirs
        #         rgb_gt = dset.rays.gt
        #         rays = svox2.Rays(batch_origins, batch_dirs)
        #         rgb_pred = 

        #         return rgb_pred


        # rgb_pred = train_step()
        cube_face = 0 # face_From_cam TODO
        rgb_pred = F.interpolate(dset.gt.permute(0, 3, 1, 2), size=grid_reso[0])[0].permute(1, 2, 0) # [0:nb_views,...]
        b_mask  = (~torch.isclose(rgb_pred.mean(dim=-1) ,torch.ones_like(rgb_pred.mean(dim=-1)),rtol=0.0, atol=1e-4)).to(torch.float)
        # print("$$$$$$$$$$$$4",b_mask.sum())
        densities  = density_factor * b_mask
        # imageio.imsave(os.path.join("/home/hamdi/notebooks/learning_torch/data/meshgen/results/01/0027/renderings","relection.png"),rgb_pred.numpy())
        # imageio.imsave(os.path.join("/home/hamdi/notebooks/learning_torch/data/meshgen/results/01/0027/renderings","mask.png"),b_mask[...,None].numpy())

        grid[:,:,0] = densities
        grid[:,:,(1,1+self.sh_dim,1+2*self.sh_dim,)] = (rgb_pred - 0.5) * b_mask[...,None]
        grid = grid[:,:,None,...].repeat(1,1,grid_reso[0],1)#.permute(0,1,2,3)
        # grid[:,:,0:10,:] = grid[:,:,22:32,:] = 0.0

        # debugging with one voxel in the middle 
        # grid[16,16,0] = density_factor
        # grid[16,16,1::] = torch.Tensor([0.5,-0.5,-0.5])[...,None].repeat(1,9).view(-1)
        # grid = grid[:,:,None,...].repeat(1,1,grid_reso[0],1)#.permute(0,1,2,3)
        # grid[:,:,0:15,:] = grid[:,:,16:32,:] = 0.0
        

        # gstep_id_base += 1

        #  ckpt_path = path.join(train_dir, f'ckpt_{epoch_id:05d}.npz')
        # if save_every > 0 and (epoch_id + 1) % max(
        #         factor, save_every) == 0 and not tune_mode:
        #     print('Saving', ckpt_path)
        #     grid.load(ckpt_path)


        del dset 
        sout = grid.to_sparse(3)

        return sout.indices().T.data.to(device), sout.values().data.to(device), dset.c2w, dset.gt


    def post_optimize(self,data_dir,grid,rays_batch_size=8000,n_iters=15 * 12800,randomization=False,split="train"):
        batch_size = rays_batch_size
        device = grid.density_data.device

        sizes = (int(0.5*self.vox_res),int(0.5*self.vox_res),int(0.5*self.vox_res),self.vox_res,self.vox_res,self.vox_res )
        resolution="[[{}, {}, {}], [{}, {}, {}]]".format(*sizes)
        reso_list = json.loads(resolution)
        reso_id = 0

        # args = parser.parse_args()
        args = AttrDict({})
        res_alias  = str(reso_list[-1][0])# if reso_list[-1][0] != 512 else ""
        args.config = os.path.join(data_dir,self.relative_features_dir,"meta.json")

        config_util.maybe_merge_config_file(args,allow_invalid=True)


        
        upsamp_every= int(0.3 * n_iters)
        lr_sigma =2e-1
        lr_sigma_final =5e-2
        lr_sigma_decay_steps =250000
        lr_sigma_delay_steps =15000
        lr_sigma_delay_mult =1e-2
        sh_optim = 'rmsprop'
        density_thresh = 5.0
        lr_sh = 5e-3
        lr_sh_final =5e-6
        lr_sh_decay_steps =250000
        lr_sh_delay_steps =0
        lr_sh_delay_mult =1e-2
        lr_basis=1e-6
        lr_basis_final=1e-6
        lr_basis_decay_steps=250000
        lr_basis_delay_steps=0
        lr_basis_begin_step=0
        lr_basis_delay_mult=1e-2
        lr_fg_begin_step= 0
        init_sigma = 0.1
        lambda_tv=0.0
        tv_sparsity=1.0
        tv_logalpha = False
        lambda_tv_sh=0.0
        tv_sh_sparsity=1.0
        rms_beta = 0.95
        lambda_tv_lumisphere=0.0
        tv_lumisphere_sparsity=0.01
        tv_lumisphere_dir_factor=0.0
        tv_decay=1.0
        lambda_l2_sh = 0.0
        tv_early_only = 1
        weight_thresh =0.0005 * 512
        max_grid_elements=44_000_000




        factor = 1
        dset = datasets["nerf"](
                    data_dir,
                    split="train",
                    device=device,
                    factor=factor,
                    n_images=self.nb_views,
                    randomization=randomization,
                    data_split = split,
                    verbose=False,
                    **config_util.build_data_options(args)
                    )


        optim_basis_mlp = None


        grid.requires_grad_(True)
        config_util.setup_render_opts(grid.opt, args)
        # print('Render options', grid.opt)

        gstep_id_base = 0

        resample_cameras = [
                svox2.Camera(c2w.to(device=device),
                            dset.intrins.get('fx', i),
                            dset.intrins.get('fy', i),
                            dset.intrins.get('cx', i),
                            dset.intrins.get('cy', i),
                            width=dset.get_image_size(i)[1],
                            height=dset.get_image_size(i)[0],
                            ndc_coeffs=dset.ndc_coeffs) for i, c2w in enumerate(dset.c2w)
            ]
        # ckpt_path = path.join(train_dir, 'ckpt.npz')

        lr_sigma_func = get_expon_lr_func(lr_sigma, lr_sigma_final, lr_sigma_delay_steps,
                                        lr_sigma_delay_mult, lr_sigma_decay_steps)
        lr_sh_func = get_expon_lr_func(lr_sh, lr_sh_final, lr_sh_delay_steps,
                                    lr_sh_delay_mult, lr_sh_decay_steps)
        lr_basis_func = get_expon_lr_func(lr_basis, lr_basis_final, lr_basis_delay_steps,
                                    lr_basis_delay_mult, lr_basis_decay_steps)

        lr_sigma_factor = 1.0
        lr_sh_factor = 1.0
        lr_basis_factor = 1.0

        last_upsamp_step = 0

        epoch_id = -1
        while gstep_id_base < n_iters:
            dset.shuffle_rays()
            epoch_id += 1
            epoch_size = dset.rays.origins.size(0)
            batches_per_epoch = (epoch_size-1)//batch_size+1


            def train_step():
                # print('Train step')
                pbar = enumerate(range(0, epoch_size, batch_size))
                stats = {"mse" : 0.0, "psnr" : 0.0, "invsqr_mse" : 0.0}
                for iter_id, batch_begin in pbar:
                    gstep_id = iter_id + gstep_id_base
                    if lr_fg_begin_step > 0 and gstep_id == lr_fg_begin_step:
                        grid.density_data.data[:] = init_sigma
                    lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
                    lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
                    lr_basis = lr_basis_func(gstep_id - lr_basis_begin_step) * lr_basis_factor
                    # lr_sigma_bg = lr_sigma_bg_func(gstep_id - lr_basis_begin_step) * lr_basis_factor
                    # lr_color_bg = lr_color_bg_func(gstep_id - lr_basis_begin_step) * lr_basis_factor
                    # if not lr_decay:
                    lr_sigma = lr_sigma * lr_sigma_factor
                    lr_sh = lr_sh * lr_sh_factor
                    lr_basis = lr_basis * lr_basis_factor

                    batch_end = min(batch_begin + batch_size, epoch_size)
                    batch_origins = dset.rays.origins[batch_begin: batch_end]
                    batch_dirs = dset.rays.dirs[batch_begin: batch_end]
                    rgb_gt = dset.rays.gt[batch_begin: batch_end]
                    rays = svox2.Rays(batch_origins, batch_dirs)

                    #  with Timing("volrend_fused"):
                    rgb_pred = grid.volume_render_fused(rays, rgb_gt,
                            beta_loss=0.0,
                            sparsity_loss=0.0,
                            randomize=False)

                    #  with Timing("loss_comp"):
                    mse = F.mse_loss(rgb_gt, rgb_pred)

                    # Stats
                    mse_num : float = mse.detach().item()
                    psnr = -10.0 * math.log10(mse_num)
                    stats['mse'] += mse_num
                    stats['psnr'] += psnr
                    stats['invsqr_mse'] += 1.0 / mse_num ** 2

                    if lambda_tv > 0.0:
                        #  with Timing("tv_inpl"):
                        grid.inplace_tv_grad(grid.density_data.grad,
                                scaling=lambda_tv,
                                sparse_frac=tv_sparsity,
                                logalpha=tv_logalpha,
                                ndc_coeffs=dset.ndc_coeffs,
                                contiguous=1)
                    if lambda_tv_sh > 0.0:
                        #  with Timing("tv_color_inpl"):
                        grid.inplace_tv_color_grad(grid.sh_data.grad,
                                scaling=lambda_tv_sh,
                                sparse_frac=tv_sh_sparsity,
                                ndc_coeffs=dset.ndc_coeffs,
                                contiguous=1)
                    if lambda_tv_lumisphere > 0.0:
                        grid.inplace_tv_lumisphere_grad(grid.sh_data.grad,
                                scaling=lambda_tv_lumisphere,
                                dir_factor=tv_lumisphere_dir_factor,
                                sparse_frac=tv_lumisphere_sparsity,
                                ndc_coeffs=dset.ndc_coeffs)
                    if lambda_l2_sh > 0.0:
                        grid.inplace_l2_color_grad(grid.sh_data.grad,
                                scaling=lambda_l2_sh)

                    # Manual SGD/rmsprop step
                    if gstep_id >= lr_fg_begin_step:
                        grid.optim_density_step(lr_sigma, beta=rms_beta, optim='rmsprop')
                        grid.optim_sh_step(lr_sh, beta=rms_beta, optim=sh_optim)
                    # if grid.use_background:
                    #     grid.optim_background_step(lr_sigma_bg, lr_color_bg, beta=rms_beta, optim=bg_optim)
                    # if gstep_id >= lr_basis_begin_step:
                        if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                            grid.optim_basis_step(lr_basis, beta=rms_beta, optim='rmsprop')
                        elif grid.basis_type == svox2.BASIS_TYPE_MLP:
                            optim_basis_mlp.step()
                            optim_basis_mlp.zero_grad()

            train_step()
            gc.collect()
            gstep_id_base += batches_per_epoch


            if (gstep_id_base - last_upsamp_step) >= upsamp_every:
                last_upsamp_step = gstep_id_base
                if reso_id < len(reso_list) - 1:
                    # print('* Upsampling from', [self.vox_res,self.vox_res,self.vox_res], 'to', reso_list[reso_id + 1])
                    if tv_early_only > 0:
                        # print('turning off TV regularization')
                        lambda_tv = 0.0
                        lambda_tv_sh = 0.0
                    elif tv_decay != 1.0:
                        lambda_tv *= tv_decay
                        lambda_tv_sh *= tv_decay

                    reso_id += 1
                    use_sparsify = False
                    z_reso = [self.vox_res,self.vox_res,self.vox_res] if isinstance([self.vox_res,self.vox_res,self.vox_res], int) else [self.vox_res,self.vox_res,self.vox_res][2]
                    grid.resample(reso=[self.vox_res,self.vox_res,self.vox_res],
                            sigma_thresh=density_thresh,
                            weight_thresh=weight_thresh / z_reso if use_sparsify else 0.0,
                            dilate=2, #use_sparsify,
                            cameras=resample_cameras ,
                            max_elements=max_grid_elements)

                if factor > 1 and reso_id < len(reso_list) - 1:
                    print('* Using higher resolution images due to large grid; new factor', factor)
                    factor //= 2
                    dset.gen_rays(factor=factor)
                    dset.shuffle_rays()


        del dset 
        return grid 

    def fast_reflect1(self,data_dir,rays_batch_size=8000,device="cuda:0",n_iters=15 * 12800,randomization=False,split="train"):
        batch_size = rays_batch_size
        sizes = (int(0.5*self.vox_res),int(0.5*self.vox_res),int(0.5*self.vox_res),self.vox_res,self.vox_res,self.vox_res )
        resolution="[[{}, {}, {}], [{}, {}, {}]]".format(*sizes)
        reso_list = json.loads(resolution)
        reso_id = 0
        basis_dim = self.sh_dim

        # args = parser.parse_args()
        args = AttrDict({})
        res_alias  = str(reso_list[-1][0]) #if reso_list[-1][0] != 512 else ""
        args.config = os.path.join(data_dir,self.relative_features_dir,"meta.json")

        config_util.maybe_merge_config_file(args,allow_invalid=True)


        
        upsamp_every= int(0.3 * n_iters)
        lr_sigma =3e1
        lr_sigma_final =5e-2
        lr_sigma_decay_steps =250000
        lr_sigma_delay_steps =15000
        lr_sigma_delay_mult =1e-2
        sh_optim = 'rmsprop'
        density_thresh = 5.0
        lr_sh = 1e-2
        lr_sh_final =5e-6
        lr_sh_decay_steps =250000
        lr_sh_delay_steps =0
        lr_sh_delay_mult =1e-2
        lr_basis=1e-6
        lr_basis_final=1e-6
        lr_basis_decay_steps=250000
        lr_basis_delay_steps=0
        lr_basis_begin_step=0
        lr_basis_delay_mult=1e-2
        lr_fg_begin_step= 0
        init_sigma = 0.1
        lambda_tv=0.0
        tv_sparsity=1.0
        tv_logalpha = False
        lambda_tv_sh=0.0
        tv_sh_sparsity=1.0
        rms_beta = 0.95
        lambda_tv_lumisphere=0.0
        tv_lumisphere_sparsity=0.01
        tv_lumisphere_dir_factor=0.0
        tv_decay=1.0
        lambda_l2_sh = 0.0
        tv_early_only = 1
        weight_thresh =0.0005 * 512
        max_grid_elements=44_000_000




        factor = 1
        dset = datasets["nerf"](
                    data_dir,
                    split="train",
                    device=device,
                    factor=factor,
                    n_images=self.nb_views,
                    randomization=randomization,
                    data_split = split,
                    verbose=False,
                    **config_util.build_data_options(args)
                    )


        global_start_time = datetime.now()

        grid = svox2.SparseGrid(reso=[self.vox_res,self.vox_res,self.vox_res],
                                center=dset.scene_center,
                                radius=dset.scene_radius,
                                use_sphere_bound=dset.use_sphere_bound,
                                basis_dim=basis_dim,
                                use_z_order=True,
                                device=device,
                                basis_reso=self.vox,
                                basis_type=svox2.__dict__['BASIS_TYPE_' + "sh".upper()],
                                mlp_posenc_size=4,
                                mlp_width=32,
                                background_nlayers=0,
                                background_reso=[self.vox_res,self.vox_res,self.vox_res],
                            )

        grid.sh_data.data[:] = 0.0
        grid.density_data.data[:] = init_sigma


        optim_basis_mlp = None

        if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
            grid.reinit_learned_bases(init_type='sh')
            #  grid.reinit_learned_bases(init_type='fourier')
            #  grid.reinit_learned_bases(init_type='sg', upper_hemi=True)
            #  grid.basis_data.data.normal_(mean=0.28209479177387814, std=0.001)

        elif grid.basis_type == svox2.BASIS_TYPE_MLP:
            # MLP!
            optim_basis_mlp = torch.optim.Adam(
                            grid.basis_mlp.parameters(),
                            lr=lr_basis
                        )


        grid.requires_grad_(True)
        config_util.setup_render_opts(grid.opt, args)
        # print('Render options', grid.opt)

        gstep_id_base = 0

        resample_cameras = [
                svox2.Camera(c2w.to(device=device),
                            dset.intrins.get('fx', i),
                            dset.intrins.get('fy', i),
                            dset.intrins.get('cx', i),
                            dset.intrins.get('cy', i),
                            width=dset.get_image_size(i)[1],
                            height=dset.get_image_size(i)[0],
                            ndc_coeffs=dset.ndc_coeffs) for i, c2w in enumerate(dset.c2w)
            ]
        # ckpt_path = path.join(train_dir, 'ckpt.npz')

        lr_sigma_func = get_expon_lr_func(lr_sigma, lr_sigma_final, lr_sigma_delay_steps,
                                        lr_sigma_delay_mult, lr_sigma_decay_steps)
        lr_sh_func = get_expon_lr_func(lr_sh, lr_sh_final, lr_sh_delay_steps,
                                    lr_sh_delay_mult, lr_sh_decay_steps)
        lr_basis_func = get_expon_lr_func(lr_basis, lr_basis_final, lr_basis_delay_steps,
                                    lr_basis_delay_mult, lr_basis_decay_steps)

        lr_sigma_factor = 1.0
        lr_sh_factor = 1.0
        lr_basis_factor = 1.0

        last_upsamp_step = 0

        epoch_id = -1
        while gstep_id_base < n_iters:
            dset.shuffle_rays()
            epoch_id += 1
            epoch_size = dset.rays.origins.size(0)
            batches_per_epoch = (epoch_size-1)//batch_size+1


            def train_step():
                # print('Train step')
                pbar = enumerate(range(0, epoch_size, batch_size))
                stats = {"mse" : 0.0, "psnr" : 0.0, "invsqr_mse" : 0.0}
                for iter_id, batch_begin in pbar:
                    gstep_id = iter_id + gstep_id_base
                    if lr_fg_begin_step > 0 and gstep_id == lr_fg_begin_step:
                        grid.density_data.data[:] = init_sigma
                    lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
                    lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
                    lr_basis = lr_basis_func(gstep_id - lr_basis_begin_step) * lr_basis_factor
                    # lr_sigma_bg = lr_sigma_bg_func(gstep_id - lr_basis_begin_step) * lr_basis_factor
                    # lr_color_bg = lr_color_bg_func(gstep_id - lr_basis_begin_step) * lr_basis_factor
                    # if not lr_decay:
                    lr_sigma = lr_sigma * lr_sigma_factor
                    lr_sh = lr_sh * lr_sh_factor
                    lr_basis = lr_basis * lr_basis_factor

                    batch_end = min(batch_begin + batch_size, epoch_size)
                    batch_origins = dset.rays.origins[batch_begin: batch_end]
                    batch_dirs = dset.rays.dirs[batch_begin: batch_end]
                    rgb_gt = dset.rays.gt[batch_begin: batch_end]
                    rays = svox2.Rays(batch_origins, batch_dirs)

                    #  with Timing("volrend_fused"):
                    rgb_pred = grid.volume_render_fused(rays, rgb_gt,
                            beta_loss=0.0,
                            sparsity_loss=0.0,
                            randomize=False)

                    #  with Timing("loss_comp"):
                    mse = F.mse_loss(rgb_gt, rgb_pred)

                    # Stats
                    mse_num : float = mse.detach().item()
                    psnr = -10.0 * math.log10(mse_num)
                    stats['mse'] += mse_num
                    stats['psnr'] += psnr
                    stats['invsqr_mse'] += 1.0 / mse_num ** 2

                    if lambda_tv > 0.0:
                        #  with Timing("tv_inpl"):
                        grid.inplace_tv_grad(grid.density_data.grad,
                                scaling=lambda_tv,
                                sparse_frac=tv_sparsity,
                                logalpha=tv_logalpha,
                                ndc_coeffs=dset.ndc_coeffs,
                                contiguous=1)
                    if lambda_tv_sh > 0.0:
                        #  with Timing("tv_color_inpl"):
                        grid.inplace_tv_color_grad(grid.sh_data.grad,
                                scaling=lambda_tv_sh,
                                sparse_frac=tv_sh_sparsity,
                                ndc_coeffs=dset.ndc_coeffs,
                                contiguous=1)
                    if lambda_tv_lumisphere > 0.0:
                        grid.inplace_tv_lumisphere_grad(grid.sh_data.grad,
                                scaling=lambda_tv_lumisphere,
                                dir_factor=tv_lumisphere_dir_factor,
                                sparse_frac=tv_lumisphere_sparsity,
                                ndc_coeffs=dset.ndc_coeffs)
                    if lambda_l2_sh > 0.0:
                        grid.inplace_l2_color_grad(grid.sh_data.grad,
                                scaling=lambda_l2_sh)

                    # Manual SGD/rmsprop step
                    if gstep_id >= lr_fg_begin_step:
                        grid.optim_density_step(lr_sigma, beta=rms_beta, optim='rmsprop')
                        grid.optim_sh_step(lr_sh, beta=rms_beta, optim=sh_optim)
                    # if grid.use_background:
                    #     grid.optim_background_step(lr_sigma_bg, lr_color_bg, beta=rms_beta, optim=bg_optim)
                    # if gstep_id >= lr_basis_begin_step:
                        if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                            grid.optim_basis_step(lr_basis, beta=rms_beta, optim='rmsprop')
                        elif grid.basis_type == svox2.BASIS_TYPE_MLP:
                            optim_basis_mlp.step()
                            optim_basis_mlp.zero_grad()

            train_step()
            gc.collect()
            gstep_id_base += batches_per_epoch


            if (gstep_id_base - last_upsamp_step) >= upsamp_every:
                last_upsamp_step = gstep_id_base
                if reso_id < len(reso_list) - 1:
                    # print('* Upsampling from', [self.vox_res,self.vox_res,self.vox_res], 'to', reso_list[reso_id + 1])
                    if tv_early_only > 0:
                        # print('turning off TV regularization')
                        lambda_tv = 0.0
                        lambda_tv_sh = 0.0
                    elif tv_decay != 1.0:
                        lambda_tv *= tv_decay
                        lambda_tv_sh *= tv_decay

                    reso_id += 1
                    use_sparsify = False
                    z_reso = [self.vox_res,self.vox_res,self.vox_res] if isinstance([self.vox_res,self.vox_res,self.vox_res], int) else [self.vox_res,self.vox_res,self.vox_res][2]
                    grid.resample(reso=[self.vox_res,self.vox_res,self.vox_res],
                            sigma_thresh=density_thresh,
                            weight_thresh=weight_thresh / z_reso if use_sparsify else 0.0,
                            dilate=2, #use_sparsify,
                            cameras=resample_cameras ,
                            max_elements=max_grid_elements)

                if factor > 1 and reso_id < len(reso_list) - 1:
                    print('* Using higher resolution images due to large grid; new factor', factor)
                    factor //= 2
                    dset.gen_rays(factor=factor)
                    dset.shuffle_rays()


        del dset 
        in_ccords_ , in_feat_ =  self.coords_and_feats_from_grid(grid)
        return in_ccords_, in_feat_, dset.c2w, dset.gt
    def obtain_reflection(self,data_dir,rays_batch_size=4000,device="cuda:0",n_iters=15 * 12800,randomization=False,split="train"):
        batch_size = rays_batch_size
        sizes = (int(0.5*self.vox_res),int(0.5*self.vox_res),int(0.5*self.vox_res),self.vox_res,self.vox_res,self.vox_res )
        resolution="[[{}, {}, {}], [{}, {}, {}]]".format(*sizes)
        reso_list = json.loads(resolution)
        reso_id = 0
        basis_dim = self.sh_dim

        # args = parser.parse_args()
        args = AttrDict({})
        res_alias  = str(reso_list[-1][0])# if reso_list[-1][0] != 512 else ""
        res_alias = reso_list[-1][0]

        args.config = os.path.join(data_dir,self.relative_features_dir,"meta.json")

        config_util.maybe_merge_config_file(args,allow_invalid=True)

        # assert lr_sigma_final <= lr_sigma, "lr_sigma must be >= lr_sigma_final"
        # assert lr_sh_final <= lr_sh, "lr_sh must be >= lr_sh_final"
        # assert lr_basis_final <= lr_basis, "lr_basis must be >= lr_basis_final"

        # os.makedirs(train_dir, exist_ok=True)
        # summary_writer = SummaryWriter(train_dir)



        # with open(path.join(train_dir, 'json'), 'w') as f:
        #     json.dump(__dict__, f, indent=2)
        #     # Changed name to prevent errors
        #     shutil.copyfile(__file__, path.join(train_dir, 'opt_frozen.py'))
        
        
        upsamp_every= int(0.3 * n_iters)
        lr_sigma =3e1
        lr_sigma_final =5e-2
        lr_sigma_decay_steps =250000
        lr_sigma_delay_steps =15000
        lr_sigma_delay_mult =1e-2
        sh_optim = 'rmsprop'
        density_thresh = 5.0
        lr_sh = 1e-2
        lr_sh_final =5e-6
        lr_sh_decay_steps =250000
        lr_sh_delay_steps =0
        lr_sh_delay_mult =1e-2
        lr_basis=1e-6
        lr_basis_final=1e-6
        lr_basis_decay_steps=250000
        lr_basis_delay_steps=0
        lr_basis_begin_step=0
        lr_basis_delay_mult=1e-2
        lr_fg_begin_step= 0
        init_sigma = 0.1
        lambda_tv=1e-5
        tv_sparsity=0.01
        tv_logalpha = False
        lambda_tv_sh=1e-3
        tv_sh_sparsity=0.01
        rms_beta = 0.95
        lambda_tv_lumisphere=0.0
        tv_lumisphere_sparsity=0.01
        tv_lumisphere_dir_factor=0.0
        tv_decay=1.0
        lambda_l2_sh = 0.0
        tv_early_only = 1
        weight_thresh =0.0005 * 512
        max_grid_elements=44_000_000




        factor = 1
        dset = datasets["nerf"](
                    data_dir,
                    split="train",
                    device=device,
                    factor=factor,
                    n_images=self.nb_views,
                    randomization=randomization,
                    data_split = split,
                    verbose=False,
                    **config_util.build_data_options(args)
                    )

        # if background_nlayers > 0 and not dset.should_use_background:
        #     warn('Using a background model for dataset type ' + str(type(dset)) + ' which typically does not use background')

        # dset_test = datasets["nerf"](data_dir, split="test", **config_util.build_data_options(args))

        global_start_time = datetime.now()

        grid = svox2.SparseGrid(reso=[self.vox_res,self.vox_res,self.vox_res],
                                center=dset.scene_center,
                                radius=dset.scene_radius,
                                use_sphere_bound=dset.use_sphere_bound,
                                basis_dim=basis_dim,
                                use_z_order=True,
                                device=device,
                                basis_reso=self.vox,
                                basis_type=svox2.__dict__['BASIS_TYPE_' + "sh".upper()],
                                mlp_posenc_size=4,
                                mlp_width=32,
                                background_nlayers=0,
                                background_reso=[self.vox_res,self.vox_res,self.vox_res],
                            )

        # print(torch.unique(grid.links).shape,torch.unique(grid.links))
        # raise Exception("STOP HERE ")
        # DC -> gray; mind the SH scaling!
        grid.sh_data.data[:] = 0.0
        grid.density_data.data[:] = init_sigma

        # if grid.use_background:
        #     grid.background_data.data[..., -1] = init_sigma_bg
            #  grid.background_data.data[..., :-1] = 0.5 / svox2.utils.SH_C0

        #  grid.sh_data.data[:, 0] = 4.0
        #  osh = grid.density_data.data.shape
        #  den = grid.density_data.data.view(grid.links.shape)
        #  #  den[:] = 0.00
        #  #  den[:, :256, :] = 1e9
        #  #  den[:, :, 0] = 1e9
        #  grid.density_data.data = den.view(osh)

        optim_basis_mlp = None

        if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
            grid.reinit_learned_bases(init_type='sh')
            #  grid.reinit_learned_bases(init_type='fourier')
            #  grid.reinit_learned_bases(init_type='sg', upper_hemi=True)
            #  grid.basis_data.data.normal_(mean=0.28209479177387814, std=0.001)

        elif grid.basis_type == svox2.BASIS_TYPE_MLP:
            # MLP!
            optim_basis_mlp = torch.optim.Adam(
                            grid.basis_mlp.parameters(),
                            lr=lr_basis
                        )


        grid.requires_grad_(True)
        config_util.setup_render_opts(grid.opt, args)
        # print('Render options', grid.opt)

        gstep_id_base = 0

        resample_cameras = [
                svox2.Camera(c2w.to(device=device),
                            dset.intrins.get('fx', i),
                            dset.intrins.get('fy', i),
                            dset.intrins.get('cx', i),
                            dset.intrins.get('cy', i),
                            width=dset.get_image_size(i)[1],
                            height=dset.get_image_size(i)[0],
                            ndc_coeffs=dset.ndc_coeffs) for i, c2w in enumerate(dset.c2w)
            ]
        # ckpt_path = path.join(train_dir, 'ckpt.npz')

        lr_sigma_func = get_expon_lr_func(lr_sigma, lr_sigma_final, lr_sigma_delay_steps,
                                        lr_sigma_delay_mult, lr_sigma_decay_steps)
        lr_sh_func = get_expon_lr_func(lr_sh, lr_sh_final, lr_sh_delay_steps,
                                    lr_sh_delay_mult, lr_sh_decay_steps)
        lr_basis_func = get_expon_lr_func(lr_basis, lr_basis_final, lr_basis_delay_steps,
                                    lr_basis_delay_mult, lr_basis_decay_steps)
        # lr_sigma_bg_func = get_expon_lr_func(lr_sigma_bg, lr_sigma_bg_final, lr_sigma_bg_delay_steps,
        #                             lr_sigma_bg_delay_mult, lr_sigma_bg_decay_steps)
        # lr_color_bg_func = get_expon_lr_func(lr_color_bg, lr_color_bg_final, lr_color_bg_delay_steps,
        #                             lr_color_bg_delay_mult, lr_color_bg_decay_steps)
        lr_sigma_factor = 1.0
        lr_sh_factor = 1.0
        lr_basis_factor = 1.0

        last_upsamp_step = 0



        # if enable_random:
        #     warn("Randomness is enabled for training (normal for LLFF & scenes with background)")

        epoch_id = -1
        while gstep_id_base < n_iters:
            dset.shuffle_rays()
            epoch_id += 1
            epoch_size = dset.rays.origins.size(0)
            batches_per_epoch = (epoch_size-1)//batch_size+1
            # Test
            # def eval_step():
            #     # Put in a function to avoid memory leak
            #     print('Eval step')
            #     with torch.no_grad():
            #         stats_test = {'psnr' : 0.0, 'mse' : 0.0}

            #         # Standard set
            #         N_IMGS_TO_EVAL = min(20 if epoch_id > 0 else 5, dset_test.n_images)
            #         N_IMGS_TO_SAVE = N_IMGS_TO_EVAL # if not tune_mode else 1
            #         img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
            #         img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE)
            #         img_ids = range(0, dset_test.n_images, img_eval_interval)

            #         # Special 'very hard' specular + fuzz set
            #         #  img_ids = [2, 5, 7, 9, 21,
            #         #             44, 45, 47, 49, 56,
            #         #             80, 88, 99, 115, 120,
            #         #             154]
            #         #  img_save_interval = 1

            #         n_images_gen = 0
            #         for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
            #             c2w = dset_test.c2w[img_id].to(device=device)
            #             cam = svox2.Camera(c2w,
            #                             dset_test.intrins.get('fx', img_id),
            #                             dset_test.intrins.get('fy', img_id),
            #                             dset_test.intrins.get('cx', img_id),
            #                             dset_test.intrins.get('cy', img_id),
            #                             width=dset_test.get_image_size(img_id)[1],
            #                             height=dset_test.get_image_size(img_id)[0],
            #                             ndc_coeffs=dset_test.ndc_coeffs)
            #             rgb_pred_test = grid.volume_render_image(cam, use_kernel=True)
            #             rgb_gt_test = dset_test.gt[img_id].to(device=device)
            #             all_mses = ((rgb_gt_test - rgb_pred_test) ** 2).cpu()
            #             if i % img_save_interval == 0:
            #                 img_pred = rgb_pred_test.cpu()
            #                 img_pred.clamp_max_(1.0)
            #                 summary_writer.add_image(f'test/image_{img_id:04d}',
            #                         img_pred, global_step=gstep_id_base, dataformats='HWC')
            #                 if log_mse_image:
            #                     mse_img = all_mses / all_mses.max()
            #                     summary_writer.add_image(f'test/mse_map_{img_id:04d}',
            #                             mse_img, global_step=gstep_id_base, dataformats='HWC')
            #                 if log_depth_map:
            #                     depth_img = grid.volume_render_depth_image(cam,
            #                                 log_depth_map_use_thresh if
            #                                 log_depth_map_use_thresh else None
            #                             )
            #                     depth_img = viridis_cmap(depth_img.cpu())
            #                     summary_writer.add_image(f'test/depth_map_{img_id:04d}',
            #                             depth_img,
            #                             global_step=gstep_id_base, dataformats='HWC')

            #             rgb_pred_test = rgb_gt_test = None
            #             mse_num : float = all_mses.mean().item()
            #             psnr = -10.0 * math.log10(mse_num)
            #             if math.isnan(psnr):
            #                 print('NAN PSNR', i, img_id, mse_num)
            #                 assert False
            #             stats_test['mse'] += mse_num
            #             stats_test['psnr'] += psnr
            #             n_images_gen += 1

                    # if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE or grid.basis_type == svox2.BASIS_TYPE_MLP:
                    #     # Add spherical map visualization
                    #     EQ_RESO = 256
                    #     eq_dirs = generate_dirs_equirect(EQ_RESO * 2, EQ_RESO)
                    #     eq_dirs = torch.from_numpy(eq_dirs).to(device=device).view(-1, 3)

                    #     if grid.basis_type == svox2.BASIS_TYPE_MLP:
                    #         sphfuncs = grid._eval_basis_mlp(eq_dirs)
                    #     else:
                    #         sphfuncs = grid._eval_learned_bases(eq_dirs)
                    #     sphfuncs = sphfuncs.view(EQ_RESO, EQ_RESO*2, -1).permute([2, 0, 1]).cpu().numpy()

                    #     stats = [(sphfunc.min(), sphfunc.mean(), sphfunc.max())
                    #             for sphfunc in sphfuncs]
                    #     sphfuncs_cmapped = [viridis_cmap(sphfunc) for sphfunc in sphfuncs]
                    #     for im, (minv, meanv, maxv) in zip(sphfuncs_cmapped, stats):
                    #         cv2.putText(im, "{:.4f} {:.4f} {:.4f}".format(minv,meanv,maxv), (10, 20),
                    #                     0, 0.5, [255, 0, 0])
                    #     sphfuncs_cmapped = np.concatenate(sphfuncs_cmapped, axis=0)
                        # summary_writer.add_image(f'test/spheric',
                        #         sphfuncs_cmapped, global_step=gstep_id_base, dataformats='HWC')
                        # END add spherical map visualization

            #         stats_test['mse'] /= n_images_gen
            #         stats_test['psnr'] /= n_images_gen
            #         for stat_name in stats_test:
            #             summary_writer.add_scalar('test/' + stat_name,
            #                     stats_test[stat_name], global_step=gstep_id_base)
            #         summary_writer.add_scalar('epoch_id', float(epoch_id), global_step=gstep_id_base)
            #         print('eval stats:', stats_test)
            # if epoch_id % max(factor, eval_every) == 0: #and (epoch_id > 0 or not tune_mode):
            #     # NOTE: we do an eval sanity check, if not in tune_mode
            #     eval_step()
            #     gc.collect()

            def train_step():
                # print('Train step')
                pbar = enumerate(range(0, epoch_size, batch_size))
                stats = {"mse" : 0.0, "psnr" : 0.0, "invsqr_mse" : 0.0}
                for iter_id, batch_begin in pbar:
                    gstep_id = iter_id + gstep_id_base
                    if lr_fg_begin_step > 0 and gstep_id == lr_fg_begin_step:
                        grid.density_data.data[:] = init_sigma
                    lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
                    lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
                    lr_basis = lr_basis_func(gstep_id - lr_basis_begin_step) * lr_basis_factor
                    # lr_sigma_bg = lr_sigma_bg_func(gstep_id - lr_basis_begin_step) * lr_basis_factor
                    # lr_color_bg = lr_color_bg_func(gstep_id - lr_basis_begin_step) * lr_basis_factor
                    # if not lr_decay:
                    lr_sigma = lr_sigma * lr_sigma_factor
                    lr_sh = lr_sh * lr_sh_factor
                    lr_basis = lr_basis * lr_basis_factor

                    batch_end = min(batch_begin + batch_size, epoch_size)
                    batch_origins = dset.rays.origins[batch_begin: batch_end]
                    batch_dirs = dset.rays.dirs[batch_begin: batch_end]
                    rgb_gt = dset.rays.gt[batch_begin: batch_end]
                    rays = svox2.Rays(batch_origins, batch_dirs)

                    #  with Timing("volrend_fused"):
                    rgb_pred = grid.volume_render_fused(rays, rgb_gt,
                            beta_loss=0.0,
                            sparsity_loss=0.0,
                            randomize=False)

                    #  with Timing("loss_comp"):
                    mse = F.mse_loss(rgb_gt, rgb_pred)

                    # Stats
                    mse_num : float = mse.detach().item()
                    psnr = -10.0 * math.log10(mse_num)
                    stats['mse'] += mse_num
                    stats['psnr'] += psnr
                    stats['invsqr_mse'] += 1.0 / mse_num ** 2

                    # if (iter_id + 1) % print_every == 0:
                    #     # Print averaged stats
                    #     pbar.set_description(f'epoch {epoch_id} psnr={psnr:.2f}')
                    #     for stat_name in stats:
                    #         stat_val = stats[stat_name] / print_every
                    #         summary_writer.add_scalar(stat_name, stat_val, global_step=gstep_id)
                    #         stats[stat_name] = 0.0
                    #     #  if lambda_tv > 0.0:
                    #     #      with torch.no_grad():
                    #     #          tv = grid.tv(logalpha=tv_logalpha, ndc_coeffs=dset.ndc_coeffs)
                    #     #      summary_writer.add_scalar("loss_tv", tv, global_step=gstep_id)
                    #     #  if lambda_tv_sh > 0.0:
                    #     #      with torch.no_grad():
                    #     #          tv_sh = grid.tv_color()
                    #     #      summary_writer.add_scalar("loss_tv_sh", tv_sh, global_step=gstep_id)
                    #     #  with torch.no_grad():
                    #     #      tv_basis = grid.tv_basis() #  summary_writer.add_scalar("loss_tv_basis", tv_basis, global_step=gstep_id)
                    #     summary_writer.add_scalar("lr_sh", lr_sh, global_step=gstep_id)
                    #     summary_writer.add_scalar("lr_sigma", lr_sigma, global_step=gstep_id)
                    #     # if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                    #     #     summary_writer.add_scalar("lr_basis", lr_basis, global_step=gstep_id)
                    #     # if grid.use_background:
                    #     #     summary_writer.add_scalar("lr_sigma_bg", lr_sigma_bg, global_step=gstep_id)
                    #     #     summary_writer.add_scalar("lr_color_bg", lr_color_bg, global_step=gstep_id)

                    #     if weight_decay_sh < 1.0:
                    #         grid.sh_data.data *= weight_decay_sigma
                    #     if weight_decay_sigma < 1.0:
                    #         grid.density_data.data *= weight_decay_sh

                    #  # For outputting the % sparsity of the gradient
                    #  indexer = grid.sparse_sh_grad_indexer
                    #  if indexer is not None:
                    #      if indexer.dtype == torch.bool:
                    #          nz = torch.count_nonzero(indexer)
                    #      else:
                    #          nz = indexer.size()
                    #      with open(os.path.join(train_dir, 'grad_sparsity.txt'), 'a') as sparsity_file:
                    #          sparsity_file.write(f"{gstep_id} {nz}\n")

                    # Apply TV/Sparsity regularizers
                    if lambda_tv > 0.0:
                        #  with Timing("tv_inpl"):
                        grid.inplace_tv_grad(grid.density_data.grad,
                                scaling=lambda_tv,
                                sparse_frac=tv_sparsity,
                                logalpha=tv_logalpha,
                                ndc_coeffs=dset.ndc_coeffs,
                                contiguous=1)
                    if lambda_tv_sh > 0.0:
                        #  with Timing("tv_color_inpl"):
                        grid.inplace_tv_color_grad(grid.sh_data.grad,
                                scaling=lambda_tv_sh,
                                sparse_frac=tv_sh_sparsity,
                                ndc_coeffs=dset.ndc_coeffs,
                                contiguous=1)
                    if lambda_tv_lumisphere > 0.0:
                        grid.inplace_tv_lumisphere_grad(grid.sh_data.grad,
                                scaling=lambda_tv_lumisphere,
                                dir_factor=tv_lumisphere_dir_factor,
                                sparse_frac=tv_lumisphere_sparsity,
                                ndc_coeffs=dset.ndc_coeffs)
                    if lambda_l2_sh > 0.0:
                        grid.inplace_l2_color_grad(grid.sh_data.grad,
                                scaling=lambda_l2_sh)
                    # if grid.use_background and (lambda_tv_background_sigma > 0.0 or lambda_tv_background_color > 0.0):
                    #     grid.inplace_tv_background_grad(grid.background_data.grad,
                    #             scaling=lambda_tv_background_color,
                    #             scaling_density=lambda_tv_background_sigma,
                    #             sparse_frac=tv_background_sparsity,
                    #             contiguous=tv_contiguous)
                    # if lambda_tv_basis > 0.0:
                    #     tv_basis = grid.tv_basis()
                    #     loss_tv_basis = tv_basis * lambda_tv_basis
                    #     loss_tv_basis.backward()
                    #  print('nz density', torch.count_nonzero(grid.sparse_grad_indexer).item(),
                    #        ' sh', torch.count_nonzero(grid.sparse_sh_grad_indexer).item())

                    # Manual SGD/rmsprop step
                    if gstep_id >= lr_fg_begin_step:
                        grid.optim_density_step(lr_sigma, beta=rms_beta, optim='rmsprop')
                        grid.optim_sh_step(lr_sh, beta=rms_beta, optim=sh_optim)
                    # if grid.use_background:
                    #     grid.optim_background_step(lr_sigma_bg, lr_color_bg, beta=rms_beta, optim=bg_optim)
                    # if gstep_id >= lr_basis_begin_step:
                        if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                            grid.optim_basis_step(lr_basis, beta=rms_beta, optim='rmsprop')
                        elif grid.basis_type == svox2.BASIS_TYPE_MLP:
                            optim_basis_mlp.step()
                            optim_basis_mlp.zero_grad()

            train_step()
            gc.collect()
            gstep_id_base += batches_per_epoch

            #  ckpt_path = path.join(train_dir, f'ckpt_{epoch_id:05d}.npz')
            # Overwrite prev checkpoints since they are very huge
            # if save_every > 0 and (epoch_id + 1) % max(
            #         factor, save_every) == 0 and not tune_mode:
            #     print('Saving', ckpt_path)
            #     grid.save(ckpt_path)

            if (gstep_id_base - last_upsamp_step) >= upsamp_every:
                last_upsamp_step = gstep_id_base
                if reso_id < len(reso_list) - 1:
                    # print('* Upsampling from', [self.vox_res,self.vox_res,self.vox_res], 'to', reso_list[reso_id + 1])
                    if tv_early_only > 0:
                        # print('turning off TV regularization')
                        lambda_tv = 0.0
                        lambda_tv_sh = 0.0
                    elif tv_decay != 1.0:
                        lambda_tv *= tv_decay
                        lambda_tv_sh *= tv_decay

                    reso_id += 1
                    use_sparsify = True
                    z_reso = [self.vox_res,self.vox_res,self.vox_res] if isinstance([self.vox_res,self.vox_res,self.vox_res], int) else [self.vox_res,self.vox_res,self.vox_res][2]
                    grid.resample(reso=[self.vox_res,self.vox_res,self.vox_res],
                            sigma_thresh=density_thresh,
                            weight_thresh=weight_thresh / z_reso if use_sparsify else 0.0,
                            dilate=2, #use_sparsify,
                            cameras=resample_cameras ,
                            max_elements=max_grid_elements)

                    # if grid.use_background and reso_id <= 1:
                    #     grid.sparsify_background(background_density_thresh)

                    # if upsample_density_add:
                    #     grid.density_data.data[:] += upsample_density_add

                if factor > 1 and reso_id < len(reso_list) - 1:
                    print('* Using higher resolution images due to large grid; new factor', factor)
                    factor //= 2
                    dset.gen_rays(factor=factor)
                    dset.shuffle_rays()

            # if gstep_id_base >= n_iters:
                # print('* Final eval and save')
                # eval_step()
                # global_stop_time = datetime.now()
                # secs = (global_stop_time - global_start_time).total_seconds()
                # timings_file = open(os.path.join(train_dir, 'time_mins.txt'), 'a')
                # timings_file.write(f"{secs / 60}\n")
                # if not tune_nosave:
                #     grid.save(ckpt_path)
                # break
        del dset 
        # print("$$$$$$$$$$$$$$$$$$$$$$",grid.basis_data.data.shape)

        in_ccords_ , in_feat_ =  self.coords_and_feats_from_grid(grid)
        return in_ccords_ , in_feat_ ,dset.c2w, dset.gt

    def forward_rendering(self,data_dir,c_coords,c_feats,c2ws,device="cuda:0",for_training=True,input_rendering=False,kill_density_grads=False,quantize=False):
        c_feats = self.denormalize_rf(c_feats)
        basis_dim = self.sh_dim if not input_rendering else self.input_sh_dim
        if quantize:
            c_coords, c_feats = self.crop_sparse_voxels(c_coords.to(torch.float), c_feats)

        reso_list = [self.vox_res, self.vox_res, self.vox_res]
        # reso_id = 1
        args = AttrDict({})
        args.config = os.path.join(data_dir,self.relative_features_dir,"meta.json")

        config_util.maybe_merge_config_file(args,allow_invalid=True)
        dset = datasets["nerf"](data_dir, split= "test",n_images=0,verbose=False)

        # grid = svox2.SparseGrid.load(args.ckpt, device=device)
        grid = svox2.TinySparseGrid(reso=reso_list,
                            center=dset.scene_center,
                            radius=dset.scene_radius,
                            use_sphere_bound=dset.use_sphere_bound,
                            basis_dim=basis_dim,
                            use_z_order=True,
                            device=device,
                            basis_reso=reso_list[0],
                            basis_type=svox2.__dict__['BASIS_TYPE_' + "sh".upper()],
                            mlp_posenc_size=4,
                            mlp_width=32,
                            background_nlayers=0,
                            background_reso=reso_list,
                        )
        config_util.setup_render_opts(grid.opt, args)

        grid.opt.backend = 'cuvol'  # cuvol, svox1, nvol

        grid.sh_data = c_feats[:, 1::].to(device=device).contiguous()
        grid.density_data = c_feats[:, 0][..., None].to(device=device).contiguous()
        if kill_density_grads :
            grid.density_data = grid.density_data.data

        # grid.sh_data = nn.Parameter(sh_data)
        # grid.density_data = nn.Parameter(density_data)
        # grid.links = torch.from_numpy(links).to(device=device)
        grid.capacity = grid.sh_data.size(0)
        grid.links = self.links_from_coords(c_coords, resolution=reso_list).to(
            device=device).to(torch.int).contiguous()

        grid.background_data.data = grid.background_data.data.to(device=device)
        if not for_training:
            grid.eval()
        grid.accelerate()
        n_images = c2ws.size(0)
        img_eval_interval = 1
        n_images_gen = 0
        frames = []
        #  if near_clip >= 0.0:
        grid.opt.near_clip = 0.0 #near_clip
        width = dset.get_image_size(0)[1]
        height = dset.get_image_size(0)[0]

        for img_id in range(0, n_images, img_eval_interval):
            dset_h, dset_w = height, width
            im_size = dset_h * dset_w
            w = dset_w
            h = dset_h

            cam = svox2.Camera(c2ws[img_id],
                            dset.intrins.get('fx', 0),
                            dset.intrins.get('fy', 0),
                            w * 0.5,
                            h * 0.5,
                            w, h,
                            ndc_coeffs=(-1.0, -1.0))
            # torch.cuda.synchronize(device)
            im = grid.volume_render_image(cam, use_kernel=True)
            # torch.cuda.synchronize(device)
            im.clamp_(0.0, 1.0)

            frames.append(im[None,...])
            im = None
            n_images_gen += 1
        c_feats = self.normalize_rf(c_feats)
        return torch.cat(frames,dim=0)

    def construct_grid(self,data_dir,c_coords,c_feats,for_training=False,input_construct=False):
        device = c_coords.device
        c_feats = self.denormalize_rf(c_feats.data)
        basis_dim = self.sh_dim if not input_construct else self.input_sh_dim
        reso_list = [self.vox_res, self.vox_res, self.vox_res]
        # reso_id = 1
        args = AttrDict({})
        args.config = os.path.join(data_dir,self.relative_features_dir,"meta.json")

        config_util.maybe_merge_config_file(args,allow_invalid=True)
        if self.dataset_type == "co3d":
            args["seq_id"] = None
        dset = datasets[self.dataset_type](data_dir, split="test", n_images=0, verbose=False,
                                            **config_util.build_data_options(args))

        # grid = svox2.SparseGrid.load(args.ckpt, device=device)
        grid = svox2.SparseGrid(reso=reso_list,
                            center=dset.scene_center,
                            radius=dset.scene_radius,
                            use_sphere_bound=dset.use_sphere_bound,
                            basis_dim=basis_dim,
                            use_z_order=True,
                            device=device,
                            basis_reso=reso_list[0],
                            basis_type=svox2.__dict__['BASIS_TYPE_' + "sh".upper()],
                            mlp_posenc_size=4,
                            mlp_width=32,
                            background_nlayers=0,
                            background_reso=reso_list,
                        )
        grid.sh_data = torch.nn.Parameter(c_feats[:, 1::].to(device=device).contiguous(), requires_grad=for_training)
        grid.density_data = torch.nn.Parameter(c_feats[:, 0][..., None].to(
            device=device).contiguous(), requires_grad=for_training)
        # grid.sh_data = nn.Parameter(sh_data)
        # grid.density_data = nn.Parameter(density_data)
        # grid.links = torch.from_numpy(links).to(device=device)
        grid.capacity = grid.sh_data.size(0)
        grid.links = self.links_from_coords(c_coords, resolution=reso_list).to(
            device=device).to(torch.int).contiguous()

        grid.background_data.data = grid.background_data.data.to(device=device)
        if not for_training:
            grid.eval()
        grid.accelerate()
        c_feats = self.normalize_rf(c_feats)
        return grid 

    def evaluate_grid(self, grid, data_dir, no_lpips=False, split="test"):
        device = grid.density_data.device
        return_raylen = False
        args = AttrDict({})
        args.config = os.path.join(data_dir,self.relative_features_dir,"meta.json")

        config_util.maybe_merge_config_file(args,allow_invalid=True)

        if not no_lpips:
            import lpips
            lpips_vgg = lpips.LPIPS(net="vgg",verbose=False).eval().to(device)
        # if not path.isfile(args.ckpt):
        #     args.ckpt = path.join(args.ckpt, 'ckpt.npz')

        # render_dir = path.join(path.dirname(args.ckpt),
        #             'train_renders' if args.train else 'test_renders')
        want_metrics = True
        # if args.render_path:
        #     assert not args.train
        #     render_dir += '_path'
        #     want_metrics = False

        # Handle various image transforms
        # if not args.render_path:
        #     # Do not crop if not render_path
        #     args.crop = 1.0
        # if args.crop != 1.0:
        #     render_dir += f'_crop{args.crop}'
        # if args.ray_len:
        #     render_dir += f'_raylen'
        #     want_metrics = False

        dset = datasets["nerf"](data_dir, split=split, verbose=False,
                                            **config_util.build_data_options(args))

        # grid = svox2.SparseGrid.load(args.ckpt, device=device)
        # grid = svox2.SparseGrid(reso=[self.vox_res,self.vox_res,self.vox_res],
        #                     center=dset.scene_center,
        #                     radius=dset.scene_radius,
        #                     use_sphere_bound=dset.use_sphere_bound,
        #                     basis_dim=basis_dim,
        #                     use_z_order=True,
        #                     device=device,
        #                     basis_reso=self.vox,
        #                     basis_type=svox2.__dict__['BASIS_TYPE_' + basis_type.upper()],
        #                     mlp_posenc_size=4,
        #                     mlp_width=32,
        #                     background_nlayers=0,
        #                     background_reso=[self.vox_res,self.vox_res,self.vox_res],
        #                 )

        # print(grid.use_background,grid.basis_type,grid.sh_data.shape,grid.density_data.shape,grid.capacity,(torch.unique(grid.links)).max())
        # raise Exception("STOP HERE ")
        # if grid.use_background:
        #     if args.nobg:
        #         #  grid.background_cubemap.data = grid.background_cubemap.data.cuda()
        #         grid.background_data.data[..., -1] = 0.0
        #         render_dir += '_nobg'
        #     if args.nofg:
        #         grid.density_data.data[:] = 0.0
        #         #  grid.sh_data.data[..., 0] = 1.0 / svox2.utils.SH_C0
        #         #  grid.sh_data.data[..., 9] = 1.0 / svox2.utils.SH_C0
        #         #  grid.sh_data.data[..., 18] = 1.0 / svox2.utils.SH_C0
        #         render_dir += '_nofg'

            # DEBUG
            #  grid.links.data[grid.links.size(0)//2:] = -1
            #  render_dir += "_chopx2"

        config_util.setup_render_opts(grid.opt, args)
        grid.opt.backend = 'cuvol' # cuvol, svox1, nvol

        # if args.blackbg:
        #     print('Forcing black bg')
        #     render_dir += '_blackbg'
        #     grid.opt.background_brightness = 0.0

        # print('Writing to', render_dir)
        # os.makedirs(render_dir, exist_ok=True)

        # if not args.no_imsave:
        #     print('Will write out all frames as PNG (this take most of the time)')

        # NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
        # other backends will manually generate rays per frame (slow)
        with torch.no_grad():
            n_images = dset.n_images
            img_eval_interval = 1
            avg_psnr = 0.0
            avg_ssim = 0.0
            avg_lpips = 0.0
            n_images_gen = 0
            c2ws = dset.c2w.to(device=device)
            # DEBUGGING
            #  rad = [1.496031746031746, 1.6613756613756614, 1.0]
            #  half_sz = [grid.links.size(0) // 2, grid.links.size(1) // 2]
            #  pad_size_x = int(half_sz[0] - half_sz[0] / 1.496031746031746)
            #  pad_size_y = int(half_sz[1] - half_sz[1] / 1.6613756613756614)
            #  print(pad_size_x, pad_size_y)
            #  grid.links[:pad_size_x] = -1
            #  grid.links[-pad_size_x:] = -1
            #  grid.links[:, :pad_size_y] = -1
            #  grid.links[:, -pad_size_y:] = -1
            #  grid.links[:, :, -8:] = -1

            #  LAYER = -16
            #  grid.links[:, :, :LAYER] = -1
            #  grid.links[:, :, LAYER+1:] = -1

            # frames = []
            #  im_gt_all = dset.gt.to(device=device)
            # print(grid.links.min(), grid.links.max(),len(torch.unique(grid.links)))

            for img_id in range(0, n_images, img_eval_interval):
                dset_h, dset_w = dset.get_image_size(img_id)
                im_size = dset_h * dset_w
                w = dset_w 
                h = dset_h 

                cam = svox2.Camera(c2ws[img_id],
                                dset.intrins.get('fx', img_id),
                                dset.intrins.get('fy', img_id),
                                dset.intrins.get('cx', img_id) + (w - dset_w) * 0.5,
                                dset.intrins.get('cy', img_id) + (h - dset_h) * 0.5,
                                w, h,
                                ndc_coeffs=dset.ndc_coeffs)
                # torch.cuda.synchronize(device)
                # print("\n\ngrid.density_data: ", grid.density_data.data.max(),grid.density_data.data.min())
                # print("grid.sh_data: ", grid.sh_data.data.max(),grid.sh_data.data.min())
                # print("\n\ngrid.density_data: ", torch.norm(grid.density_data.data))
                # print("grid.sh_data: ", torch.norm(grid.sh_data.data))
                # print(img_id)
                im = grid.volume_render_image(cam, use_kernel=True, return_raylen=return_raylen)
                # try:
                #     im = torch.from_numpy(im.cpu().numpy()).to(device=device)
                # except:
                #     print("ISSUE")
                # print("$$$$$$$$$$$$$$$$$$,", im.max(), im.min(), im.mean(),im[0,:1,0])

                # torch.cuda.synchronize(device)

                # if args.ray_len:
                #     minv, meanv, maxv = im.min().item(), im.mean().item(), im.max().item()
                #     im = viridis_cmap(im.cpu().numpy())
                #     cv2.putText(im, "{:.4f} {:.4f} {:.4f}".format(minv,meanv,maxv), (10, 20),
                #                 0, 0.5, [255, 0, 0])
                #     im = torch.from_numpy(im).to(device=device)

                # print("\n\nPASSED",im.max(), im.min())
                im.clamp_(0.0, 1.0) ## this was causing issues with Munkoski engine because of repeated coords with different values   

                # if not args.render_path:
                im_gt = dset.gt[img_id].to(device=device)
                mse = (im - im_gt) ** 2
                mse_num : float = mse.mean().item()
                try:
                    psnr = -10.0 * math.log10(mse_num)
                except:
                    psnr = 0.0 

                avg_psnr += psnr
                # if not args.timing:
                ssim = compute_ssim(im_gt, im).item()
                avg_ssim += ssim
                if not no_lpips:
                    lpips_i = lpips_vgg(im_gt.permute([2, 0, 1]).contiguous(),
                            im.permute([2, 0, 1]).contiguous(), normalize=True).item()
                    avg_lpips += lpips_i
                    # print(img_id, 'PSNR', psnr, 'SSIM', ssim, 'LPIPS', lpips_i)
                # else:
                #     print(img_id, 'PSNR', psnr, 'SSIM', ssim)
                # img_path = path.join(render_dir, f'{img_id:04d}.png');
                # im = im.cpu().numpy()
                # if not args.render_path:
                #     im_gt = dset.gt[img_id].numpy()
                #     im = np.concatenate([im_gt, im], axis=1)
                # if not args.timing:
                #     im = (im * 255).astype(np.uint8)
                #     if not args.no_imsave:
                #         imageio.imwrite(img_path,im)
                #     if not args.no_vid:
                #         frames.append(im)
                im = None
                n_images_gen += 1
            if want_metrics:
                # print('AVERAGES')

                avg_psnr /= n_images_gen
                # with open(path.join(render_dir, 'psnr.txt'), 'w') as f:
                #     f.write(str(avg_psnr))
                # print('PSNR:', avg_psnr)
                # if not args.timing:
                avg_ssim /= n_images_gen
                # print('SSIM:', avg_ssim)
                # with open(path.join(render_dir, 'ssim.txt'), 'w') as f:
                #     f.write(str(avg_ssim))
                if not no_lpips:
                    avg_lpips /= n_images_gen
                    # print('LPIPS:', avg_lpips)
                    # with open(path.join(render_dir, 'lpips.txt'), 'w') as f:
                    #     f.write(str(avg_lpips))
            # if not args.no_vid and len(frames):
            #     vid_path = render_dir + '.mp4'
            #     imageio.mimwrite(vid_path, frames, fps=args.fps, macro_block_size=8)  # pip install imageio-ffmpeg
        avg_lpips = 0 if  no_lpips else avg_lpips
        return {'PSNR': avg_psnr,'SSIM': avg_ssim , 'LPIPS': avg_lpips}


    def visualize_grid(self,grid,data_dir,render_dir,num_views=600,vizualization_id=0,gif=False,traj_type = 'spiral'):
        device = grid.density_data.device
        offset = "0,0,0"
        vec_up = "0.0,1.0,0.0" 
        closeup_factor = 0.5
        fps=30 # vide frames/second
        elevation = -45
        elevation2 = 20
        radius = 0.85 
        vert_shift = 0.0
        width , height = None,None
        blackbg = False
        args = AttrDict({})
        args.config = os.path.join(data_dir,self.relative_features_dir,"meta.json")
        config_util.maybe_merge_config_file(args,allow_invalid=True)
        # if not path.isfile(args.ckpt):
        #     args.ckpt = path.join(args.ckpt, 'ckpt.npz')

        # render_dir = path.join(path.dirname(args.ckpt),
        #             'train_renders' if args.train else 'test_renders')
        # if args.render_path:
        #     assert not args.train
        #     render_dir += '_path'
        #     want_metrics = False

        # Handle various image transforms
        # if not args.render_path:
        #     # Do not crop if not render_path
        #     args.crop = 1.0
        # if args.crop != 1.0:
        #     render_dir += f'_crop{args.crop}'
        # if args.ray_len:
        #     render_dir += f'_raylen'
        #     want_metrics = False

        dset = datasets["nerf"](data_dir, split= "test",n_images=0,verbose=False,
                                            **config_util.build_data_options(args))

        if vec_up is None:
            up_rot = dset.c2w[:, :3, :3].cpu().numpy()
            ups = np.matmul(up_rot, np.array([0, -1.0, 0])[None, :, None])[..., 0]
            vec_up = np.mean(ups, axis=0)
            vec_up /= np.linalg.norm(vec_up)
            print('  Auto vec_up', vec_up)
        else:
            vec_up = np.array(list(map(float, vec_up.split(","))))


        offset = np.array(list(map(float, offset.split(","))))
        if traj_type == 'spiral':
            angles = np.linspace(-180, 180, num_views + 1)[:-1]
            elevations = np.linspace(elevation, elevation2, num_views)
            c2ws = [
                pose_spherical(
                    angle,
                    ele,
                    radius,
                    offset,
                    vec_up=vec_up,
                )
                for ele, angle in zip(elevations, angles)
            ]
            c2ws += [
                pose_spherical(
                    angle,
                    ele,
                    radius,
                    offset,
                    vec_up=vec_up,
                )
                for ele, angle in zip(reversed(elevations), angles)
            ]


        elif traj_type == 'zoom':
            angles = np.linspace(-180, 180, num_views + 1)[:-1]
            elevations = np.linspace(elevation, elevation2, num_views)
            distances = np.linspace(radius, radius *
                                    closeup_factor, num_views)
            c2ws = [
                pose_spherical(
                    angle,
                    ele,
                    dist,
                    offset,
                    vec_up=vec_up,
                )
                for ele, angle, dist in zip(elevations, angles, distances)
            ]
            c2ws += [
                pose_spherical(
                    angle,
                    ele,
                    dist,
                    offset,
                    vec_up=vec_up,
                )
                for ele, angle, dist in zip(reversed(elevations), angles, reversed(distances))
            ]
        elif traj_type == 'circle':
            c2ws = [
                pose_spherical(
                    angle,
                    elevation,
                    radius,
                    offset,
                    vec_up=vec_up,
                )
                for angle in np.linspace(-180, 180, num_views + 1)[:-1]
            ]
        c2ws = np.stack(c2ws, axis=0)
        if vert_shift != 0.0:
            c2ws[:, :3, 3] += np.array(vec_up) * vert_shift
        c2ws = torch.from_numpy(c2ws).to(device=device)

        # if not path.isfile(ckpt):
        #     ckpt = path.join(ckpt, 'ckpt.npz')

        render_out_path = path.join(render_dir, '{}_renders_{}'.format(traj_type,vizualization_id))

        # Handle various image transforms
        # if crop != 1.0:
        #     render_out_path += f'_crop{crop}'
        if vert_shift != 0.0:
            render_out_path += f'_vshift{vert_shift}'

        # grid = svox2.SparseGrid.load(ckpt, device=device)
        # print(grid.center, grid.radius)

        # DEBUG
        #  grid.background_data.data[:, 32:, -1] = 0.0
        #  render_out_path += '_front'

        # if grid.use_background:
        #     if nobg:
        #         grid.background_data.data[..., -1] = 0.0
        #         render_out_path += '_nobg'
        #     if nofg:
        #         grid.density_data.data[:] = 0.0
        #         #  grid.sh_data.data[..., 0] = 1.0 / svox2.utils.SH_C0
        #         #  grid.sh_data.data[..., 9] = 1.0 / svox2.utils.SH_C0
        #         #  grid.sh_data.data[..., 18] = 1.0 / svox2.utils.SH_C0
        #         render_out_path += '_nofg'

            #  # DEBUG
            #  grid.background_data.data[..., -1] = 100.0
            #  a1 = torch.linspace(0, 1, grid.background_data.size(0) // 2, dtype=torch.float32, device=device)[:, None]
            #  a2 = torch.linspace(1, 0, (grid.background_data.size(0) - 1) // 2 + 1, dtype=torch.float32, device=device)[:, None]
            #  a = torch.cat([a1, a2], dim=0)
            #  c = torch.stack([a, 1-a, torch.zeros_like(a)], dim=-1)
            #  grid.background_data.data[..., :-1] = c
            #  render_out_path += "_gradient"

        config_util.setup_render_opts(grid.opt, args)

        if blackbg:
            print('Forcing black bg')
            render_out_path += '_blackbg'
            grid.opt.background_brightness = 0.0

        if gif:
            render_out_path += '.gif'
        else:
            render_out_path += '.mp4'
        # print('Writing to', render_out_path)

        # NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
        # other backends will manually generate rays per frame (slow)
        with torch.no_grad():
            n_images = c2ws.size(0)
            img_eval_interval = 1
            avg_psnr = 0.0
            avg_ssim = 0.0
            avg_lpips = 0.0
            n_images_gen = 0
            frames = []
            #  if near_clip >= 0.0:
            grid.opt.near_clip = 0.0 #near_clip
            if width is None:
                width = dset.get_image_size(0)[1]
            if height is None:
                height = dset.get_image_size(0)[0]

            for img_id in range(0, n_images, img_eval_interval):
                dset_h, dset_w = height, width
                im_size = dset_h * dset_w
                w = dset_w
                h = dset_h

                cam = svox2.Camera(c2ws[img_id],
                                dset.intrins.get('fx', 0),
                                dset.intrins.get('fy', 0),
                                w * 0.5,
                                h * 0.5,
                                w, h,
                                ndc_coeffs=(-1.0, -1.0))
                torch.cuda.synchronize(device)
                im = grid.volume_render_image(cam, use_kernel=True)
                torch.cuda.synchronize(device)
                im.clamp_(0.0, 1.0)

                im = im.cpu().numpy()
                im = (im * 255).astype(np.uint8)
                frames.append(im)
                im = None
                n_images_gen += 1
            if len(frames):
                vid_path = render_out_path
                if gif :
                    imageio.mimwrite(vid_path, frames, fps=0.05*fps)  # pip install imageio-ffmpeg
                else:
                    imageio.mimwrite(vid_path, frames, fps=fps, macro_block_size=8)  # pip install imageio-ffmpeg
        
        return frames 

    def load_c2ws_images(self, data_dir,device="cuda:0",split="train",c_rf_variant=0,randomized_views=False,all_views=False):

        basis_dim = self.sh_dim
        # args = parser.parse_args()
        args = AttrDict({})
        args.config = os.path.join(
            data_dir, self.relative_features_dir, "meta.json")

        config_util.maybe_merge_config_file(args, allow_invalid=True)

        density_factor = 1.0
        init_sigma = 0.0

        factor = 1
        if randomized_views:
            arr = list(range(self.max_views))
            np.random.shuffle(arr)
            indices = arr[0:self.online_nb_views]
        else:
            with open(os.path.join(data_dir, self.relative_reflect_dir, "used_images_{}.txt".format(c_rf_variant)), "r") as f:
                indices = json.loads("[" + str(f.read()) + "]")
        if all_views:
            indices = list(range(self.max_views))
        # print("$$$$$$$$$$$$$$$$$4", indices, data_dir)
        dset = datasets["fastnerf"](
            data_dir,
            split="train",
            device=device,
            factor=factor,
            n_images=None,
            randomization=False,
            data_split=split,
            indices=indices,
            verbose=False,
        )
        
        return dset.c2w, dset.gt , dset.masks

# c2ws = [x[0:3, 3] for x in dset.c2w]
    def preprocess_input_rf(self, in_ccord, in_ft, c2ws, imgs,t, transforms=None):
        if self.prune_input:
            in_ccord, in_ft = self.prune_sparse_voxels(in_ccord, in_ft, self.density_threshold)
        in_ccord, in_ft = self.quantize_sparse_voxels(in_ccord.to(torch.float), in_ft, quantization_size=self.input_quantization_size)
        in_ccord = (in_ccord * self.input_quantization_size).to(torch.int32)
        camera_feats = torch.cat([torch.Tensor(x[0:3, 3]) for x in c2ws], dim=0) # TODO : different ways to encod ecameras
        c2ws = torch.cat([torch.Tensor(x)[None, ...]for x in c2ws], dim=0)[None, ...]
        
        imgs = transforms(torch.Tensor(imgs).permute(0, 3, 1, 2))[None, ...]

        if self.encode_cameras:
            camera_feats = positional_encoding(camera_feats, num_encoding_functions=self.num_encoding_functions)[None, ...]
        if self.encode_imgs:
            pass # TODO more recesossing for the imgs before using them 
        t_embed = get_timestep_embedding(t, 1+2*self.num_encoding_functions) if self.encode_time else t
        return in_ccord, in_ft,c2ws, camera_feats, imgs,t_embed

    def construct_and_visualize_grid(self, in_ccord, in_ft, d_dir,render_dir,vizualization_id=0,traj_type="zoom", input_construct=False,gif=False,nb_frames=200):
        grid = self.construct_grid(d_dir, in_ccord, in_ft, input_construct=input_construct)
        return self.visualize_grid(grid, d_dir, render_dir, num_views=nb_frames,vizualization_id=vizualization_id, gif=gif, traj_type=traj_type)

    def extract_and_visualize_mesh(self, in_ccord, in_ft, mesh_dir, mesh_render_dir, vizualization_id=0, smooth=False, density_threshold=0.0, clean=True, render=False, img_res=400):
        mesh_name = os.path.join(mesh_dir, "{}.obj".format(vizualization_id))
        mesh = self.extract_mesh_from_sparse_voxels(in_ccord, in_ft[:, 0], mesh_name, smooth=smooth, level_set=density_threshold, clean=clean)
        if render and len(mesh.vertices) > 10:
            mesh_render_name = os.path.join(mesh_render_dir, "{}.png".format(vizualization_id))
            scene = trimesh.scene.Scene(mesh) 
            save_trimesh_scene(scene,mesh_render_name,resolution=(img_res,img_res),distance=1.2*mesh.scale,angles=(-0.5,3.0,0.0),center=mesh.center_mass,show=False)
    def construct_and_visualize_batch_stensor(self, stensor, data_dirs,render_dir,vizualization_id=0,traj_type="zoom", input_construct=False,gif=False,nb_frames=200):
        all_frames = []
        for ii,d_dir in enumerate(data_dirs):
            coords = stensor.coordinates_at(ii)
            feats = stensor.features_at(ii)
            frames = self.construct_and_visualize_grid(coords, feats, d_dir, render_dir=render_dir, vizualization_id=vizualization_id + ii,
                                                traj_type=traj_type, input_construct=input_construct, gif=gif, nb_frames=nb_frames)
            all_frames.append(frames)
        return all_frames
    def construct_and_visualize_imgs_batch_stensor(self, stensor, data_dict,render_dir_vids,render_dir_imgs,vizualization_id=0,traj_type="zoom", input_construct=False,gif=False,nb_frames=200):
        all_frames = []
        for ii, d_dir in enumerate(data_dict["labels"]):
            c_vid_in_dir = os.path.join(render_dir_vids, str(int(data_dict["in_rf_variant"][ii].item())))
            c_img_in_dir = os.path.join(render_dir_imgs, str(int(data_dict["in_rf_variant"][ii].item())))
            os.makedirs(c_vid_in_dir, exist_ok=True)
            os.makedirs(c_img_in_dir, exist_ok=True)

            if os.path.isfile(os.path.join(c_vid_in_dir, "{}_renders_{}.mp4".format(traj_type, vizualization_id+ii))) and os.path.isfile(os.path.join(c_vid_in_dir, "{}_0.jpg".format(vizualization_id+ii))):
                continue
            coords = stensor.coordinates_at(ii)
            feats = stensor.features_at(ii)
            frames = self.construct_and_visualize_grid(coords, feats, d_dir, render_dir=c_vid_in_dir, vizualization_id=vizualization_id + ii,
                                                traj_type=traj_type, input_construct=input_construct, gif=gif, nb_frames=nb_frames)
            c_imgs = data_dict["imgs"][ii].permute(0,2,3,1)
            # c_imgs = 255* (c_imgs + c_imgs.min().item()) / (c_imgs.max().item() - c_imgs.min().item())
            [imageio.imwrite(os.path.join(c_img_in_dir,"{}_{}.jpg".format(vizualization_id+ii,str(kk))), c_imgs[kk].cpu().numpy()) for kk in range(c_imgs.shape[0])]
            all_frames.append(frames)
        return all_frames
# 

def positional_encoding(
    tensor, num_encoding_functions=0, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb
def save_trimesh_scene(scene,image_name,resolution=(224,224),distance=1.0,angles=(1,0,1),center=(0,0,0),show=False,background=(255,255,255,255),smooth=True,fov=(45,45)):
    scene.set_camera(angles=angles, distance=distance,resolution=resolution, center=center,fov=fov) 
    import pyglet
    window_conf = pyglet.gl.Config(double_buffer=True, depth_size=24) 
    img = scene.save_image(show=show,background=background,smooth=smooth,window_conf=window_conf)
    with open(image_name, "wb") as file:
        file.write(img)
    if background == (0, 255, 0, 255):
        img = imageio.imread(image_name)
        bg_mask =  (img== np.array(background)).all(axis=-1)
        img[bg_mask] = (255,255,255,0)
        imageio.imwrite(image_name,img)
