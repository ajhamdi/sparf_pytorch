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
import os
import sys
from turtle import forward
import torchvision
from types import coroutine
import numpy as np
from time import time
from functools import partial

from torch.nn import functional as F
# Must be imported before large libs
try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")
from extra_utils import EmptyModule
import torch
import torch.nn as nn
import torch.utils.data
from blocks import MLP, MVAgregate
import MinkowskiEngine as ME

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
def PointCloud(points, colors=None):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

class MyPruning(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.prune = ME.MinkowskiPruning(**kwargs)
        
    def forward(self,sin,keep):
        if keep.sum() > 0:
            return self.prune(sin, keep)
        else:
            return sin

class SRFHead(nn.Module):
    def __init__(self, in_channel=13,out_channels_with_density=13,strides=1,head_depth=1):
        super().__init__()
        self.in_channel = in_channel
        self.head_depth = head_depth  
        self.out_channels_with_density = out_channels_with_density
        self.head_conv = self.srf_conv_block(self.head_depth, self.in_channel-1, self.out_channels_with_density-1, strides=strides)

    def forward(self, stensor):
        densities = ME.SparseTensor(stensor.F[:,0][...,None], coordinate_map_key=stensor.coordinate_map_key,
                                        coordinate_manager=stensor.coordinate_manager)
        stensor = ME.SparseTensor(stensor.F[:, 1::], coordinate_map_key=stensor.coordinate_map_key,
                                 coordinate_manager=stensor.coordinate_manager)
        stensor = self.head_conv(stensor)
        return ME.cat(densities, stensor)

    def srf_conv_block(self,depth,in_channels, out_channels, strides=1):
        layers = [self.srf_conv_layer(in_channels, out_channels, strides=strides) for _ in range(depth)]
        return nn.Sequential(*layers)
    def srf_conv_layer(self, in_channels, out_channels, strides=1):
        return nn.Sequential(ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=2, stride=strides, dimension=3,
                                                     ),
                             ME.MinkowskiBatchNorm(out_channels),
                             ME.MinkowskiELU(),
                             ME.MinkowskiConvolution(
            out_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU(),
        )
class SparseFeaturesConcater(nn.Module):
    def __init__(self, append_cam_features=False, append_img_features=False, append_time_features=False):
        super().__init__()
        self.append_cam_features = append_cam_features
        self.append_img_features = append_img_features
        self.append_time_features = append_time_features
    def forward(self, stensor, cam_feats, img_feats,time_feats):
        if self.append_cam_features:
            stensor = self.concat_feats_with_sparse(stensor, cam_feats)
        if self.append_img_features:
            stensor = self.concat_feats_with_sparse(stensor, img_feats)
        if self.append_time_features:
            stensor = self.concat_feats_with_sparse(stensor, time_feats)

        return stensor

    def concat_feats_with_sparse(self, stensor, feats):
        feats = self.cast_feats_to_sparse(stensor, feats)
        feats = ME.SparseTensor(feats, coordinate_map_key=stensor.coordinate_map_key,
                                coordinate_manager=stensor.coordinate_manager)
        return ME.cat(stensor, feats)

    def pad_zeros_with_sparse(self, stensor,zeros_pad=0):
        if zeros_pad == 0:
            return stensor
        feats = torch.zeros((stensor.F.shape[0], zeros_pad)).to(stensor.F.device)
        feats = ME.SparseTensor(feats, coordinate_map_key=stensor.coordinate_map_key,
                                coordinate_manager=stensor.coordinate_manager)
        return ME.cat(stensor, feats)

    def cast_feats_to_sparse(self, stensor, feats):
        """
        casts a batched torch tensor `feats` into a batched ME.sparse tensor `stensor` 
        """
        bs = feats.shape[0]
        comb_feats = []
        for jj in range(bs):
            try:
                c_feat_size = stensor.features_at(jj).shape[0]
                comb_feats.append(feats[jj][None, ...].repeat(c_feat_size,1))
            except: # if stensor[jj] is empty 
                continue
        return torch.cat(comb_feats, dim=0)


class SRFEncoder(nn.Module):

    ENC_CHANNELS = [16, 64, 256, 512]

    def __init__(self, srf_encoder_depth = 1,resolution=128, in_nchannel=512, out_nchannel=512, strides=3, normalize="const",  added_cam_latent_channels=0, added_img_latent_channels=0, pooling_method="mean", added_time_latent_channels=0 ,** kwargs):
        nn.Module.__init__(self)

        self.resolution = resolution
        self.srf_encoder_depth = srf_encoder_depth
        append_cam_features = added_cam_latent_channels != 0
        append_img_features = added_img_latent_channels != 0
        append_time_features = added_time_latent_channels != 0
        self.in_nchannel = in_nchannel
        self.out_nchannel = out_nchannel
        self.feat_concater = SparseFeaturesConcater(append_cam_features=append_cam_features, append_img_features=append_img_features, append_time_features=append_time_features)
        self.added_cam_latent_channels = added_cam_latent_channels
        self.added_img_latent_channels = added_img_latent_channels
        self.added_time_latent_channels = added_time_latent_channels

        self.added_latent_channels = added_cam_latent_channels + added_img_latent_channels + added_time_latent_channels
        upconv = True  # strides != 1
        self.expand_coordinates = upconv
        self.strides = strides
        self.upconv = ME.MinkowskiGenerativeConvolutionTranspose if upconv else ME.MinkowskiConvolution
        self.normalize = ME.MinkowskiTanh() if normalize == "tanh" else EmptyModule(1)
        self.globalpooling = ME.MinkowskiGlobalAvgPooling() if pooling_method == "mean" else ME.MinkowskiGlobalMaxPooling()


        # Input sparse tensor must have tensor stride 128.
        enc_ch = self.ENC_CHANNELS

        enc_ch[0] = self.in_nchannel
        enc_ch[-1] = self.out_nchannel
        # Encoder
        self.enc_block_s1 = self.srf_conv_layer(self.in_nchannel, enc_ch[0], strides=1) 
        self.enc_block_s2s4 = self.srf_conv_layer(enc_ch[0], enc_ch[1], strides=self.strides)
        layers = [self.srf_conv_layer(enc_ch[2], enc_ch[2], strides=1) for _ in range(self.srf_encoder_depth-1)]
        self.enc_block_s4s8 = nn.Sequential(self.srf_conv_layer(enc_ch[1], enc_ch[2], strides=self.strides), *layers)
        self.enc_block_s8s16 = self.srf_conv_layer(enc_ch[2], enc_ch[3], strides=1)

    def srf_conv_layer(self, in_channels,out_channels,strides=1):
        return nn.Sequential(ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=2, stride=strides, dimension=3,
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                out_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU(),
        )
    def forward(self, partial_in):
        # print("\n\n\n\partial_in", "$$$$$$$$$$$$$$$$$$",partial_in.coordinates.max(), partial_in.features.shape)

        enc_s1 = self.enc_block_s1(partial_in)
        # print("\n\n\n\nenc_s1","$$$$$$$$$$$$$$$$$$",enc_s1.coordinates.max(),enc_s1.features.shape)

        enc_s4 = self.enc_block_s2s4(enc_s1)
        # print("enc_s4","$$$$$$$$$$$$$$$$$$",enc_s4.coordinates.max(),enc_s4.features.shape)
        ####################################CCCCCCCCCCCCCCCCCCCCCCCCCCCC#######################################################
        # APPENDING LATENT FEATURES
        # enc_s4 = self.feat_concater(enc_s4, c2ws_feats, img_feats)
        # print("enc_s4", "$$$$$$$$$$$$$$$$$$",enc_s4.coordinates.max(), enc_s4.features.shape)

        ####################################CCCCCCCCCCCCCCCCCCCCCCCCCCCC#######################################################

        enc_s8 = self.enc_block_s4s8(enc_s4)
        # print("enc_s8","$$$$$$$$$$$$$$$$$$",enc_s8.coordinates.max(),enc_s8.features.shape)
        enc_s16 = self.enc_block_s8s16(enc_s8)
        # print("enc_s16","$$$$$$$$$$$$$$$$$$",enc_s16.coordinates.max(),enc_s16.features.shape)

        ####################################CCCCCCCCCCCCCCCCCCCCCCCCCCCC#######################################################
        # APPENDING LATENT FEATURES
        # enc_s16 = self.feat_concater(enc_s16, c2ws_feats, img_feats)
        # print("enc_s16", "$$$$$$$$$$$$$$$$$$",enc_s16.coordinates.max(), enc_s16.features.shape)
        out_feats = self.globalpooling(enc_s16)
        return out_feats.F
class MiniCompletionNet(nn.Module):

    ENC_CHANNELS = [16, 64, 128, 512]
    DEC_CHANNELS = [16, 64, 128, 512]

    def __init__(self,head_depth, resolution=128, in_nchannel=512, out_nchannel=512, batch_size=2, enable_pruning=False, prune_last_layer=False, strides=3, normalize="const",  added_cam_latent_channels=0, added_img_latent_channels=0, added_time_latent_channels=0, joint_heads=False, ** kwargs):
        nn.Module.__init__(self)

        self.resolution = resolution
        self.joint_heads = joint_heads
        append_cam_features = added_cam_latent_channels != 0
        append_img_features = added_img_latent_channels != 0
        append_time_features = added_time_latent_channels!= 0
        self.in_nchannel = in_nchannel
        self.out_nchannel = out_nchannel
        self.feat_concater = SparseFeaturesConcater(append_cam_features=append_cam_features, append_img_features=append_img_features, append_time_features=append_time_features)

        self.added_cam_latent_channels = added_cam_latent_channels
        self.added_img_latent_channels = added_img_latent_channels
        self.added_time_latent_channels = added_time_latent_channels
        
        self.added_latent_channels = added_cam_latent_channels + added_img_latent_channels + added_time_latent_channels
        # print("FFFFFFFFFFFFF", self.added_cam_latent_channels,self.added_img_latent_channels,self.added_time_latent_channels)
        upconv = True # strides != 1
        self.expand_coordinates = upconv
        self.strides = strides
        size_tensor = torch.empty((batch_size, self.out_nchannel, self.resolution, self.resolution, self.resolution)).size()
        self.upconv = ME.MinkowskiGenerativeConvolutionTranspose if upconv else ME.MinkowskiConvolution
        self.normalize = ME.MinkowskiTanh() if normalize == "tanh" else EmptyModule(1)

        # Input sparse tensor must have tensor stride 128.
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS
        dec_ch[0] = self.out_nchannel
        enc_ch[0] = self.out_nchannel
        self.head = SRFHead(self.out_nchannel, self.out_nchannel,head_depth=head_depth) if not self.joint_heads else EmptyModule(1)

        # Encoder
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.in_nchannel, enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiELU(),
        )


        self.enc_block_s2s4 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[0], enc_ch[1], kernel_size=2, stride=self.strides, dimension=3,
            ),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                enc_ch[1], enc_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s4s8 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[1]+self.added_latent_channels, enc_ch[2], kernel_size=2, stride=self.strides, dimension=3,
            ),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                enc_ch[2], enc_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s8s16 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[2], enc_ch[3], kernel_size=2, stride=1, dimension=3,
            ),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                enc_ch[3], enc_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
        )

        self.dec_block_s64s32 = nn.Sequential(
            # ME.MinkowskiGenerativeConvolutionTranspose(
            # self.upconv(
            ME.MinkowskiConvolution(
                enc_ch[3]+self.added_latent_channels,
                dec_ch[2],
                kernel_size=4,
                stride=1,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                dec_ch[2], dec_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
        )

        self.dec_s32_cls = ME.MinkowskiConvolution(
            dec_ch[2], 1, kernel_size=1, bias=True, dimension=3) if enable_pruning else EmptyModule(1)


        self.dec_block_s16s8 = nn.Sequential(
            # ME.MinkowskiGenerativeConvolutionTranspose(
            self.upconv(
                dec_ch[2],
                dec_ch[1]+self.added_latent_channels,
                kernel_size=2,
                stride=self.strides,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[1]+self.added_latent_channels),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                dec_ch[1]+self.added_latent_channels, dec_ch[1]+self.added_latent_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]+self.added_latent_channels),
            ME.MinkowskiELU(),
        )

        self.dec_s8_cls = ME.MinkowskiConvolution(
            dec_ch[1], 1, kernel_size=1, bias=True, dimension=3) if enable_pruning else EmptyModule(1)

        self.dec_block_s8s4 = nn.Sequential(
            # ME.MinkowskiGenerativeConvolutionTranspose(
            self.upconv(
                dec_ch[1]+2*self.added_latent_channels,
                dec_ch[0],
                kernel_size=2,
                stride=self.strides,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                dec_ch[0], dec_ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
        )

        self.dec_s4_cls = ME.MinkowskiConvolution(
            dec_ch[0]+self.added_latent_channels, enc_ch[3], 1, kernel_size=1, bias=True, dimension=3) if enable_pruning else EmptyModule(1)


        self.dec_block_s2s1 = nn.Sequential(
            # ME.MinkowskiGenerativeConvolutionTranspose(
            # self.upconv(
            ME.MinkowskiConvolution(
                dec_ch[0],
                self.out_nchannel,
                kernel_size=2,
                stride=1,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(self.out_nchannel),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                self.out_nchannel, self.out_nchannel, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(self.out_nchannel),
            ME.MinkowskiELU(),
        )
        self.dec_s1_cls = ME.MinkowskiConvolution(
            self.out_nchannel, 1, kernel_size=1, bias=True, dimension=3) if enable_pruning else EmptyModule(1)

        # pruning
        self.pruning = MyPruning() if enable_pruning else EmptyModule(1)
        self.pruning_last = MyPruning() if prune_last_layer else EmptyModule(1)
        self.to_dense = ME.MinkowskiToDenseTensor(size_tensor)


    def forward(self, partial_in, c2ws_feats, img_feats,time_feats):
        enc_s1 = self.enc_block_s1(partial_in)
        # print("\n\n\n\nenc_s1","$$$$$$$$$$$$$$$$$$",enc_s1.coordinates.max(),enc_s1.features.shape)

        enc_s4 = self.enc_block_s2s4(enc_s1)
        # print("enc_s4","$$$$$$$$$$$$$$$$$$",enc_s4.coordinates.max(),enc_s4.features.shape)
        ####################################CCCCCCCCCCCCCCCCCCCCCCCCCCCC#######################################################
        # APPENDING LATENT FEATURES
        enc_s4 = self.feat_concater(enc_s4, c2ws_feats, img_feats,time_feats)
        # print("enc_s4", "$$$$$$$$$$$$$$$$$$",enc_s4.coordinates.max(), enc_s4.features.shape)

        ####################################CCCCCCCCCCCCCCCCCCCCCCCCCCCC#######################################################

        enc_s8 = self.enc_block_s4s8(enc_s4)
        # print("enc_s8","$$$$$$$$$$$$$$$$$$",enc_s8.coordinates.max(),enc_s8.features.shape)
        enc_s16 = self.enc_block_s8s16(enc_s8)
        # print("enc_s16","$$$$$$$$$$$$$$$$$$",enc_s16.coordinates.max(),enc_s16.features.shape)

        ####################################CCCCCCCCCCCCCCCCCCCCCCCCCCCC#######################################################
        # APPENDING LATENT FEATURES
        enc_s16 = self.feat_concater(enc_s16, c2ws_feats, img_feats,time_feats)
        # print("enc_s16", "$$$$$$$$$$$$$$$$$$",enc_s16.coordinates.max(), enc_s16.features.shape)

        ####################################CCCCCCCCCCCCCCCCCCCCCCCCCCCC#######################################################

        # Decoder 64 -> 32
        ##################################################
        dec_s32 = self.dec_block_s64s32(enc_s16)
        # print("dec_s32","$$$$$$$$$$$$$$$$$$",dec_s32.coordinates.max(),dec_s32.features.shape)

        # Add encoder features
        dec_s32 = dec_s32 + enc_s8
        dec_s32_cls = self.dec_s32_cls(dec_s32)
        keep_s32 = (dec_s32_cls.F > 0).squeeze()

        dec_s32 = self.pruning(dec_s32, keep_s32)

        ##################################################
        # Decoder 16 -> 8
        ##################################################
        dec_s8 = self.dec_block_s16s8(dec_s32)
        # print("dec_s8","$$$$$$$$$$$$$$$$$$",dec_s8.coordinates.max(),dec_s8.features.shape)

        # Add encoder features
        dec_s8 = dec_s8 + enc_s4
        dec_s8_cls = self.dec_s8_cls(dec_s8)
        keep_s8 = (dec_s8_cls.F > 0).squeeze()



        # Remove voxels s16
        dec_s8 = self.pruning(dec_s8, keep_s8)

        # APPENDING LATENT FEATURES
        dec_s8 = self.feat_concater(dec_s8, c2ws_feats, img_feats,time_feats)
        ####################################CCCCCCCCCCCCCCCCCCCCCCCCCCCC#######################################################
        dec_s2 = self.dec_block_s8s4(dec_s8)
        # print("dec_s2","$$$$$$$$$$$$$$$$$$",dec_s2.coordinates.max(),dec_s2.features.shape)

        # Add encoder features
        dec_s2 = dec_s2 + enc_s1
        dec_s2_cls = self.dec_s4_cls(dec_s2)
        keep_s2 = (dec_s2_cls.F > 0).squeeze()


        # Remove voxels s2
        dec_s2 = self.pruning(dec_s2, keep_s2)

        ##################################################
        # Decoder 2 -> 1
        ##################################################
        dec_s1 = self.dec_block_s2s1(dec_s2)
        # print("dec_s1", "$$$$$$$$$$$$$$$$$$",dec_s1.coordinates.max(), dec_s1.features.shape)

        dec_s1_cls = self.dec_s1_cls(dec_s1)

        # Add encoder features

        dec_s1 = dec_s1 + enc_s1
        dec_s1_cls = self.dec_s1_cls(dec_s1)
        keep_s1 = (dec_s1_cls.F > 0).squeeze()

        dec_s1 = self.pruning_last(dec_s1, keep_s1)
        dec_s1 = self.head(dec_s1)
        dec_s1 = self.normalize(dec_s1)

        # print("dec_s1","$$$$$$$$$$$$$$$$$$",dec_s1.coordinates.max(),dec_s1.features.shape)
        return dec_s1  # , self.to_dense(dec_s1)

class CompletionNet(nn.Module):

    ENC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    DEC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]

    def __init__(self,head_depth, resolution=128, in_nchannel=512, out_nchannel=512, batch_size=2, enable_pruning=False, prune_last_layer=False, strides=3, normalize="const",  added_cam_latent_channels=0, added_img_latent_channels=0, added_time_latent_channels=0,joint_heads=False, ** kwargs):
        nn.Module.__init__(self)

        self.resolution = resolution
        self.joint_heads = joint_heads

        append_cam_features = added_cam_latent_channels != 0 
        append_img_features = added_img_latent_channels != 0
        append_time_features = added_time_latent_channels != 0
        self.in_nchannel = in_nchannel
        self.out_nchannel = out_nchannel
        self.feat_concater = SparseFeaturesConcater(append_cam_features=append_cam_features, append_img_features=append_img_features, append_time_features=append_time_features)
        self.added_cam_latent_channels = added_cam_latent_channels
        self.added_img_latent_channels = added_img_latent_channels
        self.added_time_latent_channels = added_time_latent_channels
        
        self.added_latent_channels = added_cam_latent_channels + added_img_latent_channels + added_time_latent_channels
        upconv = strides != 1
        self.expand_coordinates = upconv
        self.strides = strides
        size_tensor = torch.empty((batch_size, self.out_nchannel, self.resolution, self.resolution, self.resolution)).size()
        self.upconv = ME.MinkowskiGenerativeConvolutionTranspose if  upconv else ME.MinkowskiConvolution
        self.normalize = ME.MinkowskiTanh() if normalize=="tanh" else EmptyModule(1)


        # Input sparse tensor must have tensor stride 128.
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS
        dec_ch[0] = self.out_nchannel
        enc_ch[0] = self.out_nchannel
        self.head = SRFHead(self.out_nchannel,self.out_nchannel,head_depth=head_depth) if not self.joint_heads else EmptyModule(1)

        # Encoder
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(self.in_nchannel, enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s1s2 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[0], enc_ch[1], kernel_size=2, stride=1, dimension=3,
            ),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s2s4 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[1], enc_ch[2], kernel_size=2, stride=self.strides, dimension=3,
            ),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s4s8 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[2]+self.added_latent_channels, enc_ch[3], kernel_size=2, stride=self.strides, dimension=3,
            ),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s8s16 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[3], enc_ch[4], kernel_size=2, stride=self.strides, dimension=3,
            ),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s16s32 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[4], enc_ch[5], kernel_size=2, stride=1, dimension=3,
            ),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s32s64 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[5], enc_ch[6], kernel_size=2, stride=1, dimension=3,
            ),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[6], enc_ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
        )

        # Decoder
        self.dec_block_s64s32 = nn.Sequential(
            # ME.MinkowskiGenerativeConvolutionTranspose(
                # self.upconv(
                ME.MinkowskiConvolution(
                    enc_ch[6]+self.added_latent_channels,
                dec_ch[5],
                kernel_size=4,
                stride=1,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[5], dec_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
        )

        self.dec_s32_cls = ME.MinkowskiConvolution(dec_ch[5], 1, kernel_size=1, bias=True, dimension=3) if enable_pruning else EmptyModule(1)

        self.dec_block_s32s16 = nn.Sequential(
            # ME.MinkowskiGenerativeConvolutionTranspose(
                self.upconv(
                enc_ch[5],
                dec_ch[4],
                kernel_size=2,
                stride=self.strides,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
        )

        self.dec_s16_cls = ME.MinkowskiConvolution(dec_ch[4], 1, kernel_size=1, bias=True, dimension=3) if enable_pruning else EmptyModule(1)

        self.dec_block_s16s8 = nn.Sequential(
            # ME.MinkowskiGenerativeConvolutionTranspose(
                self.upconv(
                dec_ch[4],
                dec_ch[3],
                kernel_size=2,
                stride=self.strides,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
        )

        self.dec_s8_cls = ME.MinkowskiConvolution(
            dec_ch[3], 1, kernel_size=1, bias=True, dimension=3) if enable_pruning else EmptyModule(1)

        self.dec_block_s8s4 = nn.Sequential(
            # ME.MinkowskiGenerativeConvolutionTranspose(
                self.upconv(
                dec_ch[3],
                dec_ch[2],
                kernel_size=2,
                stride=self.strides,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
        )

        self.dec_s4_cls = ME.MinkowskiConvolution(dec_ch[2]+self.added_latent_channels, enc_ch[3], 1, kernel_size=1, bias=True, dimension=3) if enable_pruning else EmptyModule(1)

        self.dec_block_s4s2 = nn.Sequential(
            # ME.MinkowskiGenerativeConvolutionTranspose(
                # self.upconv(

                ME.MinkowskiConvolution(
                    dec_ch[2]+self.added_latent_channels,
                dec_ch[1],
                kernel_size=2,
                stride=1,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
        )

        self.dec_s2_cls = ME.MinkowskiConvolution(dec_ch[1], 1, kernel_size=1, bias=True, dimension=3) if enable_pruning else EmptyModule(1)

        self.dec_block_s2s1 = nn.Sequential(
            # ME.MinkowskiGenerativeConvolutionTranspose(
                # self.upconv(
                ME.MinkowskiConvolution(
                dec_ch[1],
                dec_ch[0],
                kernel_size=2,
                stride=1,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
        )

        self.dec_s1_cls = ME.MinkowskiConvolution(dec_ch[0], 1, kernel_size=1, bias=True, dimension=3) if enable_pruning else EmptyModule(1)

        # pruning
        self.pruning = MyPruning() if enable_pruning else EmptyModule(1)
        self.pruning_last = MyPruning() if  prune_last_layer else EmptyModule(1)
        self.to_dense = ME.MinkowskiToDenseTensor(size_tensor)
        # if self.out_nchannel != self.in_nchannel:
        #     self.in_to_output_skip = nn.Sequential(ME.MinkowskiConvolution(self.in_nchannel, self.out_nchannel, kernel_size=1, stride=1, dimension=3),ME.MinkowskiBatchNorm(self.out_nchannel),
        #     ME.MinkowskiELU(),
        #     )
        # else : 
        #     self.in_to_output_skip = EmptyModule(1)

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
            cm = out.coordinate_manager
            strided_target_key = cm.stride(
                target_key, out.tensor_stride[0],
            )
            kernel_map = cm.kernel_map(
                out.coordinate_map_key,
                strided_target_key,
                kernel_size=kernel_size,
                region_type=1,
            )
            for k, curr_in in kernel_map.items():
                target[curr_in[0].long()] = 1
        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, partial_in, c2ws_feats,img_feats,time_feats):
        # out_cls, targets = [], []
        # print("\n\n\n\nenc_s1", "$$$$$$$$$$$$$$$$$$", self.in_nchannel,partial_in.coordinates.max(), partial_in.features.shape)
        
        enc_s1 = self.enc_block_s1(partial_in)
        # print("\n\n\n\nenc_s1","$$$$$$$$$$$$$$$$$$",enc_s1.coordinates.max(),enc_s1.features.shape)

        enc_s2 = self.enc_block_s1s2(enc_s1)
        # print("enc_s2","$$$$$$$$$$$$$$$$$$",enc_s2.coordinates.max(),enc_s2.features.shape)

        enc_s4 = self.enc_block_s2s4(enc_s2)
        # print("enc_s4","$$$$$$$$$$$$$$$$$$",enc_s4.coordinates.max(),enc_s4.features.shape)
        ####################################CCCCCCCCCCCCCCCCCCCCCCCCCCCC#######################################################
        # APPENDING LATENT FEATURES
        enc_s4 = self.feat_concater(enc_s4, c2ws_feats, img_feats, time_feats)
        # print("enc_s4", "$$$$$$$$$$$$$$$$$$",enc_s4.coordinates.max(), enc_s4.features.shape)
        ####################################CCCCCCCCCCCCCCCCCCCCCCCCCCCC#######################################################

        enc_s8 = self.enc_block_s4s8(enc_s4)
        # print("enc_s8","$$$$$$$$$$$$$$$$$$",enc_s8.coordinates.max(),enc_s8.features.shape)
        enc_s16 = self.enc_block_s8s16(enc_s8)
        # print("enc_s16","$$$$$$$$$$$$$$$$$$",enc_s16.coordinates.max(),enc_s16.features.shape)

        enc_s32 = self.enc_block_s16s32(enc_s16)
        # print("enc_s32","$$$$$$$$$$$$$$$$$$",enc_s32.coordinates.max(),enc_s32.features.shape)

        enc_s64 = self.enc_block_s32s64(enc_s32)
        # print("enc_s32","$$$$$$$$$$$$$$$$$$",enc_s32.coordinates.max(),enc_s32.features.shape)

        ####################################CCCCCCCCCCCCCCCCCCCCCCCCCCCC#######################################################
        # APPENDING LATENT FEATURES
        enc_s64 = self.feat_concater(enc_s64, c2ws_feats, img_feats, time_feats)
        # print("enc_s32","$$$$$$$$$$$$$$$$$$",enc_s32.coordinates.max(),enc_s32.features.shape)

        ####################################CCCCCCCCCCCCCCCCCCCCCCCCCCCC#######################################################


        # Decoder 64 -> 32
        ##################################################
        dec_s32 = self.dec_block_s64s32(enc_s64)
        # print("dec_s32","$$$$$$$$$$$$$$$$$$",dec_s32.coordinates.max(),dec_s32.features.shape)

        # Add encoder features
        dec_s32 = dec_s32 + enc_s32
        dec_s32_cls = self.dec_s32_cls(dec_s32)
        keep_s32 = (dec_s32_cls.F > 0).squeeze()

        # target = self.get_target(dec_s32, target_key)
        # targets.append(target)
        # out_cls.append(dec_s32_cls)

        # if self.training:
        #     keep_s32 += target

        # Remove voxels s32

        dec_s32 = self.pruning(dec_s32, keep_s32)
        
        ##################################################
        # Decoder 32 -> 16
        ##################################################
        dec_s16 = self.dec_block_s32s16(dec_s32)
        # print("dec_s16","$$$$$$$$$$$$$$$$$$",dec_s16.coordinates.max(),dec_s16.features.shape)

        # Add encoder features
        dec_s16 = dec_s16 + enc_s16
        dec_s16_cls = self.dec_s16_cls(dec_s16)
        keep_s16 = (dec_s16_cls.F > 0).squeeze()

        # target = self.get_target(dec_s16, target_key)
        # targets.append(target)
        # out_cls.append(dec_s16_cls)

        # if self.training:
        #     keep_s16 += target

        # Remove voxels s16
        dec_s16 = self.pruning(dec_s16, keep_s16)

        ##################################################
        # Decoder 16 -> 8
        ##################################################
        dec_s8 = self.dec_block_s16s8(dec_s16)
        # print("dec_s8","$$$$$$$$$$$$$$$$$$",dec_s8.coordinates.max(),dec_s8.features.shape)

        # Add encoder features
        dec_s8 = dec_s8 + enc_s8
        dec_s8_cls = self.dec_s8_cls(dec_s8)
        keep_s8 = (dec_s8_cls.F > 0).squeeze()

        # target = self.get_target(dec_s8, target_key)
        # targets.append(target)
        # out_cls.append(dec_s8_cls)

        # if self.training:
        #     keep_s8 += target

        # Remove voxels s16
        dec_s8 = self.pruning(dec_s8, keep_s8)

        ##################################################
        # Decoder 8 -> 4
        ##################################################
        dec_s4 = self.dec_block_s8s4(dec_s8)
        # print("dec_s4","$$$$$$$$$$$$$$$$$$",dec_s4.coordinates.max(),dec_s4.features.shape)
        ####################################CCCCCCCCCCCCCCCCCCCCCCCCCCCC#######################################################
        # APPENDING LATENT FEATURES
        dec_s4 = self.feat_concater(dec_s4, c2ws_feats, img_feats, time_feats)
        ####################################CCCCCCCCCCCCCCCCCCCCCCCCCCCC#######################################################

        # Add encoder features
        dec_s4 = dec_s4 + enc_s4
        dec_s4_cls = self.dec_s4_cls(dec_s4)
        keep_s4 = (dec_s4_cls.F > 0).squeeze()

        # target = self.get_target(dec_s4, target_key)
        # targets.append(target)
        # out_cls.append(dec_s4_cls)

        # if self.training:
        #     keep_s4 += target

        # Remove voxels s4
        dec_s4 = self.pruning(dec_s4, keep_s4)

        ##################################################
        # Decoder 4 -> 2
        ##################################################
        dec_s2 = self.dec_block_s4s2(dec_s4)
        # print("dec_s2","$$$$$$$$$$$$$$$$$$",dec_s2.coordinates.max(),dec_s2.features.shape)

        # Add encoder features
        dec_s2 = dec_s2 + enc_s2
        dec_s2_cls = self.dec_s2_cls(dec_s2)
        keep_s2 = (dec_s2_cls.F > 0).squeeze()

        # target = self.get_target(dec_s2, target_key)
        # targets.append(target)
        # out_cls.append(dec_s2_cls)

        # if self.training:
        #     keep_s2 += target

        # Remove voxels s2
        dec_s2 = self.pruning(dec_s2, keep_s2)

        ##################################################
        # Decoder 2 -> 1
        ##################################################
        dec_s1 = self.dec_block_s2s1(dec_s2)
        # print("dec_s1","$$$$$$$$$$$$$$$$$$",dec_s1.coordinates.max(),dec_s1.features.shape)

        dec_s1_cls = self.dec_s1_cls(dec_s1)

        # Add encoder features

        dec_s1 = dec_s1 + enc_s1
        dec_s1_cls = self.dec_s1_cls(dec_s1)
        keep_s1 = (dec_s1_cls.F > 0).squeeze()

        # target = self.get_target(dec_s1, target_key)
        # targets.append(target)
        # out_cls.append(dec_s1_cls)

        # Last layer does not require adding the target
        # if self.training:
        #     keep_s1 += target

        # Remove voxels s1
        # dec_s1 = self.pruning(dec_s1, keep_s1)
        dec_s1 = self.pruning_last(dec_s1, keep_s1)
        dec_s1 = self.head(dec_s1)
        dec_s1 = self.normalize(dec_s1)

        # print("dec_s1","$$$$$$$$$$$$$$$$$$",dec_s1.coordinates.max(),dec_s1.features.shape)
        return dec_s1 # , self.to_dense(dec_s1)


def conv3d(in_channels, out_channels, kernel_size, bias, padding):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups,
                                   padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups,
                                   padding=padding))


class ExtResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf
    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, **kwargs):
        super(ExtResNetBlock, self).__init__()

        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order,
                                num_groups=num_groups)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out

        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2, pool_type='max', basic_module=DoubleConv, conv_layer_order='gcr',
                 num_groups=8, padding=1):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation) followed by a basic module (DoubleConv or ExtResNetBlock).
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upsample (boole): should the input be upsampled
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=(2, 2, 2), basic_module=DoubleConv,
                 conv_layer_order='gcr', num_groups=8, mode='nearest', padding=1, upsample=True):
        super(Decoder, self).__init__()

        if upsample:
            if basic_module == DoubleConv:
                # if DoubleConv is the basic_module use interpolation for upsampling and concatenation joining
                self.upsampling = InterpolateUpsampling(mode=mode)
                # concat joining
                self.joining = partial(self._joining, concat=True)
            else:
                # if basic_module=ExtResNetBlock use transposed convolution upsampling and summation joining
                self.upsampling = TransposeConvUpsampling(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=conv_kernel_size, scale_factor=scale_factor)
                # sum joining
                self.joining = partial(self._joining, concat=False)
                # adapt the number of in_channels for the ExtResNetBlock
                in_channels = out_channels
        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, concat=True)

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


def create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                    pool_kernel_size):
    # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = Encoder(in_channels, out_feature_num,
                              apply_pooling=False,  # skip pooling in the firs encoder
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding)
        else:
            # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
            encoder = Encoder(f_maps[i - 1], out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              pool_kernel_size=pool_kernel_size,
                              padding=conv_padding)

        encoders.append(encoder)

    return nn.ModuleList(encoders)


def create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, upsample):
    # create decoder path consisting of the Decoder modules. The length of the decoder list is equal to `len(f_maps) - 1`
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv:
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
        # currently strides with a constant stride: (2, 2, 2)

        _upsample = True
        if i == 0:
            # upsampling can be skipped only for the 1st decoder, afterwards it should always be present
            _upsample = upsample

        decoder = Decoder(in_feature_num, out_feature_num,
                          basic_module=basic_module,
                          conv_layer_order=layer_order,
                          conv_kernel_size=conv_kernel_size,
                          num_groups=num_groups,
                          padding=conv_padding,
                          upsample=_upsample)
        decoders.append(decoder)
    return nn.ModuleList(decoders)


class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode='nearest'):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True
    """

    def __init__(self, in_channels=None, out_channels=None, kernel_size=3, scale_factor=(2, 2, 2)):
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor,
                                      padding=1)
        super().__init__(upsample)




class Abstract3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, **kwargs):
        super(Abstract3DUNet, self).__init__()
        self.in_nchannel = in_channels
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        upsample=True)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs logits
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        return x


# coords, feats = stensor.decomposed_coordinates_and_features

class UNet3D(Abstract3DUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.
    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     **kwargs)


class ResidualUNet3D(Abstract3DUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ExtResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             **kwargs)
class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)
def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


class SRFNet(nn.Module):
    def __init__(self, time_in_size, cam_in_size, img_in_size, learn_cam_embed, learn_time_embed, network_depth,support_network_depth, use_adv_loss, add_input_late, ** kwargs):
        super().__init__()
        self.network_depth = network_depth
        self.support_network_depth = support_network_depth
        self.use_adv_loss = use_adv_loss
        self.add_input_late = add_input_late
        self.fconcat = SparseFeaturesConcater()
        if self.network_depth == 0:
            self.subnet = nn.ModuleList([CompletionNet(head_depth=support_network_depth,**kwargs)])
        else:
            subnets = [MiniCompletionNet(head_depth=support_network_depth, **kwargs)]
            in_nchannel = int(kwargs["in_nchannel"])
            kwargs["in_nchannel"] = kwargs["out_nchannel"]
            self.subnet = nn.ModuleList(subnets+[MiniCompletionNet(**kwargs) for _ in range(1, network_depth)])
            kwargs["in_nchannel"] = in_nchannel
        if img_in_size > 0:
            indx2depth = {1: 18, 2: 34, 3: 50, 4: 101} # available pretrianed models 
            network_indx = min(max(list(indx2depth.keys())),self.support_network_depth)
            depth2featdim = {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}
            im_model = torchvision.models.__dict__["resnet{}".format(indx2depth[network_indx])](True)
            im_model.fc = torch.nn.Sequential()
            self.img_encoder = MVAgregate(im_model, agr_type=kwargs["pooling_method"], feat_dim=depth2featdim[
                                          indx2depth[self.support_network_depth]], num_classes=kwargs["added_img_latent_channels"])
        else:
            self.img_encoder = EmptyModule(1)
        self.cam_encoder = MLP(([cam_in_size, kwargs["added_cam_latent_channels"]] + self.support_network_depth * [kwargs["added_cam_latent_channels"]])
                               ) if cam_in_size > 0 and learn_cam_embed else EmptyModule(1)
        self.time_encoder = MLP(([time_in_size, kwargs["added_time_latent_channels"]] + self.support_network_depth*[kwargs["added_time_latent_channels"]])
                                ) if time_in_size > 0 and learn_time_embed else EmptyModule(1)

        self.to_feats = ME.MinkowskiToFeature()
        self.srf_encoder =  EmptyModule(1)
        if self.use_adv_loss:
            kwargs["in_nchannel"] = int(kwargs["out_nchannel"])
            kwargs["out_nchannel"] = 1
            self.srf_encoder = SRFEncoder(srf_encoder_depth=support_network_depth, **kwargs)
    def forward(self, sin,c2ws,imgs,times,skip_main=False):
        souts =[]
        if not skip_main:
            c2ws = self.cam_encoder(c2ws)
            imgs = self.img_encoder(imgs)
            times = self.time_encoder(times)

            souts.append(self.subnet[0](sin, c2ws, imgs, times))
            for ii in range(1, len(self.subnet)):
                new_sin = souts[-1] if not self.add_input_late else souts[-1] + self.fconcat.pad_zeros_with_sparse(sin, zeros_pad=souts[-1].F.shape[1]-sin.F.shape[1])
                souts.append(self.subnet[ii](new_sin, c2ws, imgs, times))
            encoded = self.srf_encoder(souts[-1])
        else :
            encoded = self.srf_encoder(sin)
        return souts, encoded

