# SPARF: Large-Scale Learning of 3D Sparse Radiance Fields from Few Input Images (ICCV 2023)
By [Abdullah Hamdi](https://abdullahamdi.com/), [Bernard Ghanem](http://www.bernardghanem.com/), [Matthias Nießner](https://niessnerlab.org/members/matthias_niessner/profile.html) 
### [Paper](https://arxiv.org/abs/2212.09100) | [Video](https://youtu.be/VcjypZ0hp4w) | [Website](https://abdullahamdi.com/sparf/) | [Dataset](https://drive.google.com/drive/folders/19zCvjQJEh30vCzNC32Bvkc8s_s7GjbKR?usp=sharing) . <br>
<p float="left">
<img src="https://user-images.githubusercontent.com/26301932/208697062-829496a7-4a25-42cf-8a67-41cc64b0ea66.gif" align="left" width="250">
<img src="https://user-images.githubusercontent.com/26301932/208697090-2bb7ade0-1cce-4ebe-bbd8-c61d4fcfb587.gif" align="center" width="250">
<img src="https://user-images.githubusercontent.com/26301932/208697114-ce5e0a29-4cec-41ec-b995-e6b41495b042.gif" align="center" width="250">
</p>
 
The official Pytroch code of the paper [SPARF: Large-Scale Learning of 3D Sparse Radiance Fields from Few Input Images](https://arxiv.org/abs/2212.09100). SPARF is a large-scale sparse radiance field dataset consisting of ~ 1 million SRFs with multiple voxel resolutions (32, 128, and 512) and 17 million posed images with a resolution of 400 X 400. Furthermore, we propose SuRFNet, a pipline to generate SRFs conditioned on input images, achieving SOTA on ShapeNet novel views synthesis from one or few input images. 

# Environment setup

To start, we prefer creating the environment using conda:
```sh
conda env create -f environment.yml
conda activate sparf
```
Please make sure you have up-to-date NVIDIA drivers supporting CUDA 11.3 at least.

Alternatively use `pip -r requirements.txt`.
## SPARF Posed Multi-View Image Dataset 
The dataset is released in the [link](https://drive.google.com/drive/folders/19zCvjQJEh30vCzNC32Bvkc8s_s7GjbKR?usp=sharing). Each of SPARF's classes has the same structure of [NeRF-synthetic](https://github.com/sxyu/pixel-nerf) dataset and can be loaded similarly. Download all content in the link and place inside `data/SPARF_images`. Then you can run the [notebook example](https://github.com/ajhamdi/sparf_pytorch/blob/main/examples/mvimage_load.ipynb). 


## SPARF Radiance Field Dataset
The dataset is released in the [link](https://drive.google.com/drive/folders/1Qd_hBrRKR1vlCacOSyK_FN4igkHSbPSM?usp=sharing). Each of SPARF's instances has (beside the posed images above) two directories: `STF` (RGB voxels) and `SRF` (Spherical Harmonics voxels). The full radiance fileds are available under `<CLASS_LABEL>/<INSTANCE_ID>SRF/vox_<VOXEL-RESOLUTION>/full`, where `<VOXEL-RESOLUTION>` is the resolution (32, 128 or 512). The partial SRFs are stored in `<CLASS_LABEL>/<INSTANCE_ID>SRF/vox_<VOXEL-RESOLUTION>/partial` similarly. The partisioning (shards) and splits of the dataset is available on the file `SNRL_splits.csv` in the root of the dataset. 

Download all content in the link and place inside `data/SPARF_srf`. Then you can run the [main training code](#).

## Training and Inference on SPARF


## Citation
If you find our work useful in your research, please consider citing:
```bibtex
@InProceedings{hamdi2022sparf,
  title={SPARF: Large-Scale Learning of 3D Sparse Radiance Fields from Few Input Images},
  author={Hamdi, Abdullah and Ghanem, Bernard and Nie{\ss}ner, Matthias},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
  year={2023}
}
```

