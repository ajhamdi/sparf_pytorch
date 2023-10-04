# SPARF: Large-Scale Learning of 3D Sparse Radiance Fields from Few Input Images (ICCV 2023)
By [Abdullah Hamdi](https://abdullahamdi.com/), [Bernard Ghanem](http://www.bernardghanem.com/), [Matthias Nießner](https://niessnerlab.org/members/matthias_niessner/profile.html) 
### [Paper](https://openaccess.thecvf.com/content/ICCV2023W/AI3DCC/html/Hamdi_SPARF_Large-Scale_Learning_of_3D_Sparse_Radiance_Fields_from_Few_ICCVW_2023_paper.html) | [Video](https://youtu.be/VcjypZ0hp4w) | [Website](https://abdullahamdi.com/sparf/) | [Dataset](https://drive.google.com/drive/folders/19zCvjQJEh30vCzNC32Bvkc8s_s7GjbKR?usp=sharing) . <br>
<p float="left">
<img src="https://user-images.githubusercontent.com/26301932/208697062-829496a7-4a25-42cf-8a67-41cc64b0ea66.gif" align="left" width="250">
<img src="https://user-images.githubusercontent.com/26301932/208697090-2bb7ade0-1cce-4ebe-bbd8-c61d4fcfb587.gif" align="center" width="250">
<img src="https://user-images.githubusercontent.com/26301932/208697114-ce5e0a29-4cec-41ec-b995-e6b41495b042.gif" align="center" width="250">
</p>
 
The official Pytroch code of the paper [SPARF: Large-Scale Learning of 3D Sparse Radiance Fields from Few Input Images](https://arxiv.org/abs/2212.09100). SPARF is a large-scale sparse radiance field dataset consisting of ~ 1 million SRFs with multiple voxel resolutions (32, 128, and 512) and 17 million posed images with a resolution of 400 X 400. Furthermore, we propose SuRFNet, a pipline to generate SRFs conditioned on input images, achieving SOTA on ShapeNet novel views synthesis from one or few input images. 

# Environment setup

follow instructions in [INTALL.md](https://github.com/ajhamdi/sparf_pytorch/blob/main/INSTALL.md) to setup the conda environment.

## SPARF Posed Multi-View Image Dataset 
The dataset is released in the [link](https://drive.google.com/drive/folders/19zCvjQJEh30vCzNC32Bvkc8s_s7GjbKR?usp=sharing). Each of SPARF's classes has the same structure of [NeRF-synthetic](https://github.com/sxyu/pixel-nerf) dataset and can be loaded similarly. Download all content in the link and place inside `data/SPARF_images`. Then you can run the [notebook example](https://github.com/ajhamdi/sparf_pytorch/blob/main/examples/mvimage_load.ipynb). 


## SPARF Radiance Field Dataset
The dataset is released in the [link](https://drive.google.com/drive/folders/1Qd_hBrRKR1vlCacOSyK_FN4igkHSbPSM?usp=sharing). Each of SPARF's instances has (beside the posed images above) two directories: `STF` (RGB voxels) and `SRF` (Spherical Harmonics voxels). The full radiance fileds are available under `<CLASS_LABEL>/<INSTANCE_ID>/SRF/vox_<VOXEL-RESOLUTION>/full`, where `<VOXEL-RESOLUTION>` is the resolution (32, 128 or 512). The partial SRFs are stored in `<CLASS_LABEL>/<INSTANCE_ID>/STF/vox_<VOXEL-RESOLUTION>/partial` similarly. The partitioning (shards) and splits of the dataset is available on the file `SNRL_splits.csv` in the root of the dataset.  The voxles information are stored as sparse voxels in `data_0.npz`as coords and values. 

Download all content in the link and place inside `data/SPARF_srf`. Then you can run the [main training code](https://github.com/ajhamdi/sparf_pytorch/blob/main/run_sparf.py).

## Script for rendering ShapeNet images used in creating SPARF 
make sure that `ShapeNetCore.v2` is downloaded and placed in `data/ShapeNetCore.v2`. Then run the following script to render the images used in creating SPARF. 
```bash
python run_sparf.py --run render --data_dir data/SPARF_srf/ --nb_views 400 --object_class car 
```
## Script for extracting SPARF Radiance Fields (full SRFs with voxel res=128 and SH dim=4)
make sure that `SPARF_images` is downloaded and placed in `data/SPARF_images`. Then run the following script to extract the SRFs.
```bash
python run_sparf.py --run extract --vox_res 128 --sh_dim 4 --object_class airplane --data_dir data/SPARF_images/ --visualize --evaluate 
```

## Script for extracting SPARF Radiance Fields (partial SRFs with voxel res=512 and SH dim=1, nb_views=3)
make sure that `SPARF_images` is downloaded and placed in `data/SPARF_images`. Then run the following script to extract the SRFs.
```bash
python run_sparf.py --run preprocess --vox_res 512 --sh_dim 1 --rf_variant 0 --object_class airplane --nb_views 3 --data_dir data/SPARF_images/ --randomized_views
```
## Training and Inference pipeline on SPARF Radiance Fields
make sure that `SPARF_srf` is downloaded and placed in `data/SPARF_srf`. Then run the following script to train on SRFs.
```bash
python run_sparf.py --vox_res 128 --nb_views 3 --nb_rf_variants 4 --input_quantization_size 1.0 --strides 2 --lr_decay 0.99 --batch_size 6 --lr 1e-2 --visualize --normalize_input const --lambda_cls 30.0 --lambda_main 2.0 --augment_type none --mask_type densepoints --ignore_loss_mask --nb_frames 200 --validate_training  --data_dir data/SPARF_srf/ --run train --object_class airplane 
```

## Citation
If you find our work useful in your research, please consider citing:
```bibtex
@InProceedings{Hamdi_2023_ICCV,
    author    = {Hamdi, Abdullah and Ghanem, Bernard and Nie{\ss}sner, Matthias},
    title     = {SPARF: Large-Scale Learning of 3D Sparse Radiance Fields from Few Input Images},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2023},
    pages     = {2930-2940}
}
```

