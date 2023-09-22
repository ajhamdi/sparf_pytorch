## Installation


## Requirements

you need a Cuda `+11.1` installed on your system (check with command `nvcc --version` ). You also need `gcc --version` to be `>= 8.2.0`. Then follow the following steps:

1. install `Plenxels` and `Minkowski Engine` (depending on your system from [here](https://nvidia.github.io/MinkowskiEngine/quick_start.html) and [here](https://github.com/sxyu/svox2) ). 
Alternatively you can follow the following steps:where `CONDA_PREFIX` is the path to your conda environment.
```bash
conda create --name sparf0 python=3.6
conda activate sparf0
conda install numpy pytorch=1.9.0 torchvision cudatoolkit=11.1 openblas-devel open3d=0.9.0 pytorch-lightning pyyaml jupyterlab -c open3d-admin -c conda-forge -c anaconda -c pytorch -c nvidia
git clone https://github.com/NVIDIA/MinkowskiEngine
cd MinkowskiEngine
python setup.py install --blas=openblas --blas_include_dirs=${CONDA_PREFIX}/include
cd ..
pip install imageio imageio-ffmpeg ipdb lpips opencv-python Pillow pyyaml tensorboard imageio imageio-ffmpeg PyMCubes moviepy matplotlib scipy wandb pandas trimesh pyglet einops pyhocon ConfigArgParse timm dotmap pretrainedmodels scikit-image ipdb tqdm ipyplot 
cd Svox2
pip install .
python3 -m pip install pyvirtualdisplay # optional
``` 
