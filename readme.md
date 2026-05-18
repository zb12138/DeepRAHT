# DeepRAHT: Learning Predictive RAHT for Point Cloud Attribute Compression 
### [AAAI 26](https://arxiv.org/abs/2601.12255) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qYqlAt6rdJzR9yBrnL8mFFgCX1O3hwby)
## Install

Python3.8 and cuda 11.x is essentially required

```bash
conda create -n py38 python=3.8.20
conda activate py38

# if cuda-11.3 installed
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# if cuda-11.7 installed
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# to install MinkowskiEngine (ref https://nvidia.github.io/MinkowskiEngine/overview.html#anaconda) 
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
conda install openblas-devel -c anaconda
# export CONDA_PREFIX=$(dirname $(dirname $(which python)))
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
# other packages
pip install git+https://github.com/zb12138/ptio.git # ptio
pip install bitstring
# pip install loguru cprint torchac h5py plyfile tensorboard ninja
# pip install plyfile open3d==0.18.0
```

Reference for the compilation (```exe/DeepRAHT```) platform
```
DeepRAHT version 25.7.1
==========System==========
Linux-4.4.0-176-generic-x86_64-with-glibc2.17
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=18.04
DISTRIB_CODENAME=bionic
DISTRIB_DESCRIPTION="Ubuntu 18.04.4 LTS"
3.8.20 (default, Oct  3 2024, 15:24:27) 
[GCC 11.2.0]
==========Pytorch==========
1.12.1+cu113
torch.cuda.is_available(): True
==========NVIDIA-SMI==========
/usr/bin/nvidia-smi
Driver Version 470.256.02
CUDA Version 11.4
==========NVCC==========
/usr/local/cuda/bin/nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_May__3_18:49:52_PDT_2022
Cuda compilation tools, release 11.7, V11.7.64
Build cuda_11.7.r11.7/compiler.31294372_0
==========CC==========
/usr/bin/c++
c++ (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
==========MinkowskiEngine==========
0.5.4
MinkowskiEngine compiled with CUDA Support: True
NVCC version MinkowskiEngine is compiled: 11070
CUDART version MinkowskiEngine is compiled: 11070
```

## Run test model
```bash
cd exe
export LD_LIBRARY_PATH=$(dirname $(dirname $(which python)))/lib:$LD_LIBRARY_PATH
chmod +x ./DeepRAHT
./DeepRAHT -h
./DeepRAHT -ply loot_vox10_1200.ply -mode=0 -qs=8 # compress
./DeepRAHT -ply loot_vox10_1200.ply -mode=1 -qs=8 # decompress
# 1/1 compressing loot_vox10_1200.ply to bin/loot_vox10_1200.bin
# psnr [43.8363946  52.18927304 52.24188538] bpp 0.5038 time 4.0916 attributes shape (805285, 3)
```
If the block chunking is enabled, it will generate the *.binBk as binfiles; But *.bin (not *.binBk) is still required as input at decoding. e.g., 

```bash
./DeepRAHT -ply loot_viewdep_vox12.ply -mode=0 -qs=8 -bin loot/test.bin
# partition loot_viewdep_vox12.ply to 3 blocks
# 1/3 compressing loot_viewdep_vox12.ply to loot/test.bin0
# 2/3 compressing loot_viewdep_vox12.ply to loot/test.bin1
# 3/3 compressing loot_viewdep_vox12.ply to loot/test.bin2
# psnr [46.24046535 54.4028619  54.34530696] bpp 0.2425 time 8.2284 attributes shape (3017285, 3)
./DeepRAHT -ply loot_viewdep_vox12.ply -mode=1 -qs=8 -bin loot/test.bin
# partition loot_viewdep_vox12.ply to 3 blocks
# 1/3 decompressing loot/test.bin0 to loot_viewdep_vox12_recon_qs8.ply
# 2/3 decompressing loot/test.bin1 to loot_viewdep_vox12_recon_qs8.ply
# 3/3 decompressing loot/test.bin2 to loot_viewdep_vox12_recon_qs8.ply
# psnr [46.24046535 54.4028619  54.34530696] bpp 0.2425 time 8.3533 attributes shape (3017285, 3)
```

Some suggestions for handling errors

fuse: device not found, try 'modprobe fuse' first  
Cannot mount AppImage, please check your FUSE setup.
You might still be able to extract the contents of this AppImage
if you run it with the --appimage-extract option.  
(You might have encountered issues due to using Docker. We will decompress run the application.)
```bash
sudo apt install -y libfuse2 # and try again
```
or
```bash
cd exe
./DeepRAHT --appimage-extract
cd squashfs-root/bin
export LD_LIBRARY_PATH=$pwd:$LD_LIBRARY_PATH
./DeepRAHT -h
```
error while loading shared libraries: libpython3.8.so.1.0: cannot open shared object file: No such file or directory  
(Make sure you are using py38)
```bash
# add the lib path to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(dirname $(dirname $(which python)))/lib:$LD_LIBRARY_PATH
```

ModuleNotFoundError: No module named 'torch' (if installed)
```bash
# add the packages path to PYTHONPATH
export PYTHONPATH=$(dirname $(dirname $(which python)))/lib/python3.8/site-packages:$PYTHONPATH
export PYTHONPATH=$(dirname $(dirname $(which python)))/lib/python3.8/site-packages/MinkowskiEngine-0.5.4-py3.8-linux-x86_64.egg:$PYTHONPATH
```


## Dataset
Download data and copy them into Data. 
```
Data/
├── MPEGCAT1
│   ├── Arco_Valentino_Dense_vox12.ply
│   ├── Staue_Klimt_vox20.ply
│   └── ...
├── 8iVSLF/Static
│   ├── boxer_viewdep_vox12.ply
│   ├── longdress_viewdep_vox12.ply
│   ├── loot_viewdep_vox12.ply
│   ├── redandblack_viewdep_vox12.ply
│   ├── soldier_viewdep_vox12.ply
│   └── Thaidancer_viewdep_vox12.ply
├── RWTT/point_cloud
│    ├──RWT1_scene_dense_mesh_refine_texture_vox10.ply
│    └── ...
└── Owlii
    ├── basketball_player_vox11/basketball_player_vox11_00000001.ply
    ├── dancer_vox11/*
    ├── exercise_vox11/*
    └── model_vox11/*
```
## Citation
```
@article{DeepRAHT, title={DeepRAHT: Learning Predictive RAHT for Point Cloud Attribute Compression}, volume={40},
 url={https://ojs.aaai.org/index.php/AAAI/article/view/38493}, DOI={10.1609/aaai.v40i17.38493}, number={17},
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Fu, Chunyang and Qin, Tai and Wang, Shiqi and Li, Zhu},
year={2026}, month={Mar.}, pages={14738–14746} }
```
