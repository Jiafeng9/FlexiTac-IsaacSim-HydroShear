# IsaacSim Tactile Environment for FlexiTac

**Authors:**  
Binghao Huang¹, Yunzhu Li¹  
¹Columbia University

FlexiTac is an open-source, scalable tactile sensing platform designed to make touch sensing more accessible for robotics. It supports flexible sensor fabrication, real-time tactile readout, and integration into manipulation systems such as grippers, robot arms, and tactile skins. The broader FlexiTac project also includes hardware tutorials, simulation support, and system design examples for robotic applications.

**Project website:**  [https://flexitac.github.io/](https://flexitac.github.io/)


To build the tactile sensors, follow the hardware assembly tutorial below:
[**Hardware Assembly Tutorial**](https://docs.google.com/document/d/1bvz6AL7BUkhj4Dj7n9DFXTjnGIX4-ziN8-smiCKpVZU/edit?usp=sharing)


## Overview of This Repo

This repository contains tools for running FlexiTac in IsaacSim, including:

- **trajectory replay with tactile visualization**
- **a browser-based Viser interface for interactive control**


![Demo](isaacsim.gif)

## 🛠️ Installation

### 1. Install Isaac Sim
Skip this step if Isaac Sim is already installed.

Follow the official NVIDIA documentation:  
[Isaac Sim Installation Guide](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/quick-install.html)

Download Isaac Sim 5.1.0 or later:  
   [Isaac Sim 5.1.0 Linux Download](https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip)

Extract the downloaded archive into an `isaac-sim` directory.
On Linux, run:

```bash
./post_install.sh
./isaac-sim.selector.sh
```

### 2. Clone this repository

```bash
git clone https://github.com/binghao-huang/FlexiTac-IsaacSim-Simulation.git
cd FlexiTac-IsaacSim-Simulation
```

### 3. Install IsaacLab and dependencies

Set up IsaacLab by following the official installation guide:  
[IsaacLab pip installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)

Create and activate a conda environment:

```bash
conda create -n tactile_isaaclab python=3.11
conda activate tactile_isaaclab
```

Install Isaac Sim pip packages:

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

Install a CUDA-enabled PyTorch build that matches your system architecture:

```bash
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

Verifying the Isaac Sim installation:

```bash
isaacsim
```

Run the install command that iterates over all the extensions in source directory and installs them using pip (with --editable flag):

```bash
cd ~/FlexiTac-IsaacSim-Simulation
./isaaclab.sh --install # or "./isaaclab.sh -i"
```


Install additional dependencies:

```bash
pip uninstall -y opencv-python-headless
pip install opencv-python==4.11.0.86
pip install viser "websockets>=13"
```

## 🖥️ Replay a Recorded Trajectory

Launch the tactile replay viewer with the default WarpSDF normal tactile backend:

```bash
./isaaclab.sh -p Isaacsim_tactile_env/reply_with_tactile.py \
    --dataset_npz Isaacsim_tactile_env/data/dataset_train.npz \
    --normalization_pth Isaacsim_tactile_env/data/dataset_normalizer.npz \
    --episode_idx 3 \
    --replay_key joint_states \
    --steps_per_frame 3
```

Run the same replay with the surface-point HydroShear backend:

```bash
./isaaclab.sh -p Isaacsim_tactile_env/reply_with_tactile.py \
    --dataset_npz Isaacsim_tactile_env/data/dataset_train.npz \
    --normalization_pth Isaacsim_tactile_env/data/dataset_normalizer.npz \
    --episode_idx 3 \
    --replay_key joint_states \
    --steps_per_frame 3 \
    --tactile_backend surface_hydro
```

Replay options:

- `--tactile_backend normal`: use the original WarpSDF normal-force tactile grid.
- `--tactile_backend taxel_shear`: use WarpSDF normal force plus a taxel-level shear baseline.
- `--tactile_backend surface_hydro`: use the surface-point HydroShear marker-field backend.
- `--compare_hydro_normal`: additionally show a HydroShear output beside the selected main tactile backend.

## HydroShear Vedo GIF Demos

Render the Vedo object gallery with the per-bump HydroShear readout and four-way rotating press motion:

```bash
/home/jiafeng/miniconda3/envs/tactile_isaaclab/bin/python scripts/demo/demo_vedo.py \
    --gallery \
    --use-bump \
    --motion-script press_four_way_spin \
    --frames 36 \
    --fps 12 \
    --num-points 100 \
    --initial-samples 3000 \
    --gif Isaacsim_tactile_env/output/hydroshear_gallery_bump_four_way_rotation.gif
```

Render the dragging/sliding bump demo with the same bump HydroShear backend:

```bash
/home/jiafeng/miniconda3/envs/tactile_isaaclab/bin/python scripts/demo/demo_vedo.py \
    --gallery \
    --use-bump \
    --motion-script press_slide \
    --frames 36 \
    --fps 12 \
    --num-points 100 \
    --initial-samples 3000 \
    --gif Isaacsim_tactile_env/output/hydroshear_gallery_bump_dragging.gif
```

## 🦾 Launch the Interactive Web Interface

Start the Viser-based teleoperation interface with the default WarpSDF normal tactile backend:

```bash
./isaaclab.sh -p Isaacsim_tactile_env/viser_interface.py \
    --headless \
    --enable_cameras
```

Start Viser with the surface-point HydroShear tactile backend:

```bash
./isaaclab.sh -p Isaacsim_tactile_env/viser_interface.py \
    --headless \
    --enable_cameras \
    --tactile_backend surface_hydro
```

Start Viser with the generated bump-pad gripper and per-bump HydroShear readout for interactive dragging:

```bash
./isaaclab.sh -p Isaacsim_tactile_env/viser_interface.py \
    --enable_cameras \
    --tactile_backend surface_hydro \
    --use_bump_pad
```

Viser options:

- `--tactile_backend normal`: use the original WarpSDF normal-force tactile grid.
- `--tactile_backend taxel_shear`: use WarpSDF normal force plus a taxel-level shear baseline.
- `--tactile_backend surface_hydro`: use the surface-point HydroShear marker-field backend.
- `--use_bump_pad`: use the generated bump-pad ALOHA URDF and enable the per-bump HydroShear readout when paired with `--tactile_backend surface_hydro`.
- `--compare_hydro_normal`: add a separate HydroShear comparison panel.
- `--hydro_normal_scale`: scale the HydroShear normal channel in comparison mode.
- `--hydro_shear_scale`: scale the HydroShear shear channels in comparison mode.
- `--hydro_shear_stiffness`: override the HydroShear tangential stiffness in comparison mode.

Then open the following in your browser:

```text
http://localhost:8080
```

![Demo](web_demo.gif)


# Acknowledgements

We thank Jimmy Wang, Yifan He, Xuihui Kang, Yuhao Zhou, Jie Xu, Iretiayo Akinola
, Yu-Wei Chao, Siyu Ma, and Chang Yu for their contribution, valueable feedback and suggestions on this simulation environment.
