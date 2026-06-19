# FlexiTac Isaac Sim HydroShear

This repository contains our current FlexiTac tactile simulation branch on top of Isaac Sim and Isaac Lab. The main line is:

1. Put a bump-pad tactile surface on the ALOHA gripper.
2. Compute normal and shear tactile readouts with HydroShear.
3. Validate the readout with controlled dynamic probes.
4. Replay recorded trajectories with tactile visualization.
5. Generate object-contact demo GIFs for visual comparison.

The environment exposes a Gymnasium-style `reset()` / `step()` API, but it runs on Isaac Sim through Isaac Lab. It is not the old standalone Isaac Gym simulator.

## Key Files

- `Isaacsim_tactile_env/viser_interface.py`: interactive Viser drag-control interface.
- `Isaacsim_tactile_env/reply_with_tactile.py`: replay recorded ALOHA trajectories.
- `Isaacsim_tactile_env/aloha/`: ALOHA Isaac Sim environment, robot/object/tactile adapters.
- `Isaacsim_tactile_env/sensors/warp_sdf_tactile/`: our custom Warp SDF tactile sensor, kept outside IsaacLab core.
- `Isaacsim_tactile_env/tactile/`: HydroShear geometry, readout, and shear model.
- `Isaacsim_tactile_env/assets/aloha_tactile_bump.urdf`: ALOHA robot with bump tactile pads.
- `Isaacsim_tactile_env/assets/meshes/aloha_bump_pad.obj`: current bump-pad mesh.
- `scripts/demo/demo_vedo.py`: four-object / gallery HydroShear GIF demo.

`Isaacsim_tactile_env/output/` is generated at runtime and is intentionally not tracked.

## Environment

Use an Isaac Sim / official Isaac Lab environment, not system Python. This repository no longer vendors the full IsaacLab source tree; `isaaclab` should come from your installed environment.

Activate the environment and check that official IsaacLab is already installed:

```bash
conda activate tactile_isaaclab
python -c "import isaaclab; print('isaaclab:', isaaclab.__file__)"
```

If that import fails, install the official IsaacLab wheel into the same environment:

```bash
conda activate tactile_isaaclab
pip install -U "isaaclab==2.3.2.post1" --extra-index-url https://pypi.nvidia.com
```

For Isaac Sim 5.1 / Python 3.11, `isaaclab==2.3.2.post1` is the tested version in this repository. Use the IsaacLab version that matches your Isaac Sim version if you change Isaac Sim. Official install docs:

- https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
- https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html

Then install the FlexiTac-side Python packages:

```bash
cd /home/jiafeng/FlexiTac-IsaacSim-Simulation
conda activate tactile_isaaclab
pip install viser "websockets>=13" vedo imageio open3d trimesh
```

If `torch` is missing, you are probably running the wrong Python. Prefer:

```bash
./isaaclab.sh -p <script.py> ...
```

or the Python inside the Isaac/conda environment.

## Interactive Dragging

Start the main Viser interface with the bump gripper and HydroShear backend:

```bash
./isaaclab.sh -p Isaacsim_tactile_env/viser_interface.py \
  --enable_cameras \
  --use_bump_pad
```

Then open:

```text
http://localhost:8080
```

In the browser:

- Drag the left or right end-effector gizmo to move the gripper.
- Use the gripper sliders to open/close the fingers.
- The right-side panels update live: tactile normal, shear X, shear Y, shear magnitude, and camera feed if cameras are enabled.
- Turn on `Show shear arrows` to see 3D shear vectors above the pads.

Headless server mode:

```bash
./isaaclab.sh -p Isaacsim_tactile_env/viser_interface.py \
  --headless \
  --use_bump_pad
```

For a real gripper or tactile pad, enter the real active bump area in `bump_active_width_mm` / `bump_active_length_mm` and the real bump spacing in `bump_pitch_mm`; use `bump_sim_active_width_mm` / `bump_sim_active_length_mm` only when the simulated pad mesh should be scaled differently from the real hardware.

## HydroShear Call Path

The Viser script only creates the config and displays the observations. The HydroShear backend is created inside the ALOHA environment:

```text
viser_interface.py
  cfg.tactile.backend = HydroShearTactileBackendCfg(...)

AlohaTactileEnv.__init__
  self._tactile_setup = AlohaTactileSetup(...)

AlohaTactileSetup.__init__
  self.backend = self._create_backend(...)

AlohaTactileSetup._create_backend
  HydroShearTactileBackend(...)

AlohaTactileSetup.__init__
  self.sensors, self.sensor_slot_order = self.backend.create_sensors(...)
  -> 4 HydroShearSensorState objects
```

Those four sensor states correspond to the four elastomer pads:

```text
left arm / left finger
left arm / right finger
right arm / left finger
right arm / right finger
```

After the scene is spawned, the backend is initialized:

```text
AlohaTactileEnv._post_spawn_init()
  self._tactile_setup.initialize_after_sim_reset(stage)

AlohaTactileSetup.initialize_after_sim_reset()
  self.backend.initialize_after_sim_reset(self.sensors, stage, self.target_tracker)
```

At every simulation step, tactile is updated through the environment:

```text
viser_interface.py
  obs, _, _, _, _ = env.step(action)

AlohaTactileEnv.step()
  self._sim.step(...)
  self._robot_manager.update(dt)
  self._objects.update(dt)
  self._tactile_setup.update(dt)
  obs = self._get_obs()

AlohaTactileSetup.update()
  self.backend.update(dt, self.sensors, self.target_tracker)

HydroShearTactileBackend.update()
  state.core.update(...)
```

The shear values are written into observations and then displayed by Viser:

```text
tactile/hydroshear.py
  observations["tactile_shear"] = readout.tactile_shear

aloha/tactile/backend.py
  obs["tactile_shear"] = stacked four-pad tactile shear grid

aloha/observation.py
  obs = {**tactile_out.observations, ...}

viser_interface.py
  tactile_shear = obs["tactile_shear"]
```

## HydroShear Shear Logic

The implementation has two levels:

1. Surface-point HydroShear: object surface samples produce per-point normal and shear state.
2. Readout: per-point or per-bump state is converted into a tactile grid observation.

For each object surface point, HydroShear first computes the contact displacement in elastomer frame:

```python
d_total = prev_points_e - current_points_e
d_contact = alpha * d_total
```

`alpha` is the fraction of the frame-to-frame segment that is inside contact. The shear logic then removes the normal component from `d_contact`.

If `normal_axis = 2`, the z axis is normal:

```text
d_contact = [0.03, -0.01, 0.005]
normal component = z = 0.005
d_tangent = [0.03, -0.01, 0.0]
```

If `normal_axis = 0`, the x axis is normal:

```text
d_contact = [0.03, -0.01, 0.005]
normal component = x = 0.03
d_tangent = [0.0, -0.01, 0.005]
```

This is the simplified version of the code in `Isaacsim_tactile_env/tactile/hydroshear.py`:

```python
d_n_scalar = normal_direction * d_contact[..., normal_axis]
d_tangent = d_contact.clone()
d_tangent[..., normal_axis] = 0.0
```

The shear state is recurrent. It keeps the previous shear, decays it, and adds the new tangential motion:

```python
shear_update_e = shear_decay * prev_shear + shear_stiffness * area_scale * d_tangent
```

Then it applies a Coulomb-style friction limit from the normal displacement:

```python
limit = friction_coefficient * normal_displacement
shear_force <= limit
```

In code, this clamp is:

```python
shear_norm = shear_candidate.norm(dim=-1)
shear_scale = min(1, limit / shear_norm)
shear_force_e = shear_candidate * shear_scale
```

When `bump_enabled=False`, each surface point keeps its own shear state, then `SurfacePointForceProjector` projects all surface-point shear values to the taxel grid with distance weights:

```python
weight = exp(-lambda_s * distance_to_taxel**2)
taxel_shear = sum(weight * surface_point_shear)
```

When `bump_enabled=True`, each surface point is first assigned to the nearest bump center in the tangential plane. For example, if z is the normal axis, assignment only compares x/y distance:

```python
diff = point_xy - bump_center_xy
bump_id = argmin(sum(diff**2))
```

The normal coordinate is not used for bump assignment; it is used separately for penetration and normal force. This means two points with the same x/y location but different z depths go to the same bump.

After assignment, `d_tangent` is aggregated per bump. With the default `weighted_mean` aggregation:

```text
bump_delta_shear =
  sum(d_tangent_i * area_i for points assigned to this bump)
  / sum(area_i for points assigned to this bump)
```

Example:

```text
p0 d_tangent = [0.02, 0.00, 0.00], area = 1
p1 d_tangent = [0.04, 0.02, 0.00], area = 2

bump_delta_shear =
([0.02, 0.00, 0.00] * 1 + [0.04, 0.02, 0.00] * 2) / 3
= [0.0333, 0.0133, 0.0]
```

Then bump shear updates with the same structure:

```python
bump_shear = shear_decay * prev_bump_shear + shear_stiffness * bump_delta_shear
bump_shear = clamp_by_friction_limit(bump_shear, friction_coefficient * bump_normal_force)
```

In bump mode, the bump grid is also the output grid: `_bump_readout()` reshapes each bump force into `(bump_rows, bump_cols, 3)`, where each cell is:

```text
[normal_force, shear_u, shear_v]
```

## Four-Object GIF Demo

The Vedo demo uses four mesh objects:

- `sphere`
- `cross`
- `cow`
- `torus`

The default mesh asset directory is:

```text
/home/jiafeng/hydroshear/demo_assets
```

It should contain:

```text
gsmini_elastomer.obj
sphere.obj
cross2_09x.stl
small_cow.stl
torus_7mm.stl
```

Quick command from the repository root:

```bash
cd /home/jiafeng/FlexiTac-IsaacSim-Simulation
conda activate tactile_isaaclab
mkdir -p Isaacsim_tactile_env/output/demo_gifs

for obj in sphere cross cow torus; do
  ./isaaclab.sh -p scripts/demo/demo_vedo.py \
    --asset-root /home/jiafeng/hydroshear/demo_assets \
    --object "$obj" \
    --use-bump \
    --motion-script press_four_way_spin \
    --frames 36 \
    --fps 12 \
    --num-points 100 \
    --initial-samples 3000 \
    --gif "Isaacsim_tactile_env/output/demo_gifs/${obj}_bump_four_way.gif"
done
```

Generate one GIF per object:

```bash
mkdir -p Isaacsim_tactile_env/output/demo_gifs

for obj in sphere cross cow torus; do
  ./isaaclab.sh -p scripts/demo/demo_vedo.py \
    --asset-root /home/jiafeng/hydroshear/demo_assets \
    --object "$obj" \
    --use-bump \
    --motion-script press_four_way_spin \
    --frames 36 \
    --fps 12 \
    --num-points 100 \
    --initial-samples 3000 \
    --gif "Isaacsim_tactile_env/output/demo_gifs/${obj}_bump_four_way.gif"
done
```

Generate the sliding/dragging version:

```bash
mkdir -p Isaacsim_tactile_env/output/demo_gifs

for obj in sphere cross cow torus; do
  ./isaaclab.sh -p scripts/demo/demo_vedo.py \
    --asset-root /home/jiafeng/hydroshear/demo_assets \
    --object "$obj" \
    --use-bump \
    --motion-script press_slide \
    --frames 36 \
    --fps 12 \
    --num-points 100 \
    --initial-samples 3000 \
    --gif "Isaacsim_tactile_env/output/demo_gifs/${obj}_bump_slide.gif"
done
```

Optional combined gallery GIF:

```bash
./isaaclab.sh -p scripts/demo/demo_vedo.py \
  --asset-root /home/jiafeng/hydroshear/demo_assets \
  --gallery \
  --use-bump \
  --motion-script press_four_way_spin \
  --frames 36 \
  --fps 12 \
  --num-points 100 \
  --initial-samples 3000 \
  --gif Isaacsim_tactile_env/output/demo_gifs/gallery_bump_four_way.gif
```

Note: `--gallery` currently includes the mesh objects plus procedural objects available in `demo_vedo.py`.

## Replay Validation

Replay the recorded trajectory with HydroShear:

```bash
./isaaclab.sh -p Isaacsim_tactile_env/reply_with_tactile.py \
  --dataset_npz Isaacsim_tactile_env/data/dataset_train.npz \
  --normalization_pth Isaacsim_tactile_env/data/dataset_normalizer.npz \
  --episode_idx 3 \
  --replay_key joint_states \
  --steps_per_frame 3
```

## Dynamic Probe Validation

Controlled probe scripts do not use robot IK; they isolate the HydroShear readout:

```bash
./isaaclab.sh -p Isaacsim_tactile_env/tools/visualize_hydroshear_dynamic_axis_probe.py \
  --use-bump \
  --output_dir Isaacsim_tactile_env/output/hydroshear_dynamic_axis_probe

./isaaclab.sh -p Isaacsim_tactile_env/tools/visualize_hydroshear_cube_axis_probe.py \
  --use-bump \
  --output_dir Isaacsim_tactile_env/output/hydroshear_cube_axis_probe
```

For quick regression checks:

```bash
./isaaclab.sh -p Isaacsim_tactile_env/tools/check_hydroshear_core.py
```

## Cleanup

Generated artifacts can be removed safely:

```bash
rm -rf Isaacsim_tactile_env/output
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete
```

Do not delete these for the current main line:

- `Isaacsim_tactile_env/aloha`
- `Isaacsim_tactile_env/sensors/warp_sdf_tactile`
- `Isaacsim_tactile_env/tactile`
- `Isaacsim_tactile_env/assets/aloha_tactile_bump.urdf`
- `Isaacsim_tactile_env/assets/meshes/aloha_bump_pad.obj`
