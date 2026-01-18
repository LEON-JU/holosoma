# Robot Retarget from Her (Pipeline)

This README describes how to run the Her (Parkour) retargeting pipeline and how to visualize results.

## 1) Quick Start

Run from the package directory:

```bash
python -m holosoma_retargeting.pipeline --seq smooth --robot g1
```

## 2) Config: `config.py`

Edit `config.py` (MANUAL CONFIGURATION section) to match your machine:

- `results_data_dir`: Human mesh output folder, contains SMPL-X results 
- `her_parkour_dir`: Front-End output folder 
- `smpl_model_path`: SMPL/SMPL-X model folder
- `runs_root`: output workspace for pipeline

You can also override via CLI (Tyro nested args), for example:

```bash
python -m holosoma_retargeting.pipeline --seq smooth --robot g1 \
  --manual.results-data-dir /path/to/results \
  --manual.her-parkour-dir /path/to/Her_data/Parkour \
  --manual.smpl-model-path /path/to/SMPL_models/models \
  --manual.runs-root /path/to/holosoma_runs
```

## 3) Pipeline Behavior

Pipeline steps:

1. **Ground alignment (optional UI)**  
   - Uses RGB-D + predicted masks to build a scene cloud.
   - Interactive plane fit with Viser.
   - Saves `transform.json` to a stable artifact path.

2. **Prepare retarget data**  
   - Converts SMPL-X results to InterMimic-style `.pt`.
   - Applies ground transform and recovered scene scale.

3. **Robot retarget**  
   - Runs `examples/robot_retarget.py`.
   - Applies custom scale factor (robot height vs human height).

4. **Save run summary**  
   - Writes `pipeline_config.json` for downstream tools (e.g., visualizer).

### Common flags

- `--seq`: `smooth` | `wall_smooth` | `rooftop_smooth`
- `--robot`: `g1` | `t1`
- `--human-height-m`: default `1.7`
- `--reuse-existing-transform` / `--no-reuse-existing-transform`
- `--run-ground-alignment` / `--no-run-ground-alignment`

If `--reuse-existing-transform` is true and the transform exists, the UI step is skipped.

## 4) Outputs

All outputs are written under:

```
{runs_root}/{seq}/{robot}/
```

Key files:

- `artifacts/ground_alignment/transform.json`  
  Ground alignment 4x4 matrix.

- `inputs/retarget_data/{seq}.pt`  
  InterMimic-style motion data (after alignment + scene scale recovery).

- `outputs/retarget/{seq}.npz`  
  Retargeted robot motion (`qpos`, `fps`).

- `pipeline_config.json`  
  Run summary (paths, scale_factor, vertical_bias_z_min_*).

Additional file:

- `{her_parkour_dir}/{scene_name}/aligned_scene_manual_{seq}.ply`  
  Optional aligned scene PLY saved from ground alignment UI.

## 5) Visualizer (Reconstruction + Retarget)

Run from the package directory:

```bash
python -m holosoma_retargeting.viser_player_recon --seq smooth --robot g1
```

Notes:

- The visualizer reads `transform.json` and `pipeline_config.json` automatically.
- It loads the retargeted motion from `outputs/retarget/{seq}.npz`.
- Point clouds are downsampled by default for remote use.

Optional flags (examples):

```bash
python -m holosoma_retargeting.viser_player_recon --seq smooth --robot g1 --scene-source rgbd
python -m holosoma_retargeting.viser_player_recon --seq smooth --robot g1 --qpos-npz /path/to/custom.npz
python -m holosoma_retargeting.viser_player_recon --seq smooth --robot g1 --robot-urdf models/g1/g1_29dof.urdf
```

If you change `runs_root` or other manual paths, pass the same overrides here as in the pipeline.
