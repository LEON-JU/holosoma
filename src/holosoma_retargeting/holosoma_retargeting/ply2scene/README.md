# ply2scene：从场景点云生成 MuJoCo 场景（方案设计）

本目录用于后续实现一个“场景点云 -> Open3D 网格 -> `.obj` + `.urdf` + MuJoCo `.xml`”的转换链路。

当前阶段只输出设计文档，用于指导后续代码编写。

---

## 1. 背景与目标

在 `holosoma_retargeting/ground_alignment.py` 中，已经提供了从场景重建结果投影并拼接点云的能力：

- 通过 RGB + 深度 + 相机位姿投影，并跨帧拼接：`ground_alignment.load_scene_pointcloud_from_rgbd(...)`
- 坐标系转换（OpenCV -> Z-up）：`ground_alignment.opencv_to_z_up(...)`
- 应用 4x4 刚体变换（地面对齐）：`ground_alignment.apply_transform_matrix(...)`
- 交互式地面对齐并导出变换矩阵：`ground_alignment.main(config_dict)`（由 `pipeline.py` 调用）

下一步希望：

1) 从某个 Holosoma run 的 RGBD 重建结果出发，复用已有函数拼接点云；随后使用 Open3D 做表面重建，得到三角网格并导出 `.obj`；  
2) 自动生成一个最小可用的静态场景 URDF（用于 Viser/yourdfpy 可视化）；  
3) 自动生成一个最小可用的 MuJoCo MJCF XML（用于 MuJoCo 仿真/评估加载）；  
4) 参考现有 Viser 可视化脚本（例如 `viser_player_recon.py`），写一个“简化版”脚本用于检查转换质量（点云 vs 网格 vs XML）。

---

## 2. 输入与输出约定

### 2.1 输入（以 Holosoma run 为中心）

本方案不再重新做对齐（不调用交互式 `ground_alignment.main`），只读取 pipeline 在 run 目录下产出的配置/变换，并据此生成用于转换的点云。

**必需输入**

- `seq`、`robot`（以及可选的 `manual` 路径配置），用 `holosoma_retargeting.config.get_sequence_paths(...)` 定位 run 目录与各类 artifact：
  - `paths.pipeline_config_json`：pipeline 总结（含 `scale_factor` 等展示/缩放信息）
  - `paths.ground_transform_json`：地面对齐变换矩阵 `T_align`（4x4；由 pipeline 的 ground alignment 步骤产出）
  - `paths.predicted_dir`：逐帧 RGB/Depth/Mask/Pose
  - `paths.depth_recovered`：逐帧 scale factor（用于 scene 尺度恢复）
  - `paths.results_dir`：每帧 `smplx_params.pt`（用于读取 `cam_int`）

**点云生成方式（固定：从 RGBD 重建结果拼接）**

- 调用：`ground_alignment.load_scene_pointcloud_from_rgbd(data_dir, depth_dir, smplx_results_dir, ...)`
- 按照 `viser_player_recon.py` 的逻辑做坐标与对齐转换：
  1) `scene_points = ground_alignment.opencv_to_z_up(scene_points_cv)`
  2) `scene_points = ground_alignment.apply_transform_matrix(scene_points, T_align)`

说明：`load_scene_pointcloud_from_rgbd` 内部会基于 `depth_recovered/*_scale_factor.txt` 计算平均尺度恢复系数，并对深度与相机位姿平移做缩放；因此拼接点云已经完成“scene scale recovery”。

### 2.2 输出（目标产物）

建议为每个场景输出一个“scene 包”，目录结构参考仓库内已有 `models/largebox`、`demo_data/climb/*` 的组织方式：

```
<scene_out_dir>/
  meshes/
    scene_visual.obj
    scene_collision.obj              # 可选：更简化的碰撞网格
  scene.urdf
  scene.xml                          # MuJoCo：场景单体 XML（可独立 load）
  scene_assets.xml                   # 可选：mujocoinclude 形式（用于 include）
  scene_body.xml                     # 可选：mujocoinclude 形式（用于 include）
  meta.json                          # 可选：记录参数、尺度、对齐矩阵、统计信息
```

`scene.xml` / `scene_assets.xml` / `scene_body.xml` 三者并非都必须实现；建议先实现 `scene.xml`，再根据与机器人 XML 合并的需求扩展 include 版本。

---

## 3. 坐标系与尺度（非常关键）

### 3.1 坐标系

- MuJoCo 常用 **Z-up**；本项目中 `ground_alignment` 明确提供 OpenCV->Z-up 转换：
  - `opencv_to_z_up(xyz)`: OpenCV (x right, y down, z forward) -> Z-up (x forward, y left, z up)
- 因此，**生成的网格与 XML 都应以 Z-up 坐标系为准**。

### 3.2 场景尺度：两类 scale 不要混淆

本仓库同时存在两种“尺度概念”：

1) **场景尺度恢复（scene scale recovery）**  
   - 来源：`depth_recovered/*_scale_factor.txt`  
   - 在 `ground_alignment.load_scene_pointcloud_from_rgbd` 内用于把深度（米）和相机位姿平移恢复到真实尺度。  

2) **机器人/人尺度（retarget scale_factor）**  
   - 来源：`pipeline.py` 中 `get_scale_factor(args.robot, args.human_height_m)`，并写入 `paths.pipeline_config_json`。  
   - 在 `viser_player_recon.py` 中用于展示：`scene_points *= scale_factor`（把“人尺度场景”缩放到“机器人尺度场景”）。

对 `ply2scene` 的建议：

- 输入点云来自 RGBD 拼接（内部做 scene scale recovery），再应用 `T_align` 完成“地面对齐”，可认为处于“对齐后的重建尺度/人尺度”。
- 若最终要在 MuJoCo 里跑“机器人尺度”的运动，**场景网格需要统一缩放**，两种可选实现：
  - **方式 1：直接缩放顶点**（写入 `scene_visual.obj` 前把顶点乘以 `scale_factor`）
  - **方式 2：用 MuJoCo `<mesh scale="...">` 和 URDF `<mesh scale="...">`**（顶点保持原始，人/真实尺度；加载时缩放）

建议优先使用 **方式 2**：更便于复现与调参，并且能在 `meta.json` 中清晰记录“原始尺度 + 加载缩放”。

---

## 4. 点云 -> 网格（Open3D）建议流程

### 4.1 点云清洗与降采样

输入点云通常很密（几十万到数百万点），直接重建会慢且容易出现伪面片。

推荐的最小处理链路：

1) `voxel_down_sample(voxel_size)`：体素降采样（例如 `0.02~0.05m`）
2) `remove_statistical_outlier(nb_neighbors, std_ratio)`：去离群点（可选）
3) `estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))`
4) `orient_normals_consistent_tangent_plane(k)`：尽量统一法线朝向（Poisson 重建前很重要）
5) （可选）裁剪：仅保留 ROI / z 轴范围，避免远处稀疏点造成 Poisson 填洞

### 4.2 表面重建算法选择

#### 方案 A：Poisson 重建（推荐起步）

- 优点：对噪声更鲁棒、能生成相对连续的表面
- 缺点：会“脑补”稀疏区域，容易生成浮空/外扩伪面片，需要做密度过滤与裁剪

Open3D API 参考：

- `mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=...)`
- 密度过滤：丢弃低密度顶点（例如保留 `densities > quantile(densities, 0.05)`）
- 之后 `mesh.remove_degenerate_triangles()`、`mesh.remove_duplicated_triangles()`、`mesh.remove_non_manifold_edges()`

#### 方案 B：Ball Pivoting（BPA）

- 优点：不强制闭合，适合“表面采样较均匀”的点云
- 缺点：对点间距敏感，需要估计采样半径列表

Open3D API 参考：

- `mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)`

建议策略：先用 Poisson 跑通全流程，再考虑 BPA/混合策略。

### 4.3 网格后处理（建议至少做两件事）

1) **裁剪到点云的包围盒**：减少 Poisson 外扩
2) **简化（Decimation）**：生成“可用于碰撞”的低面数版本

输出两份网格的动机：

- `scene_visual.obj`：面数较高，用于渲染、离线检查
- `scene_collision.obj`：面数更低/更平滑，用于 MuJoCo 接触计算（更快、更稳定）

---

## 5. MuJoCo XML 生成策略

### 5.1 最小可用的“场景单体” XML（推荐先实现）

目标：`mujoco.MjModel.from_xml_path(scene.xml)` 可以直接加载。

结构参考 `models/largebox/largebox.xml`：

- `<compiler meshdir="...">`：建议使用相对路径，指向 `meshes/`
- `<asset><mesh ... file="scene_visual.obj" scale="..."/></asset>`
- `<worldbody><geom type="mesh" mesh="scene_mesh" .../></worldbody>`

静态场景一般不需要 `<freejoint/>`（与 `g1_29dof_w_largebox.xml` 中“可动物体”的写法相反）。

碰撞与显示建议拆开（同一个 mesh 两个 geom）：

- visual geom：`contype="0" conaffinity="0"`
- collision geom：`rgba="0 0 0 0"`（或 `group` 隐藏）、`contype="1" conaffinity="1"`

### 5.2 include 版本（可选扩展）

当你需要把场景嵌入到某个机器人 XML 时，可参考 `demo_data/climb/*/box_assets.xml` 和 `box_body.xml`：

- `scene_assets.xml`：`<mujocoinclude> ... <mesh .../> ... </mujocoinclude>`
- `scene_body.xml`：`<mujocoinclude> ... <body/geom .../> ... </mujocoinclude>`

然后在机器人 XML 里：

- `<asset> ... <include file="scene_assets.xml"/> ... </asset>`
- `<worldbody> ... <include file="scene_body.xml"/> ... </worldbody>`

这种方式更利于组合与复用，也符合仓库中 climbing 任务的组织方式。

---

## 6. URDF 生成策略（用于 Viser/yourdfpy）

现有可视化脚本（`viser_player.py`、`viser_player_recon.py`、`src/interaction_mesh_retargeter.py`）均通过：

- `yourdfpy.URDF.load(..., load_meshes=True, build_scene_graph=True)`
- `viser.extras.ViserUrdf`

因此 URDF 的目标是“可被 yourdfpy 正确加载并显示”。

最小方案：单 link，无关节：

- `<robot name="scene">`
- `<link name="scene_link">`
  - `<visual><geometry><mesh filename="meshes/scene_visual.obj" scale="..."/></geometry></visual>`
  - `<collision><geometry><mesh filename="meshes/scene_collision.obj" .../></geometry></collision>`（可选）

注意：URDF 中 mesh 路径通常相对 URDF 文件目录解析；建议把 obj 放在 `meshes/` 下与模板保持一致。

---

## 7. 转化质量可视化（简化版 Viser 脚本设计）

目标：快速确认“点云 -> 网格 -> XML”是否一致（坐标、尺度、对齐、几何质量）。

### 7.1 建议的可视化内容

1) 点云：从 run 的 RGBD 重建结果拼接得到（`load_scene_pointcloud_from_rgbd` + `opencv_to_z_up` + `T_align`），再 `server.scene.add_point_cloud`
2) 网格：加载 `scene_visual.obj`（Open3D 或 trimesh 读 mesh + `server.scene.add_mesh_simple`）
3) （可选）从 MuJoCo XML 反解 mesh：
   - `mujoco.MjModel.from_xml_path(scene.xml)`
   - `mujoco.mj_forward(model, data)`
   - 遍历 `geom_type == mjGEOM_MESH`，用 `holosoma_retargeting/src/mujoco_utils._world_mesh_from_geom` 取世界系 V/F
   - 用 `add_mesh_simple` 画出来，与 obj 直接加载结果对比（用于排查 XML scale/pos/quat 问题）

### 7.2 建议的交互控件

- checkbox：显示/隐藏 point cloud、mesh、mujoco-mesh
- slider：mesh 透明度、point size
- checkbox：是否应用 `scale_factor`（与 `viser_player_recon.py` 的展示逻辑对齐）

### 7.3 CLI 形态（后续实现）

建议用 `tyro`（仓库已有使用）定义：

- `--seq` / `--robot` / `--manual`（对齐 `viser_player_recon.py`，用 `get_sequence_paths` 自动定位 run）
- `--scene_obj` / `--scene_xml`（用于加载生成产物做对比）
- `--scale_factor`（默认从 `paths.pipeline_config_json` 自动读取，也允许手动覆盖）
- `--max_points`（下采样展示）

---

## 8. 建议的后续代码结构（实现阶段参考）

建议在 `ply2scene/` 里放两类脚本：

1) 转换器：`ply2scene/convert.py`
   - `Ply2SceneConfig`（tyro dataclass）
   - `load_points(...)`：从 RGBD 重建结果拼接（复用 `ground_alignment.load_scene_pointcloud_from_rgbd`），再做 `opencv_to_z_up` + `T_align`
   - `reconstruct_mesh(...)`：Poisson/BPA + 后处理
   - `export_obj(...)`、`export_urdf(...)`、`export_mjcf(...)`
   - `export_meta(...)`

2) 质量检查 viewer：`ply2scene/viser_preview.py`
   - 独立运行；对齐 `viser_player_recon.py` 的视觉风格与交互习惯

（可选）模板目录：`ply2scene/templates/`（存放 MJCF/URDF 的最小模板，避免在代码里硬编码长 XML）

---

## 10. 运行方式

以下命令以 `hsretargeting` 环境为例，且不使用 `conda run`。

### 10.1 生成场景（RGBD -> mesh -> obj/urdf/mjcf）

```bash
cd /home/juyiang/data/holosoma/src/holosoma_retargeting
conda activate hsretargeting
export PYTHONPATH=/home/juyiang/data/holosoma/src/holosoma_retargeting

python -m holosoma_retargeting.ply2scene.convert \
  --seq smooth --robot g1
```

### 10.2 预览检查（点云 vs OBJ vs MJCF）

```bash
cd /home/juyiang/data/holosoma/src/holosoma_retargeting
conda activate hsretargeting
export PYTHONPATH=/home/juyiang/data/holosoma/src/holosoma_retargeting

python -m holosoma_retargeting.ply2scene.viser_preview \
  --seq smooth --robot g1 \
  --scene_obj /path/to/scene_visual.obj \
  --scene_xml /path/to/scene.xml
```

## 9. 常见坑与规避建议

- **法线质量决定 Poisson 质量**：voxel size、normal radius、orient normals 的参数需要与点间距匹配。
- **Poisson 伪面片**：务必做密度过滤 + 裁剪到 AABB/OBB。
- **面数过大导致 MuJoCo 很慢**：优先做 decimation，必要时单独准备 collision mesh。
- **OBJ 不保真颜色**：目前以几何为主；如果后续需要纹理/颜色，可再讨论 UV baking 或基于材质的简化。
- **坐标/尺度一致性**：建议在 `meta.json` 里显式记录：
  - 输入来源（RGBD 拼接）以及坐标系假设（opencv->z-up）
  - 是否应用 `T_align`（来自 `paths.ground_transform_json`）
  - `scale_factor` 的使用方式（顶点缩放 or XML/URDF scale）
