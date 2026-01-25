 # ply2scene 升级方案（按 VideoMimic 论文的环境网格流程）

本文档基于 `ply2scene/reference.md`，重新定义我们升级 `ply2scene/convert.py` 的目标与实现路径：尽量贴近 VideoMimic 论文中“局部裁剪 + 控密度 + mesh 前补洞 + 两轮重建”的思路，同时避免引入不必要的运行时参数。

---

## 1. 约束与取舍（按你的要求修订）

1) 不做置信度/质量过滤：我们没有 per-pixel confidence 数据，跳过该步骤。  
2) 尽量少加运行时参数：不引入诸如 `--mask_mode`、`--mask_dilate_px` 这类频繁调的开关；必要参数集中在 `convert.py` 的 config 里，并给出合理默认值。  
3) 支持深度梯度过滤：保留并作为默认流程的一部分。  
4) 不做“时空采样/融合策略”：不引入 `frame_stride/max_frames` 等采样逻辑；我们按“处理全帧，但每帧强裁剪 + voxel 限额”来控密度。  
5) 不启用“MuJoCo 碰撞表达升级”：不在方案中引入 plane/heightfield/proxy 等替代碰撞表达，这部分从升级计划中移除。

---

## 2. 目标流程（VideoMimic 论文风格的 3 步）

### Step 1：先裁剪——只保留“人附近”的时空范围（per-frame crop）

对每一帧的世界坐标点云，以人体为中心裁一个固定范围盒子：

- 以 SMPL（或 SMPL-X）关节/人体为中心
- 盒子大小：`±2m`（建议作为固定默认值）
- 只保留该盒子内的场景点，再把各帧保留下来的点合并

尺度说明：

- 我们当前使用的 `ground_alignment.load_scene_pointcloud_from_rgbd(...)` 在生成点云时，会利用 `depth_recovered/*_scale_factor.txt` 做 **scene scale recovery**（对深度与相机位姿平移进行缩放），因此拼接出来的点云已经处于“真实/人尺度”的米单位。
- 在 Holosoma 的 pipeline 中，人身高 prior 默认是 `1.7m`（见 `pipeline.py` / `pipeline_config.json` 的 `human_height_m`），因此 `±2m` 的裁剪盒可以直接按米使用，无需额外换算。

实现上我们不强依赖“关节”，可以用每帧的 SMPL-X `vertices`（`smplx_params.pt` 中就有）来近似人体中心：

- 把 `vertices` 转到 world（用 `camera_pose` + scene scale recovery）
- 再执行 `opencv_to_z_up` + `T_align`
- 用 vertices 的均值/中位数作为人体中心（或用其 AABB center）

这样等价于“只在机器人/人会互动的区域保留点”，大范围背景直接丢弃，显著减少伪面与计算量。

### Step 2：再控密度——体素网格下采样 + 每格限额（voxel cap）

VideoMimic 的关键不是“跳帧采样”，而是**局部裁剪后再做体素限额**：

- `voxel_size = 0.1m`
- 每个体素最多保留 `20` 个点（不是 1 个点）

这一步同时起到“降噪 + 控密度”的作用：点云数量会显著下降，但依靠“每格最多 20 点”避免过稀导致表面细节断裂。

实现上建议不用 Open3D 的 `voxel_down_sample`（每体素只保留 1 点），而是我们自己做一次 `voxel -> group -> random/sample up to K` 的限额采样。

### Step 3：mesh 前补洞——top-down ray casting 填大洞 + 两轮重建

体素限额会让表面出现空洞，VideoMimic 在 mesh 前做补洞，典型是两轮 meshify：

1) 先用更强的重建方法得到粗 mesh（VideoMimic 用 NKSR/NDC；我们可设计为“优先 NKSR，fallback Poisson”）  
2) 再从上往下做 ray casting，在 convex hull（或有效 XY 区域）内：
   - 找到未命中/大洞区域
   - 用逆距离插值（IDW）估计缺失表面高度（或生成补点）
   - 把补点与原点云合并
3) 再跑一次重建得到最终 mesh

这里的核心是：**补洞发生在 meshify 之前**，而不是事后 patch mesh。

---

## 3. 我们的 ply2scene 还欠缺什么（对照 Step 1~3）

对照当前 `ply2scene/convert.py`（baseline Poisson）：

- Step 1（per-frame 人附近裁剪）：目前没有，导致背景大范围点被网格化，伪面多、慢。  
- Step 2（0.1m voxel + 每格最多 20 点）：目前只有 `voxel_down_sample`（每格 1 点）或统计去噪，跟论文策略不同。  
- Step 3（mesh 前补洞 + 两轮重建）：目前没有，Poisson 直接上，洞/缺面会被放大。  

另外还有一个工程问题：triangle winding/法线朝向会影响单面渲染与部分几何计算的稳定性，这个应作为 mesh 后处理的一部分修复（与“碰撞表达升级”无关，但能解决你观察到的“上视为空”）。

---

## 4. 升级实现设计（落到 convert.py）

### 4.1 点云拼接改造：在“每帧”就做裁剪与梯度过滤

新增一个内部函数（替换/旁路 `ground_alignment.load_scene_pointcloud_from_rgbd`）：

1) 逐帧读取：
   - `predicted/{frame}_color.png`
   - `predicted/{frame}_depth.png`（mm -> m）
   - `predicted/{frame}_pose.txt`
   - `results/{frame}/smplx_params.pt`（取 `cam_int`、`camera_pose`、`vertices`）
2) 深度梯度过滤（默认启用）：
   - 在 depth map 上计算梯度（例如 Sobel/差分）
   - 超过阈值的像素不投影
3) 场景点云投影：
   - 调用现有 `create_pointcloud_from_rgbd`（或等价实现）
   - 做 scene scale recovery（沿用现有 avg scale factor 逻辑）
   - `opencv_to_z_up` + `T_align`
4) 计算人体中心（per-frame）：
   - 用 `smplx_params.pt` 的 `vertices`（经 `camera_pose` + scale factor + `opencv_to_z_up` + `T_align`）
   - 得到人体中心 `c`（mean/median/AABB center）
5) per-frame crop：
   - 对本帧场景点 `p`，只保留 `|p - c| <= 2m`（按 xyz 或 xy+z 不同阈值，默认 xyz 同阈值即可）
6) 合并所有帧裁剪后的点与颜色

注意：这里不引入 mask-mode 等参数；mask 的使用（如果需要）采用固定策略（例如固定去掉人体区域或固定保留场景区域），不做 CLI 开关。

### 4.2 体素限额（实现论文的“每格最多 20 点”）

在合并后的点云上做一次：

- `voxel_size = 0.1m`
- `max_points_per_voxel = 20`

实现建议：

1) 计算 `vid = floor(points / voxel_size)` 得到体素坐标  
2) 对 `vid` 分组（哈希/排序/np.unique）  
3) 每组随机/均匀采样最多 K 个点（K=20）  

输出的是“控密度后的点云”，后续 meshify 以它为输入。

### 4.3 两轮 meshify + top-down 补洞（接口先设计清楚）

我们按“可插拔重建器”设计：

- `meshify(points)->mesh`：优先 NKSR（如果依赖存在），否则 fallback Poisson

然后：

1) coarse mesh：
   - 用较粗参数跑一次 `meshify_coarse`
2) top-down ray casting 补洞：
   - 选定有效区域：用 coarse mesh 的 XY 投影凸包 / AABB（简化可先用 AABB）
   - 在该区域内按固定网格采样 XY（建议间距与 `voxel_size` 同级）
   - 沿 -Z 做 ray cast 到 coarse mesh（Open3D RaycastingScene 或 trimesh/ray）
   - 对 miss 的位置，用邻域 hit 点做 IDW 插值得到 z（或生成补点）
   - 生成补点集 `P_fill`
3) final mesh：
   - 用 `P_final = P_voxel_limited ∪ P_fill` 再跑一次 `meshify_final`

我们不在这里引入过多参数：coarse/final 的关键参数可以保留在 config 里，但应给出固定默认值并尽量少露给 CLI。

### 4.4 mesh 后处理（必须）

无论使用何种 meshify，都建议保留最小后处理：

- 去退化/重复/非流形边
- triangle winding/法线朝向修复：
  - 先 `orient_triangles`（如果可用）
  - 再用“最低 z 区域法线应指向 +Z”的规则判断是否需要全局翻面

这能解决你遇到的“从上看空”的单面显示问题，并减少后续几何算法对正反面敏感导致的不稳定。

---

## 5. 参数策略（只保留必要项）

建议保留/新增的“必要参数”（在 config 中给默认值即可）：

- `roi_half_extent_m = 2.0`（固定默认，除非你明确要调）
- `depth_gradient_thr = <固定默认>`（需要实验给一个合理值）
- `voxel_size = 0.1`
- `voxel_max_points_per_cell = 20`
- `mesh_method = poisson | nksr(auto)`（一个开关即可）
- `poisson_depth`（如果 fallback 仍用 Poisson，这是必要调参点）

不引入：

- confidence 过滤相关参数
- 时空采样（frame_stride/max_frames）
- mask_mode/mask_dilate 等频繁变化的开关（内部固定策略）
- MuJoCo 碰撞表达升级相关开关（plane/heightfield/proxy）

---

## 6. 开发顺序（里程碑）

### Milestone 1：复现 Step 1 + Step 2（最关键的“控密度”）

- [ ] per-frame 人附近裁剪（±2m）
- [ ] 深度梯度过滤（默认启用）
- [ ] 体素限额（0.1m，每格最多 20 点）

### Milestone 2：两轮 meshify + top-down 补洞（Step 3）

- [ ] coarse meshify（先用 Poisson 模拟，后续再接 NKSR）
- [ ] top-down ray casting + IDW 生成补点
- [ ] final meshify（P ∪ P_fill）

### Milestone 3：mesh 朝向修复与验收

- [ ] triangle winding/法线朝向修复（地面朝 +Z）
- [ ] 在 `viser_preview.py` / MuJoCo viewer 中验证：上视不空、洞显著减少
