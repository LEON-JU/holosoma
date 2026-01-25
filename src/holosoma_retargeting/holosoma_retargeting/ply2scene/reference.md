根据我对VideoMimic代码仓库的分析，我找到了以下相关代码：
1. 点云清理与裁剪
位置: /home/juyiang/code/RobotRetarget/VideoMimic/real2sim/stage3_postprocessing/mesh_generation.py
关键函数: filter_points_and_colors() (第74-302行)
功能包括:
- 动态人体点去除: 使用dynamic_msk掩码识别和移除动态人体点 (第188行)
- 背景噪声过滤: 
  - 置信度阈值过滤 (conf_thr) (第191行)
  - 梯度阈值过滤 (gradient_thr) 去除深度不连续的地方 (第195-198行)
- 空间裁剪: 基于SMPL关节点的3D包围盒裁剪点云，只保留人体附近区域 (第201-230行)
- 时空过滤: 从相邻帧采样点，增强点云密度 (第232-273行)
2. 点云转换网格
位置: /home/juyiang/code/RobotRetarget/VideoMimic/real2sim/stage3_postprocessing/meshification.py
关键函数: two_round_meshify_and_fill_holes() (第161-232行)
功能包括:
- 点云降采样: 使用体素网格降采样 (downsample_point_cloud()) (第234-322行)
- 法向量估计: 计算点云法向量 (estimate_point_normals()) (第325-351行)
- 网格重建: 支持两种方法
  - NKSR算法 (Neural Kernel Surface Reconstruction): 神经网络Kernel表面重建 (第177-215行)
  - NDC算法: 基于深度图的方法 (第217-232行)
- 孔洞填充: 从上方射线检测并填充网格孔洞 (get_point_cloud_to_fill_holes()) (第96-158行)
总结: 该仓库包含完整的点云清理裁剪和网格生成代码，适用于从视频中提取环境网格。