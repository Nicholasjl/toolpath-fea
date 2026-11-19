# PathFEA / toolpath-fea

考虑 3D 打印走线方向的二维与三维线弹性有限元分析原型。支持闭合 STL、正交各向异性材料、G-code 方向映射、薄壁贴体四面体及偏置载荷；提供桌面界面和适合 Agent 调用的 JSON CLI。

## 快速开始

推荐 Python 3.12（本机验证环境），从仓库根目录运行：

```powershell
python -m pip install -r requirements.txt
# 可选 GPU / CPU 走线级求解加速（warp-lang）
python -m pip install -r requirements-gpu.txt
python run3d.py example_case.json --output output/example --timeout 300 --memory-mb 2048
```

CLI 的 stdout 为单个 JSON，stderr 为进度事件。输出清单 `output/example.json` 中的 `artifacts` 指向本次报告、图片、NPZ 和 VTK；报告与图片需保存在同目录。默认例子是测试块，无需下载外部 STL。

`run3d.py` 通过 `engine`（工况字段或 `--engine`）选择引擎，默认 `auto`：
- 当工况提供 `gcode`（且未使用 `--angles`）时，默认自动启用 **`toolpath_graph`（6-DOF 走线丝束-焊缝网络）**，显式建模闭环 Z-seam / Scarf 接缝、开路径启停锚点、热冷却衰减与接头失效指数；
- 可显式切换 **`toolpath_hex`**（`--engine toolpath_hex`，默认 `0.2 mm` 稀疏砖块 GPU 无矩阵多重网格六面体连续体 + 路线 2 界面标志弱化）；
- 未提供 `gcode`（仅 STL 或测试块）或显式指定 `--engine legacy_solid` 时回落到传统贴体四面体/体素求解器（`tetra` 默认 `pq1.4Q` + 两档细化检查）。
- 走线级双引擎默认 `--device auto`（有 NVIDIA 显卡使用 `cuda:0`，无独显自动通过 Warp LLVM 后端运行多线程 `cpu`）。

独立直接入口（无需监控壳，支持大模型分批缓存）：`python run_toolpath_graph.py case.json` / `python run_toolpath_gpu.py case.json`。

桌面界面：`python app3d.py`（Windows 可双击 `start3d.cmd`）。二维界面：`python app.py`。

## 三维能力

- **走线网络（`toolpath_graph`，有 G-code 时默认）**：流式双层滑窗构建 6-DOF Timoshenko 丝束梁 + 偏心界面耦合器，不合并闭环首尾而是显式保留 `seam_closed` / `seam_scarf` 与 `start_anchor` / `stop_anchor`，GPU 连通聚合多重网格 + 外层 `fp64` 截断 FCG，支持 `0.2 mm H8` 对标刚度校准、热冷却时间衰减与基准失效指数 / FoS 筛查。
- **走线体素（`toolpath_hex`）**：两趟流式光栅化 + GPU `4×4` 子采样 -> `4×4×4` 稀疏砖块（`.tbricks`），8 色无矩阵多层多重网格 + 路线 2 启停/接缝/道间界面弱化，精确区分内部空隙与跨层换向。
- **传统实体（`legacy_solid`）**：四节点四面体，每节点三个平移自由度；局部正交各向异性本构；STL 体素或贴体四面体网格（支持 TetGen / TetWild 与表面预处理）；按最近层与最近挤出线段映射材料方向。
- 面/盒选（支持多安装孔 `boxes` 并集）固定和加载区域；远端 `load_point` 通过等效分布力及力矩表达偏置载荷。
- 作业超时、进程内存监控、取消文件、输入哈希、数值诊断与报告。

例如，板正面 Z=5 mm，挂点伸出正面 10 mm，1 kg 重物向下（-Y）的工况片段：

```json
{
  "force": [0, -9.80665, 0],
  "load_point": [118, 118, 15]
}
```

还需为实际零件设置 STL、材料、`fixed` 和 `loaded`。相对正面孔中心的弯矩是 98.0665 N·mm。这个载荷模型不包含挂钩接触。

## 文档与 Skill

- [三维使用说明](README_3D.md)
- [Agent CLI 协议](AGENT_API.md) 与 [工况 schema](case.schema.json)
- [二维使用说明](README_2D.md)
- [求解器优化记录](SOLVER_OPTIMIZATION.md)
- [本地工程集成测试](INTEGRATION_TESTS.md)：批量 CLI 测试入口、结果分类与实际工程记录。
- [toolpath-fea skill](skills/toolpath-fea/SKILL.md)：可复制该目录至 Codex 的 skills 目录；使用时指定实际仓库路径。

历史洞洞板原始模型、分析目录和大型基准输出不包含在仓库中；部分历史文档提到的本地路径需自行提供对应输入。`testdata` 包含表面处理回归用的小型几何数据及来源记录。

## 验证

```powershell
python -m unittest discover -p "test_*.py"
```

测试覆盖本构旋转、载荷与反力平衡、偏置力矩、网格与表面处理、求解器一致性、路径映射和 Agent 协议。

## 结果边界

本工具用于数值筛查。默认材料参数是演示值，未对应具体耗材和打印工艺标定。线弹性结果不预测层间开裂、塑性、接触、蠕变、屈曲或断裂；von Mises 不是打印材料的通用失效判据。

退出码 0 仅表示计算流程完成，仍需检查 `analysis.quality`、平衡、网格收敛及边界条件。体积保留、低残差或方向覆盖通过，不等于薄壁应力收敛，更不等于安全承重。

## 许可证

Copyright 2026 PathFEA contributors.

本项目采用 [Apache License 2.0](LICENSE)（SPDX: `Apache-2.0`）。第三方依赖仍遵循各自的许可证。
