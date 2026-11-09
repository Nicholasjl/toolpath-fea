# PathFEA / toolpath-fea

考虑 3D 打印走线方向的二维与三维线弹性有限元分析原型。支持闭合 STL、正交各向异性材料、G-code 方向映射、薄壁贴体四面体及偏置载荷；提供桌面界面和适合 Agent 调用的 JSON CLI。

## 快速开始

推荐 Python 3.12（本机验证环境），从仓库根目录运行：

```powershell
python -m pip install -r requirements.txt
python run3d.py example_case.json --output output/example --timeout 300 --memory-mb 2048
```

CLI 的 stdout 为单个 JSON，stderr 为进度事件。输出清单 `output/example.json` 中的 `artifacts` 指向本次报告、图片、NPZ 和 VTK；报告与图片需保存在同目录。默认例子是测试块，无需下载外部 STL。

STL 默认采用质量四面体网格 `pq1.4Q`，允许边界细分；目标尺度从2mm起步，CLI和桌面均默认进行两档细化检查。实际体积网格未变细时，不会因位移接近而报告收敛通过。`voxel` 需显式选择。

桌面界面：`python app3d.py`（Windows 可双击 `start3d.cmd`）。二维界面：`python app.py`。

## 三维能力

- 四节点四面体，每节点三个平移自由度；局部正交各向异性本构。
- STL 体素或贴体四面体网格，可选审计后的表面预处理；稀疏装配、直接法或 AMG/CG。
- 按最近层与最近挤出线段映射材料方向，并报告路径覆盖距离。
- 面/盒选固定和加载区域；远端 `load_point` 通过等效分布力及力矩表达偏置载荷。
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
