# 求解器优化与验证（2026-11-07）

## 已实现

- 三维单元分批装配、平衡归并稀疏矩阵；应力恢复按批重新计算梯度，不保留全模型 B/TB/单元刚度。局部坐标差分求梯度，减少大世界坐标对逆矩阵精度的影响。
- 对角缩放的直接求解，以及六刚体模态辅助的 AMG/CG。检查原始系统残差，记录实际算法、迭代数、装配及求解耗时。低质量贴体网格自动优先直接法，不宣称迭代法在所有模型上更快。
- 打印路径以短子段建立 KD-tree，使用距离上界筛选后计算精确线段距离；新旧映射结果通过随机长线段对照。
- 增加保留 STL 表面的 TetGen 贴体网格，细腹板不再依赖均匀体素采样。坐标数值清理、零体积级四面体剔除、体积误差、内腔及连通性均有审计。
- Agent 与桌面均接入网格/求解算法选择；同次角度比较复用网格。原生库日志隔离，NPZ 元数据无需 pickle。

## 同网格性能对比

算例：G5 加强梁局部网格，16564 节点、78960 个线性四面体。同一组节点、单元、材料、固定面、1 N 板外载荷及统一 90° 方向。每个求解器在独立 Python 进程运行，用 psutil 每 20ms 采样峰值 RSS。耗时包括算法导入、装配、求解及应力恢复，不包含后续压缩保存。

| 实现 | 耗时 | 峰值 RSS | 原始系统相对残差 |
|---|---:|---:|---:|
| 保存的旧求解器 | 8.26 s | 978.7 MiB | 8.24e-12 |
| 新分块 + 直接法 | 8.19 s | 857.1 MiB | 7.55e-12 |
| 新分块 + AMG/CG | 2.28 s | 271.7 MiB | 1.41e-9 |

此算例中 AMG 约快 3.6 倍、峰值内存下降约 72%。相对旧解，AMG 位移向量差异为 2.85e-11，应力向量差异为 4.33e-11。单机单次测量，不是普适性能保证。

原始数据：[solver_comparison.json](benchmarks/solver_comparison.json)。复现：`python benchmarks/compare_solvers.py`。旧源码保存在 `benchmarks/reference_solid.py`，SHA256：`0971d74c4b7b499cf49ac61e63afb1758692385d7501ea47f1b8e8803422a776`。

## G5 整板端到端验证

固定输入是前次从 BASE 1.1.0 提取的整板 ASCII STL 和 25 层分析路径。工况见 `benchmarks/board_case.json`：底部条带完全固定，中央表面盒选区施加 1 N 板外合力。**这是求解器数值基准，不代表真实上墙安装或挂钩接触。** 实际选中区域数量、面积及范围保存在检查结果中。

```powershell
python run3d.py benchmarks/board_case.json --output benchmarks/board_end_to_end --timeout 180 --memory-mb 2048 --no-convergence
```

本次完整运行：85455 节点、240106 单元；从校验、网格、走线映射到导出/绘图约 34.8 秒，峰值工作进程 RSS 1512.8 MiB。装配约 1.57 秒，直接线性求解约 3.95 秒，原始系统相对残差 1.46e-8，力与力矩平衡检查通过。

整板体积相对误差约 0.00094%。数值清理删除 12 个共面级四面体，总体积 1.89e-13 mm³；未删除可见细梁或填充孔洞。相比此前 2 mm 体素网格不连通，现在已能完成整板数值求解。

**结果状态仍为 needs_review**：存在低质量单元、加载区域较小、单元跨打印层、未做网格收敛等告警。保留输入表面会限制边界细化；不能以此推断最大承重或准确局部应力。首次直接在含微小退化面原始网格上启用质量细化失败，未经处理的 AMG 迭代也曾停滞，因此没有把这些路线伪装成成功的默认方案。

完整运行清单：[board_end_to_end.json](benchmarks/board_end_to_end.json)。表面网格改良、边界独立细化、材料标定和真实安装工况仍是后续精度工作。

## 回归验证

`python -m unittest -v test_solver.py test_solid.py test_agent.py test_optimized.py`：30 项通过，覆盖二维/三维解析解、平衡、网格变化、直接/迭代一致性、路径空间索引、薄体与内腔网格、元数据安全读写，以及取消/资源阈值/错误协议。桌面无显示求解与绘图检查通过。

实现接口参考：[MeshPy / TetGen](https://documen.tician.de/meshpy/tri-tet.html)、[PyAMG 聚合多重网格](https://pyamg.readthedocs.io/en/latest/generated/pyamg.aggregation.html)。

## S1B1 surface conditioning repair

S1B1W produced TetGen error 3 even with tighter T and disabled coplanar merging. Independent MeshLab self-intersection selection found zero intersecting faces; the cleaned surface contained triangles with altitude about 9.8e-6 mm. Explicit constrained surface conditioning provides a way to remove these numerical slivers, with audited geometric limits. This is a case-specific observed correlation, not a claim that all TetGen error 3 cases are false intersections. Details and controls are in AGENT_API.md. Regression tests include thin bodies with closed cavities, invalid parameters, and mesh-error classification.

Verified S1B1W run: analyses/s1b1-frame-20261107/framed_conditioned.json; 81951 nodes, 270248 tetrahedra, original residual 1.96e-10; force/moment balance passed. Source volume error 3.99e-5, sampled surface distance 0.01003 mm, material/mesh limitations remain. Both nominal pitches produced identical meshes, re-audited with the new convergence guard in repair_verified_summary.json. S1B1H conditioning at 0.7 and 0.5 mm still failed TetGen; this is not a universal repair for every PLC.

## Follow-up: S1B1H resolved

S1B1H input had zero MeshLab-detected intersecting faces, but unconditional smoothing created four. Disabling smoothing removed this failure: holes_nosmooth.json completed with 247091 tetrahedra, residual 6.19e-9, force/moment balance passed. Input/output self-intersection checks are now mandatory for conditioning. A bounded fallback for TetGen code 3/4/5 can try smoothing only on a fresh snapped copy, after all audits; this preserves compatibility with S1B1W. Forty regression tests cover intersecting closed surfaces, bounded retry, rejected fallback never reaching TetGen, thin cavities, and identical-mesh convergence. Both unchanged nominal mesh pairs remain not_checked for convergence.
