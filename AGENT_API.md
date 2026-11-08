# Agent 调用协议 v1.0

入口 `run3d.py`，无需桌面或显示服务器。安装依赖：`python -m pip install -r requirements.txt`。输入模板 `example_case.json`，字段契约 `case.schema.json`。原桌面入口保持不变。

## 薄壁与大模型选项（2026-11-07）

```json
{"stl":"part.stl","mesh_mode":"tetra","pitch":2,"snap_tolerance":0.0001,"solver":"auto","rtol":1e-9,"maxiter":2000,"chunk_size":4096}
```

- `mesh_mode=voxel` 保持旧行为；`tetra` 用 TetGen 保留输入表面生成贴体四面体，可保留小于体素间距的薄壁及封闭内腔。默认仍为 voxel，需显式选择 tetra。STL 之外的默认测试块仍使用规则网格。
- `snap_tolerance` 是坐标舍入网格间距，默认 0.0001 mm；0 禁用。仅在分析副本中清理重复/退化面，原模型不改写。清理后必须闭合、朝向一致，体积变化不得超过 0.1%，四面体体积与原 STL 偏差不得超过 0.2%。拒绝多实体装配，内腔保留。
- 极少量数值共面四面体（单个体积低于源体积的 1e-15）可被剔除；总剔除体积必须低于源体积的 1e-10，之后重新检查面连通性。数量及体积记录于 `analysis.metrics.mesh`，不用于删除真实薄梁。
- 贴体模式目前保留输入表面三角形，`pitch` 对应内部目标体积 `pitch³/6`，不是保证的最大边长或严格最大体积。表面约束可能阻止进一步细化；真实最大体积、扁薄单元比例会写入报告。默认两档比较之外会标记 `BOUNDARY_REFINEMENT=not_checked`，不能因两次内部网格近似相同就声称薄壁弯曲收敛。
- `solver=auto` 在小系统使用缩放直接法，大系统使用六个刚体模态辅助的 AMG/CG；贴体网格存在极低质量单元时，Agent 和桌面入口会优先选择直接法，避免已观察到的迭代停滞。显式 `direct` / `amg` 可覆盖选择。失败时不会自动跳到高内存算法。选择原因、迭代次数、装配/求解耗时均记录于 `analysis.metrics.solver`。
- `chunk_size`（128–16384）控制装配与应力恢复批次；默认 4096，不再同时保留全部单元刚度和应变矩阵。`rtol` 为缩放系统的 CG 相对容差，但最终仍检查原始系统残差不超过 1e-6。
- 走线索引将长线段细分建立 KD-tree，随后按距离上界搜索并计算精确线段距离；不会以最近中点近似最近线段。超出 300 万索引子段会拒绝。
- 同次作业内的不同走线角复用体网格。NPZ 中 `solver_info` / `mesh_info` 为 JSON 字符串数组，使用 `json.loads(str(data['solver_info']))` 读取；无需启用 pickle。
- 贴体模式额外依赖 MeshPy，AMG 模式额外依赖 PyAMG，均已加入 requirements。原生网格库日志进入工件 `native.log`，不会污染 stdout JSON。

验证记录与整板基准工况见 [求解器优化记录](SOLVER_OPTIMIZATION.md)。

```powershell
python D:\3d\toolpath_fea\run3d.py D:\3d\toolpath_fea\example_case.json --output D:\3d\toolpath_fea\output\job --timeout 300 --memory-mb 2048 --angles 0,45,90 --cancel-file D:\3d\toolpath_fea\output\job.cancel
```

## 输入和输出

- 工况字段与三维界面一致。未知字段、非有限数字、非正定材料、错误选区格式会被拒绝。输入相对 STL/G-code 路径以工况文件目录为基准。
- stdout 只输出一个 UTF-8 JSON 结果对象；stderr 输出逐行 JSON 进度事件。`--help` 是普通文本帮助。
- 输出前缀对应的 `.json` 是原子替换的结果清单。运行期间状态为 `running`；成功为 `completed`，失败为 `failed`。
- 每次工件位于唯一的 `<prefix>_runs/<run_id>/`，结果中 `artifacts` 提供绝对路径。失败时不列出未完成工件，不能根据旧 VTK/PNG 的存在推断本次成功。多个并行任务应使用不同的输出前缀。
- 清单包含 `schema_version`、`status`、`error`、`exit_code`、`artifacts`；进入监控后的任务还包含 `run_id`、`resources`、`progress_log`。成功求解报告在 `analysis` 中，检查 `analysis.quality` 和每条 `analysis.checks`。
- 工件包括基础工况 `result.npz`、`result.vtk`、`preview.png`、`report.html`、`analysis.json`，以及可选 `refined.npz` 和各角度的 NPZ。HTML 与图片放在同目录即可离线浏览。
- `analysis.inputs` 记录输入 STL/G-code 的 SHA256；运行期间不要修改源文件。

## 判定规则

退出码 0 表示流程完成，**不等于数值校核通过或结构安全**。

`analysis.quality` 为 `numerical_checks_passed` 或 `needs_review`；`scope` 固定为 `numerical_screening_only`。自动检查仅针对基础工况，`checks_scope=base_case_only`；角度对比不逐一细化，若需判断各方案精度，应为每个角度分别运行一次。

检查状态为 `pass` / `warn` / `not_checked`。覆盖力与力矩平衡、选区面积和范围、网格分辨率、STL 体积与抽样表面覆盖、位移量级、G-code 路径覆盖和层高欠采样。

默认用 `pitch / 1.5` 生成第二档网格，比较最大位移变化（阈值 10%）及全局单元等效应力 p95 变化（阈值 15%）。细化失败会记录 `not_checked`，不会伪装成收敛通过。`--no-convergence` 明确跳过这一检查。只有两档网格是筛查，不能证明渐近收敛或局部应力精度。

薄壁丢失检查使用 4096 个固定种子的面积加权表面采样点，并比较网格体积；不能保证发现每个细小特征。路径距离通过也不能证明与实际打印件完全对齐。材料标定、真实边界条件、填充等效和失效判据仍需外部依据。

`--require-checks` 要求基础工况所有已列校核通过；否则返回 `CHECKS_FAILED`/8，并保留工件供复核。

## 退出码和错误码

| 退出码 | 常见 error.code | 含义 |
|---|---|---|
| 0 | null | 完成；仍须读取 quality |
| 2 | INVALID_CASE / INVALID_JSON / INPUT_NOT_FOUND / INVALID_MATERIAL | 修正输入后重试 |
| 3 | MESH_ERROR / GCODE_ERROR / SOLVER_ERROR | 网格、路径或求解失败 |
| 4 | TIMEOUT | 作业超时，工作进程已终止 |
| 5 | MEMORY_LIMIT | 工作进程 RSS 超阈值 |
| 6 | CANCELLED | 取消文件或 Ctrl+C |
| 7 | IO_ERROR / INTERNAL_ERROR / WORKER_CRASH / DEPENDENCY_MISSING | 环境或内部错误 |
| 8 | CHECKS_FAILED | 严格模式的校核未通过 |

错误包含 `code`、`message`、`stage`，不要通过中文文本匹配逻辑。示例：

```json
{"schema_version":"1.0","status":"failed","error":{"code":"TIMEOUT","message":"超过 300 秒时限","stage":"base.linear_solve"},"artifacts":{},"exit_code":4}
```

## 资源、进度和取消

- 数值计算、细化、角度比较和绘图在独立子进程中执行。默认作业时限 300 秒，工作进程内存阈值 2048 MiB。
- 内存以约 100ms 间隔采样 RSS（含可见子进程），超阈值终止。**这是监控阈值，不是操作系统硬配额**；瞬时分配可能超出阈值，不能保证机器绝不发生内存压力。外层监控进程的内存不计入阈值。
- 进度是阶段事件（mesh、path_mapping、assembly、linear_solve、postprocess、render），不是虚构的百分比。线性求解阶段可能一段时间没有新事件，仍持续监控资源。
- 创建 `--cancel-file` 指定文件即可取消；文件已存在则任务不会启动。复用前请由调用者清除此取消文件。Ctrl+C 也会终止工作进程并返回结构化取消结果。
- 超时/取消可能留下独立运行目录中的部分文件，但清单不会把它们列为完成工件。不要依靠目录扫描提取结果，使用清单。

## 调用示例

```python
import json, subprocess, sys
p = subprocess.run(
    [sys.executable, r"D:\3d\toolpath_fea\run3d.py", "case.json",
     "--output", "output/job", "--timeout", "120", "--require-checks"],
    capture_output=True, encoding="utf-8", timeout=140)
result = json.loads(p.stdout)
if p.returncode != 0:
    print(result["error"])
else:
    print(result["analysis"]["quality"])
    print(result["artifacts"]["report.html"])
```

外层 `subprocess` 超时应长于工具自身时限，让工具有时间终止子进程并写结果。需要实时进度时用 `Popen` 并持续消费 stderr，避免管道写满。

## Remote force point (CLI)

Optional `load_point: [x,y,z]` is the world-space application point in mm. `force` remains the resultant in N. The selected `loaded` surface receives area-weighted nodal forces plus a distributed couple preserving both force and moment about the application point. This models an equivalent load transfer, not hook/contact mechanics or a rigid coupling. Omit the field for the original area-distributed load. A point 10 mm beyond the front of a Z=5 mm board has Z=15 mm. The GUI does not expose this field; use case JSON and run3d.py.

## Thin-surface conditioning (2026-11-07)

For a closed thin-wall STL rejected by TetGen due to near-degenerate facets, explicitly set `surface_remesh_mm: 0.7` with `mesh_mode: tetra`. Zero (default) disables conditioning. The value is an edge-length target, not a geometry tolerance or guaranteed mesh size. CLI and GUI-loaded cases preserve this setting. Requires PyMeshLab.

The analysis copy first undergoes three constrained collapse/swap/reprojection passes, with smoothing and edge splitting disabled. Input and output surfaces must have zero intersecting faces according to MeshLab. If TetGen rejects this candidate with code 3/4/5, a single separately audited smoothing candidate is created from the original snapped surface. Other native failures are not retried. Every attempt and selected smoothing flag is recorded in `analysis.metrics.mesh.tetgen_attempts` and `surface_conditioning`. A failed audit prevents submission to TetGen; topology, cavity and volume checks are repeated for the alternate candidate. Each pass uses 0.005 mm local projection tolerance. It must retain closed consistent surfaces, Euler characteristic and component count, stay within 0.1% cumulative volume change, and pass deterministic bidirectional 8192-point surface-distance sampling within 0.015 mm. This sampling is not a Hausdorff guarantee. Cavities are still supplied to TetGen; original files are never rewritten. `analysis.metrics.mesh.surface_conditioning` records this audit.

Configured coordinate snapping now precedes the final closed-surface acceptance; this allows duplicated/near-coincident STL vertices to be welded within the explicitly configured tolerance, without filling holes. `source_closed_before_snap` records the initial condition. TetGen rejection is reported as `MESH_ERROR`, not `INTERNAL_ERROR`. No volume, connectivity, residual, force/moment, or convergence checks are bypassed.

Conditioning changes the analysis surface and does not prove local stress convergence. Compare independently refined surfaces for critical thin-wall bending.

Identical node and cell arrays across two nominal pitches now yield `MESH_CONVERGENCE=not_checked`, never `pass`. This guards against false convergence when boundary preservation blocks interior refinement.

S1B1H regression: the previous unconditional smoothing introduced four intersecting faces near an installation pad despite preserving watertightness, Euler number and volume. The no-smoothing candidate has zero detected intersections and solves successfully. S1B1W uses the separately checked smoothing fallback. This is a bounded geometry-conditioning workflow, not general self-intersection repair; existing intersecting inputs are rejected.
