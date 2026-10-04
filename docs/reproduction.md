# 运行示例与复现范围

本仓库提供两个轻量入口：合成剖面的路沿检测，以及真实 CARLA 任务快照的离线规划重算。历史 NDT、HEAL 和 CARLA 指标通过 `evidence/` 原始结果查看；重跑完整传感器链另需原工程、模型、地图和运行环境。

## 1. 查看历史指标

在仓库根目录运行，使用 Python 标准库：

```bash
python scripts/show_metrics.py
```

## 2. 路沿组件与示例

Python 3.10–3.12，NumPy 1.26.4。已有环境可直接运行；新环境安装仓库依赖：

```bash
python -m venv .venv
```

Linux 激活使用 `source .venv/bin/activate`，Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`。随后：

```bash
python -m pip install -r requirements.txt
python scripts/check_curb.py
python scripts/demo_curb.py
```

组件测试复用原有合成剖面测试，覆盖曲线、缺列分段、坡道/台阶类型、混合立面与缺地面支撑。示例展示 12 cm 台阶中间 0.4 m 观测缺口，输出两个独立段。它只验证核行为，精度来自合成输入条件。

## 3. 真实任务规划回放

已冻结的回放入口使用 Linux 的 `resource`、CPU affinity 和负载接口。使用 Linux Python 3.12、NumPy 1.26.4、Shapely 2.0.3；Windows 可在 WSL 内运行。

从仓库根目录进入 `replay/` 后执行，每次使用新的输出目录：

```bash
cd replay
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -s -B simulation/carla-autoware/scripts/replay_litter_tasks.py --run input --output ../results/planning-01 --platform pc
```

系统 Python 的命令名若为 `python3`，替换上述 `python`。虚拟环境中的依赖通过前述 `requirements.txt` 安装。`-s` 禁用用户 site-packages，避免已有不同版本的 NumPy 覆盖指定环境。

五份快照各重算三次，共 15 次，检查路径点位置、航向、挡位及刷具标记。输出回归容差为 1e-6 m/rad。原规划只作为结果对照，规划输入来自任务、已准入垃圾观测与车辆配置；不会推进车辆或发送控制命令。该任务回放与 CARLA 实际驾驶分别评价。

## 4. 其他源码

`src/perception/local_world` 的空间证据实现还使用 SciPy，Numba 为可选加速。`src/perception/heal` 的图像头使用 PyTorch；完整 `SharedPerception` 需原 HEAL 实例、权重与输入。`reference/offline_coverage` 为冻结 4WS 参考源码，原生 Fields2Cover 与覆盖评价器在原实验环境管理。

`provenance.json` 记录选定文件的来源和 SHA-256；`SHA256SUMS` 覆盖最终展示包文件。`validation.json` 区分本次重新运行的检查与历史实验。
