# 实验结果与证据

实验数据采于 2026-09-08 / 2026-09-15，展示包整理于 2026-10-04。定位、感知、离线规划和车辆驾驶分别评价。原始评分 JSON 保留数值与实验标识，仅重排 JSON 并替换本机路径/主机名；转换及散列见 [provenance.json](../provenance.json)。

## 定位与地图

CARLA Town01 建图会话 long01 行驶 3,000.022 m；GLIM 生成 279 个子图，使用 GNSS 锚定后发布 4,146,129 点、380 个 20 m 瓦片。后续两个定位会话均未参与建图，评分真值只在定位完成后读取，未按真值做事后位姿拟合。

| 会话 | 处理 / 接受帧 | 纵向绝对误差 P95 | 横向绝对误差 P95 | 航向绝对误差 P95 |
| --- | --- | --- | --- | --- |
| holdout01，NDT 回放 | 1,230 / 1,222 | 7.63 cm | 8.78 cm | 0.1618° |
| online01，NDT 实时输入 | 877 / 876 | 7.79 cm | 6.69 cm | 0.1366° |

误差包含全部处理帧及初始化失败帧。该实验由 TrafficManager 控车，NDT 使用轮速/陀螺先验并给出地图校正；自建图上的驾驶控制与 EKF 完整循环待接入。

原件：[留出评分](../evidence/localization/holdout-ndt.json)、[实时评分](../evidence/localization/online-ndt.json)、[地图指标](../evidence/localization/map-metrics.json)、[GNSS 图优化](../evidence/localization/gnss-graph.json)。

## 共享路沿头

共享 HENet P8/P16，分别投影到 64 通道，融合后经深度可分离卷积输出根部热图、偏移和训练辅助区域语义。冻结继承主干及 BatchNorm，12 组/24 张工作图训练，第三会话 8 组/16 张图验证；均来自同一路段，标签为 AI 审核可见线。

RTX 4090 D 上同帧、同权重、同 Occ 出口，预热 200 次、测量 1,000 次：

| 配置 | 模型前向 P50 / P95 | PyTorch 峰值已分配显存 |
| --- | --- | --- |
| 无路沿头 | 35.98 / 38.12 ms | 824.78 MiB |
| 共享两视角头 | 36.70 / 38.87 ms | 824.90 MiB |
| 两视角重复编码 | 39.66 / 42.00 ms | 853.58 MiB |

共享头前向中位数增量 0.72 ms，相对重复编码节省 2.96 ms。计时包含模型前向与 CUDA 同步，预处理、FCOS、宿主拷贝和 CPU 连线另计。581 帧完整 HEAL/FCOS 回放的检测及 Occupancy 类别逐元素一致。

像素验证用于检查同路段会话迁移；独立路段、右侧视角、复杂遮挡和米制融合仍待验证。原件：[训练汇总](../evidence/shared-head/training-summary.json)、[性能对照](../evidence/shared-head/benchmark.json)、[选定权重诊断](../evidence/shared-head/selected-diagnostics.json)、[581 帧对拍](../evidence/shared-head/replay-comparison.json)。

## 离线覆盖规划

G3 为外框 6 × 6 m、臂宽 2 m、目标面积 20 m² 的 L 形区域。使用五刷 CAD 足迹与受限四轮反相 4WS 运动学；覆盖评价用 5 cm 栅格，车辆执行积分为 0.02 s，工具延迟 0.5 s，任务上限 900 s。

| 方法 | 最终几何覆盖率 | 耗时 | 同 397.72 s 预算覆盖率 |
| --- | --- | --- | --- |
| A：常规条带 | 80.525% | 397.72 s | 80.525% |
| B：条带加扫边 | 99.6125% | 865.46 s | 83.9375% |
| C：工具约束候选搜索 | 99.875% | 661.46 s | 92.2625% |

C 的最终覆盖提升伴随更长任务时间与重复覆盖。G3 只支持本场景中的规划取舍，几何足迹评价与垃圾清除率分别评价；该组结果的车辆模型为 4WS。整体 45 组正式试验为 34 完成、7 超时、1 跟踪失败、3 不可行。

原件：[A](../evidence/coverage/G3-A.json)、[B](../evidence/coverage/G3-B.json)、[C](../evidence/coverage/G3-C.json)、[开发集共同预算比较](../evidence/coverage/development-comparison.json)、[45 组批次摘要](../evidence/coverage/batch-summary.json)。实验冻结源码位于 [reference/offline_coverage](../reference/offline_coverage)。

## 在线路沿驾驶

第 13 趟为静止车辆场景，14 m 直线任务，CARLA autopilot 关闭；驾驶参考来自在线实测 LiDAR 路沿。独立评价器记录前进 13.8948 m、最高速度 0.5866 m/s，在前进 3.0228 m 处停车，障碍移除后恢复，终点速度 0.0000632 m/s，碰撞 0。429 个规划周期中，移除障碍后 1.0 s 恢复行驶指令，随后连续 270 周期至终点。

条件为理想化 LiDAR 与简化自车回波处理，刷具关闭；该试验评价路沿引导与停车恢复，刷缘精度和完整清扫另验。

原件：[场景评分](../evidence/driving/scenario-result.json)、[连续性复核](../evidence/driving/continuity-review.json)。驾驶器内部进度与独立真值行程分列，本文采用后者。

## 垃圾驱动作业

三轮 CARLA、任务速度上限 0.4 m/s、场景真值垃圾目标、逻辑示例刷具。先完成空目标 4 × 0.6 m 区域全覆盖，再执行两点一面任务并取消以注入遗漏，最后执行补扫。

| 任务 | 结果 | 真值行程 |
| --- | --- | --- |
| 空目标全覆盖 | 2.4 m²，账本与独立评分均为 100% | 5.910 m |
| 取消注入遗漏 | 近点完成；面目标和远点保留为遗漏 | 5.048 m |
| 补扫 | 面目标 100%、远点覆盖；账本待处理格归零 | 9.565 m |

三段碰撞与规划走廊越界均为 0，末端速度均小于 0.02 m/s。按需补扫的完整指定区域覆盖率为 28.13%，目标覆盖与全域覆盖分别计算。707 个驾驶位姿样本的覆盖账本重放逐项一致；五份任务规划快照各重复三次，原归档 15/15 一致，位置和航向最大差均为 0。

这里评价给定垃圾信息下的规划、执行与几何覆盖；示例刷具尚未实车接触标定，垃圾识别与扫后残留另验。

原件：[独立评分](../evidence/cleaning/result.json)、[账本重放](../evidence/cleaning/replay-and-motion.json)。冻结真实任务输入和回放源码在 [replay](../replay)。本次重新验证结果见 [validation.json](../validation.json)。
