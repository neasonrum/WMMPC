# WAMPC V5：冻结 V3 + 2Hz 世界模型缓存修正

本文件说明新增的 `method/full_diffusionv5`。V3、V4 的源码、配置和 checkpoint 均未修改。

> 状态：**代码已写完，只做过编辑器静态检查；没有运行过数据准备、训练、验证或 CARLA 闭环。**
> 第一次使用请先跑 `--max-updates 20` 的短训练排查问题。

---

## 1. 为什么不用 V4 继续改

V4 名义上从 V3 初始化，但并没有保留 V3：

| V4 的改动 | 后果 |
|---|---|
| 空间路径 10 点改为 6 点，路径编码、点嵌入、BEV 读取权重、输出层都做了插值改写 | 初始化时就不等于 V3 |
| 当前 BEV 读取从单点采样改为中心 + 4 个邻点（初始权重 0.8 / 0.05） | 初始化时就不等于 V3 |
| 轨迹归一化参数重新测量 | 输出尺度改变 |
| 所有 V3 参数以 1e-5 / 2e-5 继续训练 | V3 的能力持续漂移 |
| 训练时上一计划年龄固定 0.5s，但推理时每个 20Hz tick 都调用 WM，上一计划只隔 0.05s | 训练与闭环不一致 |

V5 的定位是：**V3 负责规划，WM 只负责修正。**

## 2. V5 结构

- **冻结的 V3**：直接加载 `ckps/full_diffusionv3/release/epoch_10.pt`，保留 10 个空间点、6 个时间点、原有 BEV 读取、归一化和扩散设置，参数全部冻结，始终处于 eval 模式。
- **适配器**（唯一可训练部分，WM 默认也冻结）：
  - 未来特征采样器：沿用 V4 的 95 个邻域 token（5 张未来 BEV，点数 `[9,9,16,25,36]`），围绕“驱动 WM 的旧计划”采样。
  - 每个 V3 解码层（空间、时间分支各 2 层）在 ego 注意力之后插入一个读取模块：
    - 读取前，把缓存 token 的位置、方向用里程计换到**当前自车坐标系**；
    - 每个 token 的相对时间 = 生成时的未来时间 − 缓存年龄；
    - 时间分支按**真实相对时间**匹配未来图，不再按序号对应；
    - 可靠性分支额外输入缓存年龄和时间差；
    - 输出乘以 `tanh(gate)`，**gate 零初始化**。
- **关键性质**：没有 WM 缓存，或 gate 为 0 时，V5 与 V3 **逐位相同**。所以“关闭 WM”就是原版 V3，不需要再加载第二个 checkpoint。

## 3. 三条时间轴（必须区分）

| 时间轴 | 频率 | 说明 |
|---|---|---|
| 数据记录 | 4Hz（0.25s） | Garage 配置 `carla_fps=20`、`data_save_freq=5`，中间 tick 没有保存任何观测 |
| WM 未来图 / 轨迹节点 | 0.5s 网格 | 当前之后 0.5…2.5s 的 5 张未来 BEV；6 个 0.5s 时间位移 |
| WM 刷新（上一计划驱动的推演） | 2Hz | 每 0.5s 仿真时间一次，上一计划年龄始终 0.5s，与训练一致 |
| V5 去噪规划 | 20Hz（每个 tick） | 每个 tick 都用最新观测重新规划，并读取对齐后的 WM 缓存 |
| 控制 | 20Hz | 不变，沿用 V3 的 AutoMoT 纵向 PID 和横向控制器 |

### 闭环时间线（稳定状态）

| 时刻 | WM | 当前去噪读取的未来信息 | 输出用途 |
|---|---|---|---|
| t=0 | 运行，动作 = 对齐后的 P(−0.5) | 新缓存，年龄 0 | 控制；并保存为下一次 WM 的先验 P0 |
| t=0.05…0.45 | 不运行 | t=0 的缓存，按位姿与年龄对齐 | 只用于控制 |
| t=0.5 | 运行，动作 = 对齐后的 P0 | 新缓存，年龄 0 | 控制；保存为 P0.5 |
| t=0.75 | 不运行 | t=0.5 的缓存（围绕 P0 采样），年龄 0.25 | 只用于控制 |

- 冷启动：第一次刷新没有先验，不运行 WM，输出就是 V3；这次输出仍保存为先验。
- 采样位置来自驱动 WM 的旧计划；坐标变换的起点永远是缓存生成的时刻。
- 中间 tick 的输出**不会**作为先验，否则上一计划年龄会变成 0.05s。

## 4. 训练范式：4Hz 数据现在被完整使用

- **锚点**：每个记录帧（0.25s 间隔）只要前 2 帧和后 12 帧连续，就是锚点。V4 只用了每隔一帧的锚点。
- **样本**：连续三个锚点 `(t, t+0.25, t+0.5)` 组成一个样本：
  1. 用 t 时刻 3 帧观测和对齐后的上一计划（来自同一条链的 t−0.5）运行一次 WM，得到缓存；
  2. 同一份缓存分别用于 t（年龄 0）、t+0.25（年龄 0.25）、t+0.5（年龄 0.5）三帧去噪；
  3. 三帧都用各自时刻的真实 GT 计算 10 个空间点 + 6 个时间点的 loss；
  4. t 时刻的采样计划保存为这条链在 t+0.5 的先验。
- **两条 2Hz 链**：同一路线按帧号奇偶分成两条链，先验按“路线#奇偶”分别保存。每条链内部相邻刷新正好相隔 0.5s，**不会出现 0.25s 的先验**。
- 年龄 0、0.25、0.5 都有真实观测和真实 GT，没有任何插值或伪造数据。
- 冷启动行（没有先验）等于 V3、没有梯度，loss 只在 WM 有效的行上取平均。
- 训练时生成先验所用的 DDIM 步数 `rolling_inference_steps = 25`，与推理默认步数相同，避免 V4 里“训练 1 步、推理 25 步”的差异；嫌慢可以改小，但会重新引入这个差异。

**仍然没有监督的部分**：推理时年龄 0.05、0.1…0.45s（除 0.25 外）的情况，训练中没有对应样本，只能依赖模型在年龄输入上的平滑泛化。原因是数据只有 4Hz，中间 tick 没有任何观测。

## 5. Batch size

- V4：128/卡 × 8 = 1024 个样本/次更新。
- V5 默认：`batch_per_gpu = 32`，8 卡共 **256 个样本/次更新**。每个样本含 3 帧，所以每次更新规划器实际看到 768 帧。
- 想用 512：把 `training.batch_per_gpu` 改成 64。
- 采样器要求“2Hz 链数 ≥ 全局 batch”；链数约为训练路线数的 2 倍（约 9600），256 和 512 都满足。
- 默认 5 个 epoch，每个 epoch = `ceil(训练样本数 / 全局 batch)` 次更新，起点设置，按验证结果调整。

## 6. 新增文件

| 文件 | 内容 |
|---|---|
| `method/full_diffusionv5/config.json` | V3 / WM checkpoint 身份、刷新与缓存年龄、batch、学习率 |
| `method/full_diffusionv5/model.py` | V3 加载、门控读取模块、缓存坐标/时间对齐、WM 推演 |
| `method/full_diffusionv5/prepare.py` | 4Hz 锚点、三帧样本、10 点标签；当前帧感知上下文缓存 |
| `method/full_diffusionv5/data.py` | 三帧样本数据集与 2Hz 链划分 |
| `method/full_diffusionv5/train.py` | 只训练适配器的 DDP 训练、保存、断点续训、验证 |
| `method/full_diffusionv5/inference/policy.py` | 闭环推理：20Hz 规划 + 2Hz WM 刷新 + 缓存对齐 |
| `method/full_diffusionv5/inference/carla_agent.py` | Bench2Drive 智能体入口 |

## 7. 运行步骤

所有命令在仓库根目录、已有分配的计算节点上执行（脚本会检查 `SLURM_JOB_ID` 或 `LSB_JOBID`）。

### 7.1 需要准备的文件

| 文件 | 说明 |
|---|---|
| `ckps/full_diffusionv3/release/epoch_10.pt` | V3，SHA256 `ea3eea84…b027` |
| `ckps/irwm_bev_t_large/release/best.pt` | BCDE 世界模型，SHA256 `b83eba4a…03c2a`；**当前不在仓库中，需要从训练服务器拷贝** |
| `ckps/full_diffusion/backbone/all_towns_seed1/` | TF++ 骨干（`model_0030_1.pth` + `config.json`） |
| `data/irwm_bev_t/pdm_seed901_dev44_dt05_tongji_v1/` | 原始路线划分与 `metadata/*.npz` |
| `data/irwm_bev/teacher_cache/<teacher 身份>/` | 冻结 TF++ BEV 特征缓存 |
| 原始数据 `/share/home/u21012/data/bench2drive-PDMlite` | 生成上下文缓存时读取 RGB / LiDAR |

### 7.2 生成 4Hz 标签和样本（CPU）

```bash
source scripts/activate.sh
python -m method.full_diffusionv5.prepare labels --config method/full_diffusionv5/config.json --workers 16
```

输出到 `data/full_diffusionv5/pdm_seed901_4hz_v1/`。请检查 `manifest.json` 中的：

- `anchors` / `items`：锚点数和三帧样本数（锚点约为 V4 的两倍，以实际输出为准）；
- `items_missing_teacher_features`：缺少 BEV 特征缓存而被丢弃的样本数。原 `cache_fill.py` 扫描的是每条路线所有 ≥8 的帧，正常情况下奇数帧也已缓存；如果这个数不为 0，先用 `method/irwm_bev/cache_fill.py --scan` 查缺并补齐，再删除输出目录重新生成。

### 7.3 生成当前帧感知上下文（GPU）

```bash
torchrun --standalone --nproc_per_node=8 -m method.full_diffusionv5.prepare contexts \
  --config method/full_diffusionv5/config.json --workers 4 --batch-size 16
```

每个 4Hz 锚点都需要 TF++ 的 memory / ego 特征，数量约为 V4 的两倍。

### 7.4 训练

先做短跑：

```bash
torchrun --standalone --nproc_per_node=8 -m method.full_diffusionv5.train \
  --config method/full_diffusionv5/config.json \
  --output ckps/full_diffusionv5/smoke --max-updates 20
```

正式训练：

```bash
torchrun --standalone --nproc_per_node=8 -m method.full_diffusionv5.train \
  --config method/full_diffusionv5/config.json \
  --output ckps/full_diffusionv5/frozen_v3_cached_wm_seed901
```

- 断点续训：同一输出目录加 `--resume`。
- 平滑停止：在输出目录创建 `STOP` 文件。
- `train.jsonl`：三个年龄各自的空间 / 时间 L1、WM 有效行数、各层 `tanh(gate)`。gate 从 0 开始，应逐渐打开。
- `validation.jsonl`：44 条验证路线，按年龄 0 / 0.25 / 0.5 分别给出 V5 与 V3 的空间 ADE、未来 ADE（两者共用同一个初始噪声）。`best.pt` 按三个年龄的 V5 未来 ADE 平均值选择。验证使用独立固定种子的随机数生成器，不消耗训练的随机数。

可选：在 `world.tuning` 中填入 V4 `config_joint.json` 的同名配置，即可像 V4 一样微调 WM 的最后一个转移块、解码器和动作融合（学习率 1e-6）。默认冻结，以减少变量。

### 7.5 闭环评测

用与 V3 / V4 相同的 Bench2Drive 评测方式，只替换智能体和 checkpoint：

- 智能体：`method/full_diffusionv5/inference/carla_agent.py`
- 智能体配置：V5 checkpoint 文件，或训练输出目录（默认读取其中的 `best.pt`）

环境变量：

| 变量 | 作用 |
|---|---|
| `WMMPC_V5_DISABLE_WORLD=1` | 关闭 WM，结果应与原版 V3 相同，作为对照 |
| `WMMPC_DIFFUSION_INFERENCE_STEPS` | 覆盖 DDIM 步数（默认 25） |
| `WMMPC_V3_CHECKPOINT` | V3 权重的其他位置（仍校验 SHA256） |

推理逻辑：

- 时间戳对齐到 0.05s 网格，避免浮点误差导致 0.5s 先验被误判过期；
- 时间倒退或场景切换时，清空历史、先验和缓存；
- 历史 BEV 每个 tick 都更新，WM 总能拿到 −0.5、−0.25、0s 三帧；
- 35km/h 刹车上限、停车标志、防卡死等规则与 V3 相同，逐 tick 运行。

## 8. 建议的实验

| 实验 | 目的 |
|---|---|
| V3 原版（DS 85.36） | 基线 |
| V5，`WMMPC_V5_DISABLE_WORLD=1` | 检查是否与 V3 一致 |
| V5 | 主结果 |
| 不用上一计划或零动作驱动 WM（需要用相同设置重新训练） | 证明增益来自“上一计划驱动 WM 帮助当前去噪” |
| 缓存不做时间对齐 | 证明时间对齐有效 |

CARLA 闭环结果有随机波动，1~2 分的差异需要重复运行确认。融合与阈值类设置只在 44 条验证路线上调，不在 220 条测试路线上调。

## 9. 已知限制

- 没有运行任何测试；第一次训练和第一次闭环都需要排查问题。
- 推理中除 0、0.25、0.5s 外的缓存年龄没有监督。
- 缓存的未来 BEV 来自旧观测和旧计划，0.5s 内新出现的情况只能由最新当前观测发现；当前轨迹明显偏离旧计划时，缓存可信度下降，需要靠年龄和可靠性门控学会降权。
- 每个 tick 都需要完整的去噪循环，每 0.5s 额外一次 WM 推演；没有测过实际耗时。
- 坐标约定（x 向前、y 向右，±32m）沿用 V4，未单独重新核验。
