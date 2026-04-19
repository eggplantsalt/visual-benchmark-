# LIBERO-plus 技术总览（OVERVIEW）

本文档面向新成员与开发者，覆盖项目架构、快速上手与二次开发路径。内容基于当前仓库静态分析与源码行为整理，无法从仓库直接确认的部分已标记为 [需补充]。

---

## 第一部分：项目架构分析

### 1. 项目整体架构

#### 1.1 项目目录树（核心）

    LIBERO-plus/
    ├─ README.md
    ├─ requirements.txt
    ├─ extra_requirements.txt
    ├─ setup.py
    ├─ benchmark_scripts/
    │  ├─ check_task_suites.py
    │  ├─ download_libero_datasets.py
    │  ├─ render_single_task.py
    │  └─ ...
    ├─ libero/
    │  ├─ configs/
    │  │  ├─ config.yaml
    │  │  ├─ data/default.yaml
    │  │  ├─ train/default.yaml
    │  │  ├─ eval/default.yaml
    │  │  ├─ lifelong/{base,er,ewc,agem,multitask,packnet,single_task}.yaml
    │  │  └─ policy/
    │  │     ├─ bc_rnn_policy.yaml
    │  │     ├─ bc_transformer_policy.yaml
    │  │     ├─ bc_vilt_policy.yaml
    │  │     └─ image_encoder/, language_encoder/, policy_head/, data_augmentation/
    │  ├─ libero/
    │  │  ├─ __init__.py
    │  │  ├─ benchmark/
    │  │  ├─ envs/
    │  │  ├─ bddl_files/
    │  │  └─ utils/
    │  ├─ lifelong/
    │  │  ├─ main.py
    │  │  ├─ evaluate.py
    │  │  ├─ datasets.py
    │  │  ├─ metric.py
    │  │  ├─ utils.py
    │  │  ├─ algos/
    │  │  └─ models/
    │  └─ randomizer/
    ├─ scripts/
    │  ├─ collect_demonstration.py
    │  ├─ create_dataset.py
    │  └─ ...
    ├─ notebooks/
    ├─ templates/
    └─ static/

#### 1.2 核心模块与职责

- 配置层（Hydra）
  - 路径：[libero/configs/config.yaml](libero/configs/config.yaml)
  - 职责：统一组装 data / policy / train / eval / lifelong 子配置，支持命令行覆盖。
- 训练入口（lifelong）
  - 路径：[libero/lifelong/main.py](libero/lifelong/main.py)
  - 职责：加载任务基准、构建数据集、生成语言嵌入、实例化算法与策略、训练并评估。
- 评估入口（单任务/离线 checkpoint）
  - 路径：[libero/lifelong/evaluate.py](libero/lifelong/evaluate.py)
  - 职责：按指定 benchmark/task/algo/policy/seed 加载模型并计算成功率。
- Benchmark 与任务注册
  - 路径：[libero/libero/benchmark/__init__.py](libero/libero/benchmark/__init__.py)
  - 职责：任务集合注册、任务顺序管理、bddl/init_state/demo 路径解析。
- 仿真环境封装
  - 路径：[libero/libero/envs](libero/libero/envs)
  - 职责：包装离屏渲染环境与向量化并行执行接口。
- 数据集与样本组织
  - 路径：[libero/lifelong/datasets.py](libero/lifelong/datasets.py)
  - 职责：将 HDF5 演示数据封装为序列数据，注入 task embedding，支持任务分组。
- 算法与策略注册机制
  - 路径：[libero/lifelong/algos/base.py](libero/lifelong/algos/base.py)、[libero/lifelong/models/base_policy.py](libero/lifelong/models/base_policy.py)
  - 职责：通过元类自动注册算法/策略，实现按名称动态构建。

#### 1.3 模块依赖关系图

<pre class="mermaid">
flowchart LR
  A[Hydra Config\nlibero/configs] --> B[Training Entry\nlibero/lifelong/main.py]
  B --> C[Benchmark Registry\nlibero/libero/benchmark]
  B --> D[Dataset Builder\nlibero/lifelong/datasets.py]
  B --> E[Algo Registry\nlibero/lifelong/algos]
  E --> F[Policy Registry\nlibero/lifelong/models]
  B --> G[Metrics & Eval\nlibero/lifelong/metric.py]
  G --> H[Vector Env\nlibero/libero/envs]
  C --> I[BDDL + Init States + Demo Paths]
</pre>

#### 1.4 训练/评估数据流图

<pre class="mermaid">
flowchart LR
  D1[HDF5 Demonstrations] --> D2[get_dataset]
  D2 --> D3[SequenceVLDataset / GroupedTaskDataset]
  D3 --> T1[Algo.observe]
  T1 --> T2[Policy.compute_loss]
  T2 --> T3[Optimizer + Scheduler]
  T3 --> CKPT[task*_model.pth]

  CKPT --> E1[evaluate_success]
  E1 --> E2[OffScreenRenderEnv / SubprocVectorEnv]
  E2 --> E3[Rollout + Success Rate]
  E3 --> OUT[result.pt / stats]
</pre>

#### 1.5 每模块技术栈与关键依赖

- 深度学习：PyTorch
- 配置管理：Hydra + OmegaConf + EasyDict
- 视觉语言：Transformers（BERT/GPT2/CLIP/RoBERTa embedding）
- 机器人仿真：robosuite + bddl + robomimic
- 日志与实验管理：wandb（可选）
- 复杂度统计：thop

💡 提示：该项目在 Python 依赖之外还需要系统动态库（README 中列出的 apt 包），容器或 Linux 环境更稳妥。

⚠️ 警告：仓库中的 setup.py 未声明 install_requires，实际依赖以 requirements.txt 与 extra_requirements.txt 为准。

#### Checklist

- [ ] 已理解配置层与执行层分离（configs vs lifelong）
- [ ] 已了解 benchmark、dataset、algo、policy 的调用链
- [ ] 已明确训练产物与评估产物输出位置
- [ ] 已确认本地系统依赖是否满足

---

### 2. 模块详细说明

#### 2.1 训练主流程模块

- 功能描述
  - 文件：[libero/lifelong/main.py](libero/lifelong/main.py)
  - 作用：读取 Hydra 配置，加载 benchmark 各任务 demo，构建 task embedding，训练（Sequential/ER/EWC/PackNet/Multitask 等），周期性评估并保存结果。

- 接口定义与调用关系
  - 入口函数：main(hydra_cfg)
  - 关键调用链：
    - get_benchmark(cfg.benchmark_name)(task_order_index)
    - get_dataset(dataset_path, obs_modality, seq_len)
    - get_task_embs(cfg, descriptions)
    - get_algo_class(cfg.lifelong.algo)(n_tasks, cfg)
    - evaluate_loss / evaluate_success

- 关键类/函数说明
  - get_dataset：读取 HDF5 数据并返回 SequenceDataset + shape_meta
  - create_experiment_dir：自动创建 experiments/.../run_xxx
  - compute_flops：基于单 batch 估计 GFLOPs/MParams

#### 2.2 评估模块

- 功能描述
  - 文件：[libero/lifelong/evaluate.py](libero/lifelong/evaluate.py)
  - 作用：针对指定任务 id 和 checkpoint 执行并行仿真 rollout，输出 success_rate。

- 接口定义
  - 命令参数：--benchmark --task_id --algo --policy --seed 等
  - 输出：*.stats（torch.save）以及可选视频目录

- 关键点
  - benchmark/algo/policy 使用内部映射转换为类名
  - PackNet 在评估阶段会根据 task_id 处理 mask

#### 2.3 Benchmark 模块

- 功能描述
  - 文件：[libero/libero/benchmark/__init__.py](libero/libero/benchmark/__init__.py)
  - 作用：
    - 注册各 benchmark（LIBERO_SPATIAL/OBJECT/GOAL/10/90/...）
    - 管理任务顺序（含多组随机顺序）
    - 统一提供 bddl、demo、init_state 路径

- 接口
  - get_benchmark(name)
  - benchmark.get_task(i)
  - benchmark.get_task_demonstration(i)
  - benchmark.get_task_init_states(i)

#### 2.4 数据集模块

- 功能描述
  - 文件：[libero/lifelong/datasets.py](libero/lifelong/datasets.py)
  - 作用：将 robomimic SequenceDataset 包装为任务学习友好的数据接口。

- 核心类
  - SequenceVLDataset：为每条样本追加 task_emb
  - GroupedTaskDataset：按“轮转映射”均衡多任务采样
  - TruncatedSequenceDataset：用于固定 buffer 截断

#### 2.5 配置文件作用与参数说明（核心）

- 全局配置
  - 文件：[libero/configs/config.yaml](libero/configs/config.yaml)
  - 关键项：
    - benchmark_name：任务套件（如 LIBERO_SPATIAL）
    - task_embedding_format：bert/one-hot/gpt2/clip/roberta
    - device：cuda/cpu

- 数据配置
  - 文件：[libero/configs/data/default.yaml](libero/configs/data/default.yaml)
  - 关键项：
    - seq_len（默认 10）：时序长度
    - img_h/img_w（默认 128）：渲染分辨率
    - obs.modality：rgb/depth/low_dim 键
    - task_group_size（默认 1）：任务分组粒度

- 训练配置
  - 文件：[libero/configs/train/default.yaml](libero/configs/train/default.yaml)
  - 关键项：
    - n_epochs（默认 50）
    - batch_size（默认 32）
    - grad_clip（默认 100.0）
    - use_augmentation（默认 true）

- 评估配置
  - 文件：[libero/configs/eval/default.yaml](libero/configs/eval/default.yaml)
  - 关键项：
    - n_eval（默认 20）
    - eval_every（默认 5）
    - max_steps（默认 600）
    - num_procs（默认 20）

- 算法配置
  - 文件：[libero/configs/lifelong](libero/configs/lifelong)
  - 示例：
    - er.yaml：n_memories=1000
    - ewc.yaml：e_lambda=50000, gamma=0.9
    - agem.yaml：n_memories=1000

#### Checklist

- [ ] 能说明训练入口到评估入口的函数调用顺序
- [ ] 知道 benchmark 如何解析 bddl/demo/init_state
- [ ] 能解释为何 SequenceVLDataset 需要 task_emb
- [ ] 掌握核心配置文件与默认参数

---

### 3. 代码组织逻辑

#### 3.1 命名规范与文件组织原则

- 目录按职责拆分：configs（声明）、lifelong（训练逻辑）、libero/libero（仿真与基准）
- 策略与算法通过注册机制按类名动态发现，减少硬编码分支
- 脚本目录（scripts、benchmark_scripts）承载数据准备、检查与工具命令

#### 3.2 设计模式

- 注册表模式
  - 算法注册：AlgoMeta + REGISTERED_ALGOS
  - 策略注册：PolicyMeta + REGISTERED_POLICIES
- 模板方法倾向
  - Sequential 作为基类，子类覆盖 start_task/end_task/learn 策略

#### 3.3 数据流与控制流

- 控制流：Hydra 组装配置 -> main 组织训练循环 -> metric 做评估
- 数据流：HDF5 demo -> SequenceDataset -> policy 输入 -> action -> 仿真环境反馈

💡 提示：若需快速定位“模型训练发散/性能回退”问题，优先检查 data.obs 键映射与 task embedding 维度。

⚠️ 警告：路径配置依赖 ~/.libero/config.yaml，若路径失效会在运行期触发文件不存在错误。

#### Checklist

- [ ] 已掌握注册表机制如何扩展新算法/策略
- [ ] 已理解控制流与数据流的边界
- [ ] 已确认本地 ~/.libero/config.yaml 的路径有效性

---

## 第二部分：快速上手指南

### 1. 环境准备

#### 1.1 系统要求

- 推荐系统：Ubuntu 20.04/22.04（Windows 可用 WSL2）
- Python：3.8-3.10 [需补充：仓库未显式锁定]
- GPU：建议 CUDA 可用（纯 CPU 可运行但速度较慢）

#### 1.2 依赖安装步骤

    # 1) 获取代码
    git clone https://github.com/sylvestf/LIBERO-plus.git
    cd LIBERO-plus

    # 2) 安装 Python 包
    pip install -e .
    pip install -r requirements.txt
    pip install -r extra_requirements.txt

    # 3) （Linux）安装系统依赖
    sudo apt update
    sudo apt install -y libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev

    # 4) 下载并解压 assets 到 libero/libero/assets
    #    数据与模型见 README 中的 HuggingFace 链接

💡 提示：首次导入 libero 包时会初始化 ~/.libero/config.yaml，可按提示指定 datasets 目录。

#### 1.3 常见问题与解决方案

- 问题：找不到 bddl/init_states/datasets
  - 处理：检查 ~/.libero/config.yaml 各路径是否存在
- 问题：评估阶段环境创建失败
  - 处理：降低 eval.num_procs，确保图形/渲染依赖完整
- 问题：transformers 下载慢或失败
  - 处理：预下载模型或配置镜像源 [需补充：团队镜像地址]
- 问题：Windows 直接运行仿真异常
  - 处理：优先在 WSL2 + Ubuntu 执行

#### Checklist

- [ ] 已完成 Python 与系统依赖安装
- [ ] 已下载 assets 并放置到正确目录
- [ ] 已确认 ~/.libero/config.yaml 路径有效
- [ ] 已完成一次最小化脚本启动验证

---

### 2. 项目启动流程

#### 2.1 启动命令序列（训练）

    # 方式 A：通过入口脚本
    lifelong.main benchmark_name=LIBERO_SPATIAL lifelong=base policy=bc_transformer_policy

    # 方式 B：直接 python 执行
    python -m libero.lifelong.main benchmark_name=LIBERO_SPATIAL lifelong=base policy=bc_transformer_policy

#### 2.2 参数含义与可选值

- benchmark_name
  - 常见值：LIBERO_SPATIAL / LIBERO_OBJECT / LIBERO_GOAL / LIBERO_10 / LIBERO_90
- lifelong
  - base / er / ewc / agem / packnet / multitask / single_task
- policy
  - bc_rnn_policy / bc_transformer_policy / bc_vilt_policy
- device
  - cuda / cpu
- use_wandb
  - true / false

#### 2.3 多场景启动实例

    # 1) ER 连续学习 + Transformer 策略
    lifelong.main benchmark_name=LIBERO_OBJECT lifelong=er policy=bc_transformer_policy device=cuda

    # 2) EWC + RNN 策略
    lifelong.main benchmark_name=LIBERO_GOAL lifelong=ewc policy=bc_rnn_policy

    # 3) Multitask 联合训练
    lifelong.main benchmark_name=LIBERO_10 lifelong=multitask policy=bc_vilt_policy

    # 4) 单任务评估脚本
    python -m libero.lifelong.evaluate --benchmark libero_spatial --task_id 0 --algo base --policy bc_transformer_policy --seed 10000 --device_id 0 --load_task 0

⚠️ 警告：README 指出在 LIBERO-plus 评估场景中，num_trials_per_task 建议从 50 调整为 1（若沿用原 LIBERO 评估脚本配置）。

#### Checklist

- [ ] 能使用 Hydra override 启动训练
- [ ] 能切换不同 lifelong 算法与 policy
- [ ] 能运行一次 evaluate.py 并拿到 success_rate
- [ ] 已核对评估 trial 配置是否符合 LIBERO-plus 规范

---

### 3. 训练参数配置

#### 3.1 参数列表（核心参数）

| 参数 | 默认值 | 典型范围 | 功能影响 | 配置位置 |
|---|---:|---|---|---|
| train.n_epochs | 50 | 10-200 | 训练轮数，影响收敛与耗时 | [libero/configs/train/default.yaml](libero/configs/train/default.yaml) |
| train.batch_size | 32 | 8-256 | 显存占用与梯度稳定性 | [libero/configs/train/default.yaml](libero/configs/train/default.yaml) |
| train.grad_clip | 100.0 | 1-200 | 抑制梯度爆炸 | [libero/configs/train/default.yaml](libero/configs/train/default.yaml) |
| data.seq_len | 10 | 1-20 | 时序建模长度 | [libero/configs/data/default.yaml](libero/configs/data/default.yaml) |
| data.img_h/img_w | 128 | 84-256 | 观测分辨率与速度 | [libero/configs/data/default.yaml](libero/configs/data/default.yaml) |
| eval.n_eval | 20 | 10-100 | 成功率估计方差 | [libero/configs/eval/default.yaml](libero/configs/eval/default.yaml) |
| eval.max_steps | 600 | 100-1000 | rollout 上限步数 | [libero/configs/eval/default.yaml](libero/configs/eval/default.yaml) |
| lifelong.n_memories (ER/AGEM) | 1000 | 100-50000 | 回放缓冲容量 | [libero/configs/lifelong/er.yaml](libero/configs/lifelong/er.yaml) |
| lifelong.e_lambda (EWC) | 50000 | 1e3-1e6 | 正则强度 | [libero/configs/lifelong/ewc.yaml](libero/configs/lifelong/ewc.yaml) |

#### 3.2 常用参数模板

    # 快速验证（低成本）
    lifelong.main benchmark_name=LIBERO_10 train.n_epochs=5 eval.n_eval=5 eval.num_procs=2 train.batch_size=8

    # 标准训练（单卡）
    lifelong.main benchmark_name=LIBERO_SPATIAL lifelong=er policy=bc_transformer_policy train.batch_size=32 eval.num_procs=20

    # 稳定性优先（降低并行，避免环境创建失败）
    lifelong.main benchmark_name=LIBERO_OBJECT eval.num_procs=4 eval.n_eval=20

#### 3.3 调优建议与最佳实践

- 显存紧张时优先下调 train.batch_size，再考虑降低 seq_len
- 评估抖动大时提升 eval.n_eval，而非盲目延长 max_steps
- 新算法先在 LIBERO_10 做烟囱测试，再迁移到大任务集
- 固定随机种子并保存 config.json，便于实验复现

💡 提示：create_experiment_dir 会自动递增 run_xxx，适合多轮实验并行记录。

#### Checklist

- [ ] 已掌握核心参数对速度/稳定性/效果的影响
- [ ] 已准备至少一个快速验证模板和一个标准模板
- [ ] 已规划调参顺序（batch_size -> seq_len -> eval 配置）

---

## 第三部分：深入学习路线

### 1. 代码阅读顺序

#### 1.1 推荐阅读路径

1. [README.md](README.md)
2. [libero/configs/config.yaml](libero/configs/config.yaml)
3. [libero/lifelong/main.py](libero/lifelong/main.py)
4. [libero/lifelong/datasets.py](libero/lifelong/datasets.py)
5. [libero/lifelong/algos/base.py](libero/lifelong/algos/base.py)
6. [libero/lifelong/models/base_policy.py](libero/lifelong/models/base_policy.py)
7. [libero/lifelong/metric.py](libero/lifelong/metric.py)
8. [libero/libero/benchmark/__init__.py](libero/libero/benchmark/__init__.py)

#### 1.2 必读与可选

- 必读
  - 训练入口、数据封装、算法/策略基类、评估逻辑
- 可选
  - scripts/ 与 benchmark_scripts/ 工具脚本
  - notebooks/ 示例

#### 1.3 分阶段学习目标

- 阶段一（1-2 天）：跑通训练与评估，理解配置覆盖
- 阶段二（3-5 天）：读懂算法基类与策略输入输出约定
- 阶段三（1 周+）：实现一个新算法或新策略并完成回归评估

#### Checklist

- [ ] 已完成推荐顺序的首次通读
- [ ] 能讲清一条完整的训练到评估调用链
- [ ] 已建立个人调试脚本或运行模板

---

### 2. 核心概念理解

#### 2.1 关键术语

- Lifelong Learning：按任务序列逐步学习，关注前向迁移与遗忘
- Task Embedding：将语言指令编码为向量并注入策略
- SequenceDataset：基于演示轨迹按时序切片采样
- Success Rate：在固定初始化集上 rollout 的成功比例

#### 2.2 核心算法/业务逻辑

- Sequential：连续微调基线
- ER/AGEM：经验回放抑制遗忘
- EWC：参数重要性正则约束
- PackNet：参数掩码分配与任务隔离
- Multitask：联合训练全部任务

#### 2.3 理论基础与参考资料

- LIBERO/LIBERO-plus 论文与基准定义（见 README）
- Continual Learning 基础（ER/EWC/AGEM）[需补充：团队指定阅读清单]
- robomimic/robosuite 数据与环境接口文档 [需补充：内部 Wiki 链接]

#### Checklist

- [ ] 能解释 task embedding 如何影响策略前向
- [ ] 能对比 ER/EWC/PackNet 的机制差异
- [ ] 能描述 success_rate 评估方式与局限

---

### 3. 二次开发指南

#### 3.1 可扩展点与自定义方法

- 新算法
  - 在 [libero/lifelong/algos](libero/lifelong/algos) 新增类并继承 Sequential
  - 通过 AlgoMeta 自动注册后可在配置中直接引用
- 新策略
  - 在 [libero/lifelong/models](libero/lifelong/models) 新增策略类并继承 BasePolicy
  - 在 policy YAML 声明 encoder/head 组合
- 新任务套件
  - 扩展 benchmark task_map 与 bddl/init_states/demo 组织

#### 3.2 贡献规范与开发流程

1. 新建分支并最小化改动范围
2. 先在 LIBERO_10 跑快速回归
3. 保存配置与日志（config.json/result.pt）
4. 提交 PR，附关键指标和复现实验命令

#### 3.3 调试技巧与测试方法

- 调试技巧
  - 先执行 benchmark_scripts/check_task_suites.py 验证资源完整性
  - 将 eval.num_procs 调小以减少并行渲染问题
  - 检查 cfg.data.obs_key_mapping 与环境输出键一致性
- 测试方法
  - 功能测试：最小训练 + 单任务评估
  - 回归测试：固定 seed 对比 success_rate
  - 资源测试：采样检查 demo 文件可读性与 shape_meta

#### 3.4 二开示例（2-3 个）

示例 A：新增一个简单算法（基于 Sequential）

    # 文件: libero/lifelong/algos/my_algo.py
    # 说明: 继承 Sequential，覆写 end_task 做任务收尾
    from libero.lifelong.algos.base import Sequential

    class MyAlgo(Sequential):
        def end_task(self, dataset, task_id, benchmark, env=None):
            # 在这里添加自定义统计/缓存逻辑
            pass

    # 使用方式（需在 __init__.py 导入后触发注册）
    # lifelong.main lifelong=my_algo [需补充: 对应 YAML]

示例 B：切换语言嵌入模型进行对比实验

    # 保持其余参数不变，仅切换 task embedding
    lifelong.main benchmark_name=LIBERO_SPATIAL task_embedding_format=clip
    lifelong.main benchmark_name=LIBERO_SPATIAL task_embedding_format=roberta

示例 C：添加新观察模态映射

    # 文件: libero/configs/data/default.yaml
    # 将新观测键加入 obs.modality，并在 obs_key_mapping 建立映射
    obs:
      modality:
        rgb: [agentview_rgb, eye_in_hand_rgb]
        depth: []
        low_dim: [gripper_states, joint_states]

    obs_key_mapping:
      agentview_rgb: agentview_image
      eye_in_hand_rgb: robot0_eye_in_hand_image
      gripper_states: robot0_gripper_qpos
      joint_states: robot0_joint_pos

⚠️ 警告：新增观测键后若未同步更新 shape_meta 与 encoder 输入，会在训练前向时报维度错误。

#### Checklist

- [ ] 已识别至少一个算法扩展点与一个策略扩展点
- [ ] 已建立最小回归验证命令
- [ ] 已准备 PR 所需实验记录与复现信息
- [ ] 已完成 1 个二开示例本地验证

---

## 附录：可直接执行的常用命令清单

    # 安装
    pip install -e .
    pip install -r requirements.txt
    pip install -r extra_requirements.txt

    # 训练
    lifelong.main benchmark_name=LIBERO_SPATIAL lifelong=base policy=bc_transformer_policy

    # 评估
    python -m libero.lifelong.evaluate --benchmark libero_spatial --task_id 0 --algo base --policy bc_transformer_policy --seed 10000 --device_id 0 --load_task 0

    # 资源完整性检查
    python benchmark_scripts/check_task_suites.py

💡 提示：如需对接 CI，可先将训练命令替换为 train.n_epochs=1 的烟囱测试版本，以缩短反馈周期。