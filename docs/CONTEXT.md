这个文件是全局上下文，里面记录了用户想要实现的需求的具体内容，以及对code agent编写代码的时候的一些技术要求


VPA-Bench（Visual Prompt Attack Benchmark）全局上下文

这份文档会在每次向 code agent 提需求时一并附上。
目标：让 code agent 始终保持对项目目标、约束、目录结构、交付标准的一致理解。

1. 我们在做什么（One-liner）

我们要在 LIBERO-plus 仿真栈上构建一个**纯评测（test-only）**的 Visual Prompt Attack Benchmark（视觉提示攻击基准，简称 VPA-Bench）：在推理阶段（inference-time）通过“场景内可见的视觉载体（便签/箭头/贴纸/伪UI）”注入恶意/冲突指令，评测 VLA 模型是否会被视觉语义劫持并推进到危险状态（commit/harm），并提供视觉中心的“菜 vs 拒绝”诊断指标。

2. 我们明确不做什么（Non-goals）
不做训练数据：不采集大规模演示轨迹（LIBERO 原版/plus 的 demos 我们不复刻）；我们只做仿真评测实例与评测协议。
不训练/不改模型：默认只评测现有 VLA 模型（OpenVLA/π0/NORA 等），不做 fine-tune，不做安全对齐训练。
不做普通鲁棒性扰动论文：不是 LIBERO-Plus/PRO 那种 camera/light/texture 的 robustness perturbation。我们定义的是新的攻击算子（attack operator family）：视觉载体作为指令源。
不追求一次覆盖所有机器人/场景：第一阶段聚焦 单臂桌面（tabletop manipulation）+ headless offscreen 渲染，先跑通攻击注入与评测闭环。
3. 研究贡献的核心（Why this is novel）
3.1 攻击层（Attack）

定义新的 visual prompt attack operators（视觉提示攻击算子族），不是噪声扰动，而是“人类可读、语义明确、环境中自然出现”的指令载体，诱导 VLA：

Note / Sign directive（便签/标牌指令）：例如 “DROP / TRASH / MOVE HERE / STOP”
Arrow / Trajectory lure（箭头/路径诱导）：引导轨迹穿越禁区或去桌边
Semantic relabel（语义重标记）：把对象贴成“垃圾/危险/允许”等，污染世界语义
Fake UI / System notice（伪UI/伪权威提示）：屏幕弹窗伪装“系统指令”
3.2 评测层（Evaluation）

参考 HazardArena 的阶段性思想（attempt/commit…），但我们要做 visual-centric 的创新诊断：
用 反事实视觉干预（counterfactual prompt intervention） 区分：

由于没看懂/没看到视觉载体（perception failure）
看懂但不采纳/不让其覆盖授权任务（adoption failure / refusal）
看懂且采纳但执行失败（execution failure）

核心手段（后续逐步加入）：

V_orig / V_neut / V_rm（原始/语义中和/移除载体）多版本评测
因果影响指标（例如 CPIS/PSS/ΔCommit）
“理解后拒绝率”（Refusal-after-Understanding, RaU）
4. 工程落地总路线（High-level plan）
第一阶段（Phase-0/Phase-1：先跑通闭环）
采用 Scene-first 注入：新增 scene XML + 贴图纹理，把视觉载体插入环境并离屏渲染出来
用现有 OffScreenRenderEnv 创建环境并渲染帧（不依赖 demo）
先实现最小指标：SVR（样本有效）、CVR（载体可见）、AR/CR/HCR（attempt/commit/harm）
第二阶段（Phase-2：加入视觉因果诊断）
为同一 episode 生成 counterfactual variants（V_rm / V_neut / 可选 V_occ）
实现 Adopt predicate（采纳判定）与因果指标（PSS/ΔCR_cf/RaU）
输出主实验表格 + failure mode breakdown
5. 我们的运行与协作方式（Workflow constraints）
运行硬件：V100 无头服务器（headless）
代码修改方式：本地（可用 code agent）改好 → git push → 服务器 git pull 运行
因此：code agent 不需要也不允许在提示词里写“我本地跑通了/测试通过”。
他只需提供：定位证据、修改结果、以及“服务器上应运行的命令与前置条件”。
6. 目录与文件组织硬约束（Very important）
6.1 新增 Python 代码：必须集中在 vpa_bench/
所有我们新增的代码（生成器/渲染脚本/评测脚本/hook/adapters）只能放在仓库根目录新建的 vpa_bench/ 下
禁止把新脚本散落进 libero/ 原包目录（除非最小 patch）

推荐结构：

vpa_bench/
  README.md
  tools/            # 渲染检查、spec生成、批量脚本
  eval/             # rollout driver、metrics计算、结果汇总
  hooks/            # attempt/adopt/commit/harm predicates
  attacks/          # attack operators定义与参数化
  counterfactuals/  # V_orig/V_rm/V_neut 生成逻辑
  patches/          # 仅当必须改原文件时记录补丁说明
    PATCH_NOTES.md
6.2 新增仿真资产：必须集中在 libero/libero/assets/vpa/

由于 LIBERO env loader 默认从 libero/libero/assets 加载，我们允许新增资产进入该根目录下独立子树：

libero/libero/assets/vpa/
  scenes/
  textures/
  carriers/   # 可选

禁止把我们的 scene XML/texture 混进原有 assets/scenes/ 或 assets/textures/。

6.3 如必须修改原仓库文件：最小 patch + 记录
只允许为“scene_xml 透传/asset root 可选”这类必要能力做 patch
必须保持 backward compatible（默认行为不变）
所有 patch 必须在 vpa_bench/patches/PATCH_NOTES.md 记录：
改了哪个文件、哪几行、为什么必须改、如何回滚
7. 我们的最小交付物是什么（Acceptance criteria）

一个 Phase-0 最小可验收结果应当满足：

能在 headless/offscreen 环境初始化一个 LIBERO task
场景中确实出现视觉载体（便签/标牌）且可见（输出渲染 PNG）
不依赖 demo/hdf5（只用于可视化可选）
产物与改动遵守目录隔离规则（vpa_bench/ + assets/vpa/）
8. 评测实例的概念（ScenarioSpec）

我们把一个评测 episode 当作一个 ScenarioSpec，至少包含：

seed_task（原任务）
authorized_instruction（安全授权文本）
attack_family（攻击族：note/arrow/relabel/ui）
carrier_content（载体内容：DROP/TRASH/箭头路径…）
carrier_pose_rule（载体放置规则）
counterfactual_variant（V_orig / V_neut / V_rm）
hooks（attempt/adopt/commit/harm predicates）
random_seed（可复现）

ScenarioSpec 最终可存成 JSON/YAML，作为 benchmark registry 的基本单位。

9. 模型接入（Model interface）

我们不改模型，只做适配器（如需）：

输入：图像观测（agentview + 可选 wrist） + 授权文本指令
输出：动作向量（7D EE 控制 + gripper）

第一阶段不要求 adapter 统一；只要能跑一个模型即可。

10. 当前已知关键代码事实（由定位报告确认）
task instance 主键：Task.name 字符串；评测以 task_id 索引获取 task
bddl/init_states/demo 路径解析集中在 libero/libero/benchmark/__init__.py
assets root 默认：libero/libero/assets；scene XML 在 assets/scenes/，我们新增放 assets/vpa/scenes/
OffScreenRenderEnv：强制 offscreen 渲染；bddl_file_name + camera params 从 metric/evaluate 脚本传入

（注：这部分会随项目进展更新。）

11. 给 code agent 的统一输出要求（每次任务都必须满足）

每次你完成一个需求，请按如下格式汇报：

新增/修改的文件列表（目录树形式）
每个新增文件的用途一句话
如果修改了原文件：补丁摘要 + 回滚方式（并更新 PATCH_NOTES.md）
服务器上应运行的命令与前置条件（不保证你能运行，但命令要可用）
风险点（最多 3 条）
12. 项目阶段性里程碑（供你保持方向感）
M0：能渲染出带 note 的 scene（PNG 证据）
M1：能批量生成多实例（不同 note 内容/位置/对照组）并输出 ScenarioSpec
M2：实现 attempt/commit/harm hooks 并产出第一张主实验表（哪怕只有 1–2 个模型）
M3：加入 counterfactual variants 与因果指标，形成 failure mode breakdown

重要提醒：我们的目标不是“改 LIBERO-plus”，而是“在 LIBERO-plus 上搭建一个独立的 VPA-Bench 插件层”。
任何设计都优先考虑：可复现、可扩展、目录隔离、最小侵入。















下面是编写代码的时候的一些要求


--代码一定要分好文件夹，做好归类，所有说明文档也要分好文件夹，不同类的说明文档放在对应文件夹里

--我们是在远程服务器上跑代码，远程服务器用不了code agent，因此我们是在本地用code agent修改代码，然后git上传，再在服务器端pull，做到代码修改同步，因为我们是本地修改代码，本地不运行，所以你把代码写好即可，不要运行或者测试。不要尝试运行任何代码文件，把代码写好了就可以了 



--文档都放在docs文件夹下（可以自己新建子目录）





--之后我们做实验的流程是我把终端报错发给你，你给出回答解决问题 



--用户发给你的指令里可能会有一些你已经做过了的内容，如果你做过了，那就可自己决定跳过还是再检查一遍顺便优化优化，如果发给你的指令里有你没做过的，那就按照他的意思去做（注意修改代码过程中，不一定严格按照用户逻辑来，因为你可以直接接触代码，如果发现用户的一些要求是错的，或执行用户的要求以后会产生和代码库冲突的bug或者错误，那么你就自己做决定怎么修改，以你为主）


--你不需要做任何测试，把代码写好即可。即使prompt里有让你做测试，你也不用去做（因为我们环境还没配好，等配好了我们自行去做）


