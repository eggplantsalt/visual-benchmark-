# LIBERO-plus 定位报告：benchmark 生成与评测链路

## Q1：10,030 tasks 的生成入口与输出目录
- 结论：
  - 这个仓库里未找到“10,030 tasks 一键生成主脚本”；可复现证据表明当前实现是“预生成任务清单 + 运行时注册映射 + 数据集下载”，其中 10,030 对应四个 suite（spatial/object/goal/10）的任务数之和，90 任务的 libero_90 另算。
- 证据链：
  - 位置：README.md:27
  - 关键符号：10,030 tasks 文案
  - 代码片段：
    
    We introduce LIBERO-plus, a comprehensive benchmark with 10,030 tasks spanning:
    
  - 位置：libero/libero/benchmark/__init__.py:75
  - 关键符号：libero_suites, task_num
  - 代码片段：
    
    libero_suites = [
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_90",
        "libero_10",
    ]
    ...
    task_num = [2402, 2518, 2591, 2519, 90]
    
  - 位置：libero/libero/benchmark/libero_suite_task_map.py:1
  - 关键符号：libero_task_map
  - 代码片段：
    
    libero_task_map = {
        "libero_spatial": [
            "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_table_1",
            ...
    
  - 位置：benchmark_scripts/download_libero_datasets.py:31
  - 关键符号：main, download_utils.libero_dataset_download
  - 代码片段：
    
    def main():
        args = parse_args()
        os.makedirs(args.download_dir, exist_ok=True)
        ...
        download_utils.libero_dataset_download(
            download_dir=args.download_dir, 
            datasets=args.datasets,
            use_huggingface=args.use_huggingface
        )
    
  - 位置：libero/libero/utils/download_utils.py:158
  - 关键符号：libero_dataset_download, DATASET_LINKS, snapshot_download, extractall
  - 代码片段：
    
    DATASET_LINKS = {
        "libero_object": "...zip",
        "libero_goal": "...zip",
        "libero_spatial": "...zip",
        "libero_100": "...zip",
    }
    ...
    def libero_dataset_download(...):
        ...
        if use_huggingface:
            download_from_huggingface(...)
        else:
            download_url(...)
    
    def download_url(...):
        ...
        with zipfile.ZipFile(file_to_write, "r") as archive:
            archive.extractall(path=download_dir)
    
  - 位置：libero/libero/utils/task_generation_utils.py:54
  - 关键符号：generate_bddl_from_task_info
  - 代码片段：
    
    def generate_bddl_from_task_info(folder="/tmp/pddl"):
        ...
        bddl_file_name = save_to_file(..., folder=folder)
    
  - 位置：libero/randomizer/bddl_operators.py:439
  - 关键符号：save_bddl_file
  - 代码片段：
    
    def save_bddl_file(..., bddl_dirpath: str = "/tmp/bddl_files"):
        ...
        with open(bddl_filepath, "w") as f:
            f.write(bddl_str)
    
- 定位复现命令：
    
    rg -n "10,030|10030|task_num =|libero_task_map" README.md libero/libero/benchmark/__init__.py libero/libero/benchmark/libero_suite_task_map.py
    rg -n "def main|libero_dataset_download|--download-dir|--datasets|--use-huggingface" benchmark_scripts/download_libero_datasets.py
    rg -n "DATASET_LINKS|HF_REPO_ID|snapshot_download|extractall|def libero_dataset_download" libero/libero/utils/download_utils.py
    rg -n "def generate_bddl_from_task_info|folder=\"/tmp/pddl\"" libero/libero/utils/task_generation_utils.py
    rg -n "def save_bddl_file|bddl_dirpath" libero/randomizer/bddl_operators.py
    python - << 'PY'
    task_num = [2402, 2518, 2591, 2519, 90]
    print("4 suites sum =", sum(task_num[:4]))
    print("all 5 suites sum =", sum(task_num))
    PY
    
- 服务器运行命令建议（V100无头）：
    
    conda activate <your_env>
    cd <repo_root>
    pip install -e .
    pip install -r requirements.txt
    pip install -r extra_requirements.txt
    
    # 数据下载（推荐 HF）
    python benchmark_scripts/download_libero_datasets.py --datasets all --use-huggingface
    
    # 资源路径配置（首次会生成 ~/.libero/config.yaml）
    python - << 'PY'
    from libero.libero import get_libero_path
    print("datasets =", get_libero_path("datasets"))
    print("bddl_files =", get_libero_path("bddl_files"))
    print("init_states =", get_libero_path("init_states"))
    print("assets =", get_libero_path("assets"))
    PY
    
  - 需要的数据/资源路径：
    - ~/.libero/config.yaml
    - bddl_files 路径（默认仓库内 libero/libero/bddl_files）
    - init_states 路径（默认仓库内 libero/libero/init_files）
    - assets 路径（默认仓库内 libero/libero/assets，需要你们下载并解压）
    - datasets 路径（下载脚本目标目录）
  - 依赖环境变量/conda 约定：
    - 可选：LIBERO_CONFIG_PATH（覆盖 ~/.libero）
    - 代码里设置：TOKENIZERS_PARALLELISM=false（在训练/评估入口）
  - 常见失败点（最可能 2-3 个）：
    - assets 未放到 config.yaml 指向目录，导致 XML/mesh/texture 找不到
    - datasets 不完整（hdf5 缺失）
    - 下载源不可达（原始链接过期，需使用 --use-huggingface）

## Q2：task instance 的唯一 ID
- 结论：
  - 运行时唯一标识是 task name 字符串（Task.name，来自 libero_task_map 条目）；评测入口参数使用 task_id（索引）定位到该 Task；task_classification.json 里的 id 是额外分类编号，不是运行时主键。
- 证据链：
  - 位置：libero/libero/benchmark/__init__.py:37
  - 关键符号：Task(NamedTuple)
  - 代码片段：
    
    class Task(NamedTuple):
        name: str
        language: str
        problem: str
        problem_folder: str
        bddl_file: str
        init_states_file: str
    
  - 位置：libero/libero/benchmark/__init__.py:87
  - 关键符号：for task in libero_task_map[libero_suite]
  - 代码片段：
    
    for task in libero_task_map[libero_suite]:
        ...
        task_maps[libero_suite][task] = Task(
            name=task,
            ...
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
        )
    
  - 位置：libero/lifelong/evaluate.py:76
  - 关键符号：--task_id, benchmark.get_task(args.task_id)
  - 代码片段：
    
    parser.add_argument("--task_id", type=int, required=True)
    ...
    task = benchmark.get_task(args.task_id)
    
  - 位置：benchmark_scripts/render_single_task.py:41
  - 关键符号：--task_id, benchmark_instance.get_task(task_id)
  - 代码片段：
    
    parser.add_argument("--task_id", type=int, default=0)
    ...
    task = benchmark_instance.get_task(task_id)
    
  - 位置：libero/libero/benchmark/task_classification.json:1
  - 关键符号：id, name, difficulty_level
  - 代码片段：
    
    {
      "libero_spatial": [
        {
          "id": 1,
          "name": "..._table_1",
          "category": "Background Textures",
          "difficulty_level": 2
        },
    
- 定位复现命令：
    
    rg -n "class Task|name=task|bddl_file=f\"{task}.bddl\"|init_states_file=f\"{task}.pruned_init\"" libero/libero/benchmark/__init__.py
    rg -n "--task_id|get_task\(args.task_id\)|get_task\(task_id\)" libero/lifelong/evaluate.py benchmark_scripts/render_single_task.py
    sed -n '1,40p' libero/libero/benchmark/task_classification.json
    
- 服务器运行命令建议（V100无头）：
    
    conda activate <your_env>
    cd <repo_root>
    
    # 查看某 suite 的 task_id -> task.name
    python - << 'PY'
    from libero.libero.benchmark import get_benchmark
    b = get_benchmark("libero_spatial")(0)
    for i in [0,1,2]:
        t = b.get_task(i)
        print(i, t.name, t.bddl_file, t.init_states_file)
    PY
    
  - 需要的数据/资源路径：
    - ~/.libero/config.yaml 中 bddl_files/init_states/datasets 必须有效
  - 依赖环境变量/conda 约定：
    - 可选：LIBERO_CONFIG_PATH
  - 常见失败点：
    - suite 名大小写不一致（注册表按 lower 取）
    - task_id 越界

## Q3：bddl/init_states/demo 映射定义位置
- 结论：
  - 映射核心在 benchmark 注册层定义：Task 保存 bddl_file 与 init_states_file，demo 路径由 get_task_demonstration 拼接；init_state 路径在 get_task_init_states 中按后缀规则重写。
- 证据链：
  - 位置：libero/libero/benchmark/__init__.py:138
  - 关键符号：get_task_bddl_file_path
  - 代码片段：
    
    def get_task_bddl_file_path(self, i):
        bddl_file_path = os.path.join(
            get_libero_path("bddl_files"),
            self.tasks[i].problem_folder,
            self.tasks[i].bddl_file,
        )
        return bddl_file_path
    
  - 位置：libero/libero/benchmark/__init__.py:146
  - 关键符号：get_task_demonstration
  - 代码片段：
    
    def get_task_demonstration(self, i):
        ...
        demo_path = f"{self.tasks[i].problem_folder}/{self.tasks[i].name}_demo.hdf5"
        return demo_path
    
  - 位置：libero/libero/benchmark/__init__.py:189
  - 关键符号：get_task_init_states
  - 代码片段：
    
    def get_task_init_states(self, i):
        if "_language_" in self.tasks[i].init_states_file:
            init_states_path = os.path.join(... split("_language_")[0] + ...)
        else:
            if "_view_" in self.tasks[i].init_states_file:
                init_states_path = os.path.join(... split("_view_")[0] + ...)
            else:
                if "_table_" in self.tasks[i].init_states_file:
                    init_states_path = os.path.join(... re.sub(r'_table_\\d+', '', ...))
                if "_tb_" in self.tasks[i].init_states_file:
                    init_states_path = os.path.join(... re.sub(r'_tb_\\d+', '', ...))
                if "_light_" in self.tasks[i].init_states_file:
                    init_states_path = os.path.join(... split("_light_")[0] + ...)
                if "_add_" in self.tasks[i].init_states_file or "_level" in self.tasks[i].init_states_file:
                    init_states_path = os.path.join(get_libero_path("init_states"), "libero_newobj", ...)
    
  - 位置：libero/lifelong/main.py:57
  - 关键符号：cfg.folder, cfg.bddl_folder, cfg.init_states_folder
  - 代码片段：
    
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    
- 定位复现命令：
    
    rg -n "def get_task_bddl_file_path|def get_task_demonstration|def get_task_init_states" libero/libero/benchmark/__init__.py
    rg -n "cfg\.folder|get_libero_path\(\"datasets\"\)|cfg\.bddl_folder|cfg\.init_states_folder" libero/lifelong/main.py libero/lifelong/evaluate.py
    sed -n '130,240p' libero/libero/benchmark/__init__.py
    
- 服务器运行命令建议（V100无头）：
    
    conda activate <your_env>
    cd <repo_root>
    python - << 'PY'
    from libero.libero.benchmark import get_benchmark
    from libero.libero import get_libero_path
    b = get_benchmark("libero_spatial")(0)
    i = 0
    t = b.get_task(i)
    print("task.name =", t.name)
    print("bddl =", b.get_task_bddl_file_path(i))
    print("demo =", get_libero_path("datasets") + "/" + b.get_task_demonstration(i))
    print("init_state_file =", t.init_states_file)
    PY
    
  - 需要的数据/资源路径：
    - ~/.libero/config.yaml 中 datasets/bddl_files/init_states
  - 依赖环境变量/conda 约定：
    - 可选：LIBERO_CONFIG_PATH
  - 常见失败点：
    - init_states 文件名后缀与替换规则不匹配
    - datasets 根目录配置错误导致 demo 路径失效

## Q4：scene XML / assets 目录与 loader 引用
- 结论：
  - scene XML 运行时由 BDDLBaseDomain 通过 custom_asset_dir + scene_xml 解析，默认根在 libero/libero/assets；对象 XML 分别从 assets/stable_hope_objects、assets/stable_scanned_objects、assets/articulated_objects、assets/turbosquid_objects、assets/new_objects 加载。
- 证据链：
  - 位置：libero/libero/envs/bddl_base_domain.py:123
  - 关键符号：custom_asset_dir, scene_xml, _arena_xml
  - 代码片段：
    
    self.custom_asset_dir = os.path.abspath(os.path.join(DIR_PATH, "../assets"))
    ...
    scene_xml="scenes/libero_base_style.xml",
    ...
    self._arena_xml = os.path.join(self.custom_asset_dir, scene_xml)
    
  - 位置：libero/libero/envs/problems/libero_tabletop_manipulation.py:159
  - 关键符号：scene_xml 默认值
  - 代码片段：
    
    if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
        kwargs.update({"scene_xml": "scenes/libero_tabletop_base_style.xml"})
    
  - 位置：libero/libero/envs/problems/libero_kitchen_tabletop_manipulation.py:153
  - 关键符号：scene_xml 默认值
  - 代码片段：
    
    if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
        kwargs.update({"scene_xml": "scenes/libero_kitchen_tabletop_base_style.xml"})
    
  - 位置：libero/libero/envs/objects/hope_objects.py:19
  - 关键符号：assets/stable_hope_objects
  - 代码片段：
    
    f"assets/stable_hope_objects/{obj_name}/{obj_name}.xml"
    
  - 位置：libero/libero/envs/objects/google_scanned_objects.py:23
  - 关键符号：assets/stable_scanned_objects
  - 代码片段：
    
    f"assets/stable_scanned_objects/{obj_name}/{obj_name}.xml"
    
  - 位置：libero/libero/envs/objects/articulated_objects.py:23
  - 关键符号：assets/articulated_objects
  - 代码片段：
    
    f"assets/articulated_objects/{obj_name}.xml"
    
  - 位置：libero/libero/envs/objects/turbosquid_objects.py:23
  - 关键符号：assets/turbosquid_objects
  - 代码片段：
    
    f"assets/turbosquid_objects/{obj_name}/{obj_name}.xml"
    
  - 位置：libero/libero/envs/objects/custom_objects.py:112
  - 关键符号：assets/new_objects
  - 代码片段：
    
    f"assets/new_objects/.../usd/MJCF/...xml"
    
  - 位置：README.md:63
  - 关键符号：assets 目录结构说明
  - 代码片段：
    
    assets/
    ├── articulated_objects/
    ├── new_objects/
    ├── scenes/
    ├── stable_hope_objects/
    ├── stable_scanned_objects/
    ├── textures/
    ├── turbosquid_objects/
    
- 定位复现命令：
    
    rg -n "custom_asset_dir|scene_xml|_arena_xml" libero/libero/envs/bddl_base_domain.py
    rg -n "scene_xml" libero/libero/envs/problems/*.py
    rg -n "assets/stable_hope_objects|assets/stable_scanned_objects|assets/articulated_objects|assets/turbosquid_objects|assets/new_objects" libero/libero/envs/objects/*.py
    rg -n "assets/|scenes/|textures/" README.md
    ls -la libero/libero/assets
    
- 服务器运行命令建议（V100无头）：
    
    conda activate <your_env>
    cd <repo_root>
    
    # 核心目录部署检查（3-5 个必须存在）
    test -d libero/libero/assets/scenes
    test -d libero/libero/assets/textures
    test -d libero/libero/assets/stable_scanned_objects
    test -d libero/libero/assets/stable_hope_objects
    test -d libero/libero/assets/new_objects
    
    # 抽样检查 xml 是否存在
    find libero/libero/assets/scenes -name "*.xml" | head
    find libero/libero/assets/stable_scanned_objects -name "*.xml" | head
    
  - 需要的数据/资源路径：
    - assets 根目录（config.yaml 中 assets 指向）
  - 依赖环境变量/conda 约定：
    - 可选：LIBERO_CONFIG_PATH
  - 常见失败点：
    - 只下载了代码没解压 assets.zip
    - assets 路径与 ~/.libero/config.yaml 不一致
    - scene_xml 文件存在但引用的 mesh/texture 缺失

## Q5：OffScreenRenderEnv 创建链路（bddl + camera）
- 结论：
  - 创建链路是 benchmark task 对象提供 problem_folder + bddl_file，metric/evaluate/render 脚本拼成 bddl_file_name 传入 OffScreenRenderEnv；分辨率由 cfg.data.img_h/img_w 或脚本常量决定；camera_names 默认来自 ControlEnv（agentview + robot0_eye_in_hand）。
- 证据链：
  - 位置：libero/lifelong/metric.py:71
  - 关键符号：env_args, OffScreenRenderEnv(**env_args)
  - 代码片段：
    
    env_args = {
        "bddl_file_name": os.path.join(
            cfg.bddl_folder, task.problem_folder, task.bddl_file
        ),
        "camera_heights": cfg.data.img_h,
        "camera_widths": cfg.data.img_w,
    }
    ...
    env = SubprocVectorEnv(
        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
    )
    
  - 位置：libero/lifelong/evaluate.py:236
  - 关键符号：env_args 同链路
  - 代码片段：
    
    env_args = {
        "bddl_file_name": os.path.join(
            cfg.bddl_folder, task.problem_folder, task.bddl_file
        ),
        "camera_heights": cfg.data.img_h,
        "camera_widths": cfg.data.img_w,
    }
    
  - 位置：benchmark_scripts/render_single_task.py:12
  - 关键符号：env_args 常量 128x128
  - 代码片段：
    
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    }
    
  - 位置：libero/libero/envs/env_wrapper.py:195
  - 关键符号：ControlEnv.camera_names/camera_heights/camera_widths 默认值
  - 代码片段：
    
    camera_names=[
        "agentview",
        "robot0_eye_in_hand",
    ],
    camera_heights=128,
    camera_widths=128,
    
  - 位置：libero/libero/envs/env_wrapper.py:248
  - 关键符号：TASK_MAPPING[self.problem_name](...)
  - 代码片段：
    
    problem_info = BDDLUtils.get_problem_info(bddl_file_name)
    self.problem_name = problem_info["problem_name"]
    ...
    self.env = TASK_MAPPING[self.problem_name](
        bddl_file_name,
        ...
        camera_names=camera_names,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
        ...
    )
    
  - 位置：libero/libero/envs/env_wrapper.py:396
  - 关键符号：class OffScreenRenderEnv
  - 代码片段：
    
    class OffScreenRenderEnv(ControlEnv):
        def __init__(self, **kwargs):
            kwargs["has_renderer"] = False
            kwargs["has_offscreen_renderer"] = True
            super().__init__(**kwargs)
    
  - 位置：libero/configs/data/default.yaml:13
  - 关键符号：img_h, img_w
  - 代码片段：
    
    img_h: 128
    img_w: 128
    
- 定位复现命令：
    
    rg -n "OffScreenRenderEnv|bddl_file_name|camera_heights|camera_widths|camera_names" libero/lifelong/metric.py libero/lifelong/evaluate.py benchmark_scripts/render_single_task.py libero/libero/envs/env_wrapper.py
    sed -n '60,120p' libero/lifelong/metric.py
    sed -n '176,280p' libero/libero/envs/env_wrapper.py
    sed -n '1,40p' libero/configs/data/default.yaml
    
- 服务器运行命令建议（V100无头）：
    
    conda activate <your_env>
    cd <repo_root>
    export TOKENIZERS_PARALLELISM=false
    
    # 先检查路径解析
    python - << 'PY'
    from libero.libero.benchmark import get_benchmark
    from libero.libero import get_libero_path
    b = get_benchmark("libero_spatial")(0)
    t = b.get_task(0)
    print(get_libero_path("bddl_files") + "/" + t.problem_folder + "/" + t.bddl_file)
    PY
    
  - 需要的数据/资源路径：
    - bddl_files、init_states、assets（必须可读）
  - 依赖环境变量/conda 约定：
    - TOKENIZERS_PARALLELISM=false（入口已设，但可显式导出）
    - 可选：LIBERO_CONFIG_PATH
  - 常见失败点：
    - bddl_file_name 指向不存在文件
    - Offscreen 渲染依赖缺失（Mujoco/EGL 相关）
    - 相机键名和下游取图键名不一致（agentview_image）

## Q6：render_single_task.py 用法与服务器命令
- 结论：
  - 这个脚本参数只有 4 个（benchmark_name/task_id/bddl_file/demo_file），产物是 benchmark_tasks 目录下的 PNG，不输出 MP4。
- 证据链：
  - 位置：benchmark_scripts/render_single_task.py:39
  - 关键符号：ArgumentParser 参数
  - 代码片段：
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_name", type=str)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--bddl_file", type=str)
    parser.add_argument("--demo_file", type=str)
    
  - 位置：benchmark_scripts/render_single_task.py:58
  - 关键符号：输出目录
  - 代码片段：
    
    os.makedirs("benchmark_tasks", exist_ok=True)
    
  - 位置：benchmark_scripts/render_single_task.py:77
  - 关键符号：cv2.imwrite 输出格式
  - 代码片段：
    
    image_name = demo_file.split("/")[-1].replace(".hdf5", ".png")
    cv2.imwrite(f"benchmark_tasks/{image_name}", images[::-1, :, ::-1])
    
- 定位复现命令：
    
    rg -n "ArgumentParser|--benchmark_name|--task_id|--bddl_file|--demo_file|benchmark_tasks|cv2.imwrite" benchmark_scripts/render_single_task.py
    sed -n '1,120p' benchmark_scripts/render_single_task.py
    
- 服务器运行命令建议（V100无头）：
    
    conda activate <your_env>
    cd <repo_root>
    
    # 先拿到一个可用任务的 bddl/demo 路径（静态解析）
    python - << 'PY'
    from libero.libero.benchmark import get_benchmark
    from libero.libero import get_libero_path
    b = get_benchmark("libero_spatial")(0)
    i = 0
    t = b.get_task(i)
    print("BMARK=libero_spatial")
    print("TASK_ID=", i)
    print("BDDL=", get_libero_path("bddl_files") + "/" + t.problem_folder + "/" + t.bddl_file)
    print("DEMO=", get_libero_path("datasets") + "/" + b.get_task_demonstration(i))
    PY
    
    # 最小渲染命令（1 个 task，输出 1 张 png）
    python benchmark_scripts/render_single_task.py \
      --benchmark_name libero_spatial \
      --task_id 0 \
      --bddl_file <上一步输出BDDL> \
      --demo_file <上一步输出DEMO>
    
    # 输出检查
    ls -lh benchmark_tasks/*.png | head
    
  - 输出路径与产物格式：
    - 输出目录：benchmark_tasks
    - 文件格式：PNG（由 demo 文件名改后缀得到）
  - 运行前置条件：
    - ~/.libero/config.yaml 有效
    - bddl/init_states/datasets/assets 均完整
    - Python 依赖安装完成（含 robosuite/robomimic/opencv 等）
  - 常见失败点：
    - demo_file 路径传错（hdf5 不存在）
    - bddl_file 与 task_id 不匹配导致问题解析失败
    - 资源目录缺失导致 env 初始化失败

# 附录：你执行过的检索命令清单（按顺序）
- Set-Location -Path 'e:\LIBERO-plus'; rg -n "10030|10,030|LIBERO-plus|difficulty|Level-1|L1|benchmark construction|generate"
- Set-Location -Path 'e:\LIBERO-plus'; rg -n "render_single_task|OffScreenRenderEnv|bddl_file_name|init_state|demonstration|problem_folder"
- grep_search: 10030|10,030|benchmark|difficulty|Level-1|L1|LIBERO-plus
- grep_search: render_single_task|OffScreenRenderEnv|bddl_file_name|init_state|demonstration|problem_folder
- grep_search: 10030|10,030 （限定 README/index/libero/benchmark_scripts/scripts）
- grep_search: task_classification.json|classification|difficulty|Level-1|Level-5
- grep_search: def parse_args|def main|libero_dataset_download|--download-dir|--datasets|--use-huggingface
- grep_search: class Task|def get_task_demonstration|def get_task_init_states|def get_task_bddl_file_path|task_num =|libero_suites
- grep_search: class OffScreenRenderEnv|class ControlEnv|camera_names|camera_heights|camera_widths|bddl_file_name|TASK_MAPPING[self.problem_name]
- grep_search: scene_xml|custom_asset_dir|_arena_xml|assets|bddl_file_name
- grep_search: get_libero_path("datasets")|get_libero_path("bddl_files")|get_libero_path("init_states")|OffScreenRenderEnv|camera_heights|camera_widths|bddl_file_name
- grep_search: parse_args|ArgumentParser|--benchmark_name|--task_id|--bddl_file|--demo_file|cv2.imwrite|benchmark_tasks
- grep_search: class |def |write|save|dump|bddl|scene|xml|output|task|generate （限定 libero/randomizer/bddl_operators.py）
- grep_search: libero_config_path|config_file|get_default_path_dict|assets_default_path|dataset_default_path|init_states_default_path|bddl_files_default_path
- grep_search: libero_task_map|libero_spatial|libero_object|libero_goal|libero_10|libero_90
- grep_search: generate_bddl_from_task_info|register_task_info|save_bddl_file|outputs/
- grep_search: libero_dataset_download|snapshot_download|extractall|DATASET_LINKS|HF_REPO_ID
