import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv


SCENE_XML_BY_VARIANT = {
    "orig": "vpa/scenes/vpa_tabletop_note.xml",
    "neut": "vpa/scenes/vpa_tabletop_note_neut.xml",
    "blank": "vpa/scenes/vpa_tabletop_note_blank.xml",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a LIBERO task with VPA scene XML injection (scene-first)."
    )
    parser.add_argument("--benchmark_name", type=str, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument(
        "--scene_xml",
        type=str,
        default=None,
        help="Optional explicit scene xml path under assets root. If set with single variant, it overrides built-in mapping.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="orig",
        help="Comma-separated variants from {orig,neut,blank}.",
    )
    parser.add_argument("--out_dir", type=str, default="vpa_outputs/render_check")
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument("--img_h", type=int, default=128)
    parser.add_argument("--img_w", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _save_if_exists(obs, key, out_path):
    if key not in obs:
        return False
    image = obs[key]
    if image is None:
        return False
    cv2.imwrite(str(out_path), image[::-1, :, ::-1])
    return True


def _parse_variants(variants_str):
    variants = [x.strip().lower() for x in variants_str.split(",") if x.strip()]
    if not variants:
        raise ValueError("--variants is empty")
    invalid = [x for x in variants if x not in SCENE_XML_BY_VARIANT]
    if invalid:
        raise ValueError(
            f"Unsupported variants: {invalid}. Supported: {sorted(SCENE_XML_BY_VARIANT.keys())}"
        )
    return variants


def _resolve_scene_xml(variant, explicit_scene_xml, variant_count):
    if explicit_scene_xml is not None:
        if variant_count > 1:
            raise ValueError("--scene_xml override only supports single variant run")
        return explicit_scene_xml
    return SCENE_XML_BY_VARIANT[variant]


def _render_one_variant(args, task, bddl_file, init_states_file, task_out_dir, scene_xml):
    env_args = {
        "bddl_file_name": bddl_file,
        "scene_xml": scene_xml,
        "camera_heights": args.img_h,
        "camera_widths": args.img_w,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(args.seed)

    obs = env.reset()
    _save_if_exists(obs, "agentview_image", task_out_dir / "reset_agentview.png")
    _save_if_exists(
        obs, "robot0_eye_in_hand_image", task_out_dir / "reset_eye_in_hand.png"
    )

    if os.path.exists(init_states_file):
        init_states = torch.load(init_states_file)
        if (
            hasattr(init_states, "shape")
            and len(init_states.shape) > 1
            and init_states.shape[0] > 0
        ):
            obs = env.set_init_state(init_states[0])
            _save_if_exists(obs, "agentview_image", task_out_dir / "init_agentview.png")
            _save_if_exists(
                obs,
                "robot0_eye_in_hand_image",
                task_out_dir / "init_eye_in_hand.png",
            )

    zero_action = np.zeros(7, dtype=np.float32)
    for step in range(max(0, args.n_steps)):
        obs, _, _, _ = env.step(zero_action)
        if step in {0, max(0, args.n_steps) // 2, max(0, args.n_steps) - 1}:
            _save_if_exists(
                obs,
                "agentview_image",
                task_out_dir / f"step_{step:03d}_agentview.png",
            )
            _save_if_exists(
                obs,
                "robot0_eye_in_hand_image",
                task_out_dir / f"step_{step:03d}_eye_in_hand.png",
            )

    meta = {
        "benchmark_name": args.benchmark_name,
        "task_id": args.task_id,
        "task_name": task.name,
        "bddl_file": bddl_file,
        "scene_xml": scene_xml,
        "out_dir": str(task_out_dir),
        "n_steps": args.n_steps,
    }

    with open(task_out_dir / "meta.txt", "w", encoding="utf-8") as f:
        for key, value in meta.items():
            f.write(f"{key}: {value}\n")

    env.close()


def main():
    args = parse_args()
    variants = _parse_variants(args.variants)

    benchmark = get_benchmark(args.benchmark_name)(0)
    task = benchmark.get_task(args.task_id)

    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    init_states_file = os.path.join(
        get_libero_path("init_states"), task.problem_folder, task.init_states_file
    )

    out_dir = (
        Path(args.out_dir)
        / args.benchmark_name.lower()
        / f"task_{args.task_id:04d}_{task.name}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        scene_xml = _resolve_scene_xml(variant, args.scene_xml, len(variants))
        variant_out_dir = out_dir / f"variant_{variant}"
        variant_out_dir.mkdir(parents=True, exist_ok=True)
        _render_one_variant(
            args=args,
            task=task,
            bddl_file=bddl_file,
            init_states_file=init_states_file,
            task_out_dir=variant_out_dir,
            scene_xml=scene_xml,
        )


if __name__ == "__main__":
    main()
