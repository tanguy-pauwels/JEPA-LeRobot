from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .compare_report import generate_comparison_artifacts


@dataclass
class EvalRunSpec:
    label: str
    policy_path: str
    run_dir: Path


def _build_eval_command(
    *,
    policy_path: str,
    output_dir: Path,
    env_type: str,
    env_task: str | None,
    n_episodes: int,
    batch_size: int,
    seed: int,
    device: str,
) -> list[str]:
    cmd = [
        "lerobot-eval",
        f"--policy.path={policy_path}",
        f"--env.type={env_type}",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.batch_size={batch_size}",
        f"--seed={seed}",
        f"--policy.device={device}",
        f"--output_dir={output_dir}",
    ]
    if env_task:
        cmd.append(f"--env.task={env_task}")
    return cmd


def _run_eval(spec: EvalRunSpec, cmd: list[str], env: dict[str, str]) -> Path:
    print(f"\\n[compare] Running eval for '{spec.label}'")
    print("[compare] Command:", " ".join(cmd))

    completed = subprocess.run(cmd, env=env, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Evaluation failed for '{spec.label}' with exit code {completed.returncode}."
        )

    eval_info = spec.run_dir / "eval_info.json"
    videos_dir = spec.run_dir / "videos"
    if not eval_info.exists():
        raise FileNotFoundError(
            f"Expected eval artifact missing for '{spec.label}': {eval_info}"
        )
    if not videos_dir.exists():
        raise FileNotFoundError(
            f"Expected videos directory missing for '{spec.label}': {videos_dir}"
        )
    return eval_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two policies with a shared lerobot-eval protocol and generate a report."
    )
    parser.add_argument("--policy-a-path", required=True)
    parser.add_argument("--policy-b-path", required=True)
    parser.add_argument("--label-a", default="lewm")
    parser.add_argument("--label-b", default="baseline")
    parser.add_argument("--output-root", default="comparison_runs")

    parser.add_argument("--env-type", default="aloha")
    parser.add_argument("--env-task", default="AlohaInsertion-v0")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])

    parser.add_argument(
        "--lerobot-src",
        default=None,
        help="Optional path to local lerobot/src to prepend to PYTHONPATH for subprocesses.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_root) / ts
    run_a = root / f"run_{args.label_a}"
    run_b = root / f"run_{args.label_b}"
    compare_dir = root / "comparison"

    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.lerobot_src:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{args.lerobot_src}:{existing}" if existing else args.lerobot_src

    spec_a = EvalRunSpec(label=args.label_a, policy_path=args.policy_a_path, run_dir=run_a)
    spec_b = EvalRunSpec(label=args.label_b, policy_path=args.policy_b_path, run_dir=run_b)

    cmd_a = _build_eval_command(
        policy_path=spec_a.policy_path,
        output_dir=spec_a.run_dir,
        env_type=args.env_type,
        env_task=args.env_task,
        n_episodes=args.n_episodes,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    cmd_b = _build_eval_command(
        policy_path=spec_b.policy_path,
        output_dir=spec_b.run_dir,
        env_type=args.env_type,
        env_task=args.env_task,
        n_episodes=args.n_episodes,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )

    eval_a = _run_eval(spec_a, cmd_a, env=env)
    eval_b = _run_eval(spec_b, cmd_b, env=env)

    summary_path, report_path = generate_comparison_artifacts(
        eval_a_path=eval_a,
        eval_b_path=eval_b,
        label_a=args.label_a,
        label_b=args.label_b,
        seed_start=args.seed,
        output_dir=compare_dir,
        run_a_dir=run_a,
        run_b_dir=run_b,
    )

    print("\\n[compare] Done.")
    print(f"[compare] Summary JSON: {summary_path}")
    print(f"[compare] Markdown report: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[compare] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
