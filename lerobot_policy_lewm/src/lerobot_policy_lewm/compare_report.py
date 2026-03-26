from __future__ import annotations

import json
from pathlib import Path
from typing import Any


METRICS = ("pc_success", "avg_sum_reward", "avg_max_reward", "eval_s", "eval_ep_s")


def _load_eval_info(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_overall(eval_info: dict[str, Any]) -> dict[str, Any]:
    overall = eval_info.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("Invalid eval_info.json: missing 'overall' dictionary")
    return overall


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _compute_delta_pct(a: float, b: float) -> float:
    # Delta expressed as percentage relative to baseline (policy B).
    if b == 0:
        return float("inf")
    return ((a - b) / abs(b)) * 100.0


def _winner(metric: str, a: float, b: float, label_a: str, label_b: str) -> str:
    # Higher is better for reward/success. Lower is better for timing.
    lower_is_better = metric in {"eval_s", "eval_ep_s"}
    if a == b:
        return "tie"
    if lower_is_better:
        return label_a if a < b else label_b
    return label_a if a > b else label_b


def _extract_episode_rows(eval_info: dict[str, Any], seed_start: int | None) -> list[dict[str, Any]]:
    """Flatten per-task episode stats into sortable rows.

    eval_info from lerobot-eval contains per_task[*].metrics with lists:
    sum_rewards, max_rewards, successes. Seeds are not persisted by lerobot-eval in this mode,
    so we reconstruct them from --seed and episode index when seed_start is provided.
    """
    rows: list[dict[str, Any]] = []
    per_task = eval_info.get("per_task", [])
    if not isinstance(per_task, list):
        return rows

    for task in per_task:
        task_group = task.get("task_group", "unknown")
        task_id = task.get("task_id", "unknown")
        metrics = task.get("metrics", {})
        rewards = metrics.get("sum_rewards", [])
        successes = metrics.get("successes", [])
        if not isinstance(rewards, list):
            continue
        if not isinstance(successes, list):
            successes = [None] * len(rewards)

        for idx, reward in enumerate(rewards):
            rows.append(
                {
                    "task_group": task_group,
                    "task_id": task_id,
                    "episode_index": idx,
                    "seed": (seed_start + idx) if seed_start is not None else None,
                    "sum_reward": _safe_float(reward),
                    "success": bool(successes[idx]) if idx < len(successes) else None,
                }
            )
    return rows


def _top_bottom(rows: list[dict[str, Any]], top_k: int = 3) -> dict[str, list[dict[str, Any]]]:
    if not rows:
        return {"best": [], "worst": []}

    sorted_rows = sorted(rows, key=lambda r: r["sum_reward"], reverse=True)
    return {
        "best": sorted_rows[:top_k],
        "worst": sorted_rows[-top_k:][::-1],
    }


def build_comparison_summary(
    eval_a: dict[str, Any],
    eval_b: dict[str, Any],
    label_a: str,
    label_b: str,
    seed_start: int | None,
) -> dict[str, Any]:
    overall_a = _get_overall(eval_a)
    overall_b = _get_overall(eval_b)

    comparison_metrics: dict[str, Any] = {}
    for metric in METRICS:
        a_val = _safe_float(overall_a.get(metric))
        b_val = _safe_float(overall_b.get(metric))
        comparison_metrics[metric] = {
            label_a: a_val,
            label_b: b_val,
            "delta_pct_vs_baseline_b": _compute_delta_pct(a_val, b_val),
            "winner": _winner(metric, a_val, b_val, label_a, label_b),
        }

    rows_a = _extract_episode_rows(eval_a, seed_start)
    rows_b = _extract_episode_rows(eval_b, seed_start)

    return {
        "labels": {"a": label_a, "b": label_b},
        "metrics": comparison_metrics,
        "policy_a": {
            "overall": overall_a,
            "video_paths": overall_a.get("video_paths", []),
            "top_episodes": _top_bottom(rows_a),
        },
        "policy_b": {
            "overall": overall_b,
            "video_paths": overall_b.get("video_paths", []),
            "top_episodes": _top_bottom(rows_b),
        },
        "notes": {
            "seed_reconstruction": (
                "Seeds are reconstructed as seed_start + episode_index from per-task rows. "
                "lerobot-eval does not currently persist per-episode seed in eval_info.json for this mode."
            )
        },
    }


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_markdown_report(summary: dict[str, Any], run_a_dir: Path, run_b_dir: Path) -> str:
    label_a = summary["labels"]["a"]
    label_b = summary["labels"]["b"]

    lines: list[str] = []
    lines.append("# Policy Comparison Report")
    lines.append("")
    lines.append(f"- Policy A (`{label_a}`): `{run_a_dir}`")
    lines.append(f"- Policy B (`{label_b}`): `{run_b_dir}`")
    lines.append("")
    lines.append("## Aggregated Metrics")
    lines.append("")
    lines.append("| Metric | A | B | Delta % vs B | Winner |")
    lines.append("|---|---:|---:|---:|---|")

    for metric in METRICS:
        item = summary["metrics"][metric]
        lines.append(
            f"| {metric} | {_fmt(item[label_a])} | {_fmt(item[label_b])} | "
            f"{_fmt(item['delta_pct_vs_baseline_b'])} | {item['winner']} |"
        )

    lines.append("")
    lines.append("## Top Episodes (by sum_reward)")
    lines.append("")
    for label_key, section in ((label_a, "policy_a"), (label_b, "policy_b")):
        lines.append(f"### {label_key}")
        top = summary[section]["top_episodes"]
        lines.append("- Best:")
        for row in top["best"]:
            lines.append(
                f"  - seed={row['seed']} task={row['task_group']}/{row['task_id']} "
                f"episode={row['episode_index']} reward={_fmt(row['sum_reward'])} success={row['success']}"
            )
        lines.append("- Worst:")
        for row in top["worst"]:
            lines.append(
                f"  - seed={row['seed']} task={row['task_group']}/{row['task_id']} "
                f"episode={row['episode_index']} reward={_fmt(row['sum_reward'])} success={row['success']}"
            )
        lines.append("")

    lines.append("## Videos")
    lines.append("")
    lines.append(f"### {label_a}")
    for p in summary["policy_a"]["video_paths"]:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append(f"### {label_b}")
    for p in summary["policy_b"]["video_paths"]:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- {summary['notes']['seed_reconstruction']}")

    return "\n".join(lines)


def generate_comparison_artifacts(
    *,
    eval_a_path: Path,
    eval_b_path: Path,
    label_a: str,
    label_b: str,
    seed_start: int | None,
    output_dir: Path,
    run_a_dir: Path,
    run_b_dir: Path,
) -> tuple[Path, Path]:
    eval_a = _load_eval_info(eval_a_path)
    eval_b = _load_eval_info(eval_b_path)

    summary = build_comparison_summary(eval_a, eval_b, label_a, label_b, seed_start)
    report_md = build_markdown_report(summary, run_a_dir=run_a_dir, run_b_dir=run_b_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "comparison_summary.json"
    report_path = output_dir / "comparison_report.md"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    report_path.write_text(report_md, encoding="utf-8")

    return summary_path, report_path
