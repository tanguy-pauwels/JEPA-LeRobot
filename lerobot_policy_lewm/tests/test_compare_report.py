from pathlib import Path

from lerobot_policy_lewm.compare_report import build_comparison_summary, build_markdown_report


def _fake_eval(success: float, sum_reward: float, eval_s: float):
    return {
        "overall": {
            "pc_success": success,
            "avg_sum_reward": sum_reward,
            "avg_max_reward": sum_reward / 2,
            "eval_s": eval_s,
            "eval_ep_s": eval_s / 20,
            "video_paths": ["/tmp/video0.mp4"],
        },
        "per_task": [
            {
                "task_group": "aloha",
                "task_id": 0,
                "metrics": {
                    "sum_rewards": [0.1, 0.7, 0.2],
                    "successes": [False, True, False],
                    "video_paths": ["/tmp/video0.mp4"],
                },
            }
        ],
    }


def test_build_comparison_summary_and_report():
    eval_a = _fake_eval(success=50.0, sum_reward=1.0, eval_s=10.0)
    eval_b = _fake_eval(success=25.0, sum_reward=0.5, eval_s=12.0)

    summary = build_comparison_summary(
        eval_a=eval_a,
        eval_b=eval_b,
        label_a="lewm",
        label_b="act",
        seed_start=1000,
    )

    assert summary["metrics"]["pc_success"]["winner"] == "lewm"
    assert summary["metrics"]["eval_s"]["winner"] == "lewm"
    assert summary["policy_a"]["top_episodes"]["best"][0]["seed"] == 1001

    report = build_markdown_report(
        summary,
        run_a_dir=Path("/tmp/run_a"),
        run_b_dir=Path("/tmp/run_b"),
    )
    assert "Policy Comparison Report" in report
    assert "Aggregated Metrics" in report
    assert "Videos" in report
