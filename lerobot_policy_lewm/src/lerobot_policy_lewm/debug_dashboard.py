from __future__ import annotations

import argparse
import json
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px


def _load_artifacts(artifacts_dir: Path):
    summary = json.loads((artifacts_dir / "offline_summary.json").read_text(encoding="utf-8"))
    per_dim = pd.read_csv(artifacts_dir / "per_dim_metrics.csv")
    per_episode = pd.read_csv(artifacts_dir / "per_episode_metrics.csv")
    samples = pd.read_csv(artifacts_dir / "samples_predictions.csv")
    return summary, per_dim, per_episode, samples


def _summary_markdown(summary: dict) -> str:
    keys = [
        "policy_path",
        "dataset_repo_id",
        "device",
        "num_samples",
        "action_dim",
        "global_mse",
        "global_mae",
        "avg_l2_per_sample",
        "target_mean",
        "target_std",
        "pred_mean",
        "pred_std",
    ]
    lines = ["## Offline Summary", ""]
    for k in keys:
        if k in summary:
            lines.append(f"- **{k}**: {summary[k]}")
    return "\n".join(lines)


def _per_dim_bar(per_dim: pd.DataFrame):
    return px.bar(
        per_dim,
        x="dim",
        y="mae",
        title="MAE per Action Dimension",
        labels={"dim": "Action Dimension", "mae": "MAE"},
    )


def _scatter_for_dim(samples: pd.DataFrame, dim: int):
    sub = samples[samples["dim"] == dim]
    return px.scatter(
        sub,
        x="target",
        y="pred",
        color="episode_index",
        title=f"Prediction vs Target (dim={dim})",
        opacity=0.6,
    )


def _episode_timeseries(samples: pd.DataFrame, episode_index: int, dim: int):
    sub = samples[(samples["episode_index"] == episode_index) & (samples["dim"] == dim)].copy()
    sub["sample_order"] = range(len(sub))
    fig = px.line(
        sub,
        x="sample_order",
        y=["target", "pred"],
        title=f"Episode {episode_index} - Dimension {dim}",
        labels={"sample_order": "Step", "value": "Action Value", "variable": "Series"},
    )
    return fig


def _episode_table(per_episode: pd.DataFrame):
    return per_episode.sort_values("mse", ascending=True)


def build_app(artifacts_dir: Path) -> gr.Blocks:
    summary, per_dim, per_episode, samples = _load_artifacts(artifacts_dir)

    dims = sorted(per_dim["dim"].astype(int).unique().tolist())
    episodes = sorted(per_episode["episode_index"].astype(int).unique().tolist())

    with gr.Blocks(title="LeWM Debug Dashboard") as demo:
        gr.Markdown("# LeWM Offline Debug Dashboard")
        gr.Markdown(_summary_markdown(summary))

        with gr.Row():
            gr_plot_bar = gr.Plot(value=_per_dim_bar(per_dim), label="Error by Dimension")
            gr_episode_table = gr.Dataframe(value=_episode_table(per_episode), label="Episode Ranking")

        gr.Markdown("## Detailed Inspection")
        with gr.Row():
            dim_input = gr.Dropdown(choices=dims, value=dims[0], label="Action Dimension")
            episode_input = gr.Dropdown(choices=episodes, value=episodes[0], label="Episode Index")

        scatter_plot = gr.Plot(label="Pred vs Target Scatter")
        episode_plot = gr.Plot(label="Episode Trajectory")

        def _refresh(dim: int, episode: int):
            return _scatter_for_dim(samples, dim), _episode_timeseries(samples, episode, dim)

        dim_input.change(_refresh, inputs=[dim_input, episode_input], outputs=[scatter_plot, episode_plot])
        episode_input.change(_refresh, inputs=[dim_input, episode_input], outputs=[scatter_plot, episode_plot])

        scatter_plot.value = _scatter_for_dim(samples, dims[0])
        episode_plot.value = _episode_timeseries(samples, episodes[0], dims[0])

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch local Gradio dashboard for LeWM offline artifacts")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    app = build_app(artifacts_dir)
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
