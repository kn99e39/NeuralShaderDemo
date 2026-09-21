"""Extract RNA TensorBoard scalars into a compact JSON summary and curve PNG."""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-json", required=True, type=pathlib.Path)
    parser.add_argument("--output-plot", required=True, type=pathlib.Path)
    args = parser.parse_args()

    accumulator = EventAccumulator(str(args.log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    series = {}
    for tag in tags:
        events = accumulator.Scalars(tag)
        higher_is_better = "loss" not in tag.lower()
        best = max if higher_is_better else min
        series[tag] = {
            "count": len(events),
            "final_step": int(events[-1].step) if events else None,
            "final_value": float(events[-1].value) if events else None,
            "best_value": float(best(event.value for event in events)) if events else None,
            "best_direction": "max" if higher_is_better else "min",
        }

    summary = {"log_dir": str(args.log_dir), "scalar_tags": tags, "series": series}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_tags = [tag for tag in ("val_psnr", "val_loss", "psnr", "loss") if tag in tags]
    figure, axes = plt.subplots(len(plot_tags), 1, figsize=(9, max(3, 2.5 * len(plot_tags))))
    if len(plot_tags) == 1:
        axes = [axes]
    for axis, tag in zip(axes, plot_tags):
        events = accumulator.Scalars(tag)
        axis.plot([event.step for event in events], [event.value for event in events], linewidth=1.2)
        axis.set_title(tag)
        axis.set_xlabel("global step")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_plot, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
