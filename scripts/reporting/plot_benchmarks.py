"""Plot AstraKV-W benchmark CSV outputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_csv(csv_path: str | Path, output_dir: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    rows = read_rows(csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return []

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = f"{row.get('backend', 'unknown')} / batch={row['batch_size']}"
        grouped[label].append(row)

    metrics = [
        ("ttft_ms", "TTFT (ms)", "ttft_by_context.png"),
        ("ttft_change_pct", "TTFT change (%)", "ttft_change_by_context.png"),
        ("tpot_ms", "TPOT (ms/token)", "tpot_by_context.png"),
        ("tpot_change_pct", "TPOT change (%)", "tpot_change_by_context.png"),
        ("throughput_tokens_s", "Throughput (tokens/s)", "throughput_by_context.png"),
        ("cpu_memory_peak_mb", "CPU RSS peak (MB)", "cpu_memory_by_context.png"),
        ("gpu_memory_peak_mb", "GPU memory used (MB)", "gpu_memory_by_context.png"),
        ("gpu_kv_peak_mb", "Estimated GPU KV peak (MB)", "gpu_kv_peak_by_context.png"),
        ("gpu_memory_reduction_pct", "GPU KV memory reduction (%)", "gpu_memory_reduction_by_context.png"),
        ("ssd_read_mb_s", "SSD read (MB/s)", "ssd_read_by_context.png"),
        ("ssd_write_mb_s", "SSD write (MB/s)", "ssd_write_by_context.png"),
        ("kv_cache_hit_rate", "KV cache hit rate", "kv_hit_rate_by_context.png"),
        ("prefetch_hit_rate", "Prefetch hit rate", "prefetch_hit_rate_by_context.png"),
        ("prefetch_waste_rate", "Prefetch waste rate", "prefetch_waste_rate_by_context.png"),
    ]

    outputs: list[Path] = []
    for metric, ylabel, filename in metrics:
        if not any(_has_value(row.get(metric)) for row in rows):
            continue
        fig, ax = plt.subplots(figsize=(8, 4.8))
        plotted = False
        for label, batch_rows in sorted(grouped.items(), key=lambda item: item[0]):
            sorted_rows = sorted(batch_rows, key=lambda row: int(row["context_length"]))
            x = [int(row["context_length"]) for row in sorted_rows]
            y = [_as_float(row.get(metric)) for row in sorted_rows]
            ax.plot(x, y, marker="o", label=label)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Context length")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        output_path = out_dir / filename
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        outputs.append(output_path)
    return outputs


def _has_value(value: Any) -> bool:
    return value not in (None, "", "None", "nan")


def _as_float(value: Any) -> float:
    if value in (None, "", "None", "nan"):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AstraKV-W benchmark CSV results.")
    parser.add_argument("--csv", required=True, help="Input benchmark CSV path.")
    parser.add_argument("--output-dir", required=True, help="Directory for PNG chart outputs.")
    args = parser.parse_args()
    outputs = plot_csv(args.csv, args.output_dir)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
