"""Materialize the load-vs-recompute decision-correctness probe workload.

One deterministic exact prefix (synthesized from an audited grouped prompt)
is written by a seed row and then revisited under five controlled scenarios.
Every probe row carries a ``kv_core_decision_probe`` override plus the
expected action/reason so the runtime decision can be verified against a
locked label.  Expected labels are calibrated against
``choose_load_vs_recompute`` (see tests/test_decision_probe.py) with the
same cost numbers; changing the scenario values without re-locking the unit
tests invalidates the probe contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrakv.benchmarks.runtime_workload import RuntimeWorkloadRow  # noqa: E402
from scripts.benchmark.materialize_kv_equivalence_workload import select_prompt_source  # noqa: E402


WORKLOAD_NAME = "decision_probe_single_prefix"
REPS_PER_SCENARIO = 3

# Locked scenario matrix.  Values are chosen so the expected reason is robust
# to object-size uncertainty: S1 reads at 20 GB/s, S4 at 0.1 GB/s, S2 uses a
# 300 ms deadline while SSD reads take ~1 s at the default 3 GB/s, S3 forces
# the memory-pressure gate, and S5 uses the test-only force-recompute path.
SCENARIOS: dict[str, dict[str, Any]] = {
    "S1": {
        "probe": {"ssd_gbps": 20.0, "memory_pressure": 0.0, "deadline_ns": 60_000_000_000},
        "expected_decision": "load",
        "expected_reason": "native_load_cheaper",
    },
    "S2": {
        "probe": {"deadline_ns": 300_000_000, "memory_pressure": 0.0},
        "expected_decision": "recompute",
        "expected_reason": "load_deadline_miss",
    },
    "S3": {
        "probe": {"memory_pressure": 0.95, "ssd_gbps": 20.0, "deadline_ns": 60_000_000_000},
        "expected_decision": "recompute",
        "expected_reason": "uma_memory_pressure",
    },
    "S4": {
        "probe": {"ssd_gbps": 0.1, "memory_pressure": 0.0, "deadline_ns": 60_000_000_000},
        "expected_decision": "recompute",
        "expected_reason": "recompute_cheaper",
    },
    "S5": {
        "probe": {"force_recompute": True},
        "expected_decision": "recompute",
        "expected_reason": "equivalence_probe_force_recompute",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--reps", type=int, default=REPS_PER_SCENARIO)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_tokens < 1 or args.reps < 1:
        raise SystemExit("--output-tokens and --reps must be positive")
    source = select_prompt_source(Path(args.prompts_file), "")
    rows = materialize(source, output_tokens=args.output_tokens, reps=args.reps)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workload_path = output_dir / f"{WORKLOAD_NAME}.jsonl"
    workload_path.write_text(
        "".join(json.dumps(row.to_record(), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema": "astrakv-decision-probe-workload-v1",
        "workload": WORKLOAD_NAME,
        "prompts_file": str(Path(args.prompts_file)),
        "prompts_sha256": _sha256_file(Path(args.prompts_file)),
        "source_request_id": source.request_id,
        "source_prompt_sha256": hashlib.sha256(source.prompt.encode()).hexdigest(),
        "workload_sha256": _sha256_file(workload_path),
        "reps_per_scenario": args.reps,
        "scenarios": {
            name: {
                "expected_decision": spec["expected_decision"],
                "expected_reason": spec["expected_reason"],
                "probe": spec["probe"],
            }
            for name, spec in SCENARIOS.items()
        },
        "test_only_force_recompute": True,
    }
    (output_dir / "decision_probe_workload.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


def materialize(source: RuntimeWorkloadRow, *, output_tokens: int, reps: int = REPS_PER_SCENARIO) -> list[RuntimeWorkloadRow]:
    digest = hashlib.sha256(source.request_id.encode()).hexdigest()[:12]
    rows: list[RuntimeWorkloadRow] = []
    index = 0
    rows.append(_row(
        source=source,
        request_id=f"decision-probe-{digest}-seed",
        case="decision_probe_seed",
        metadata_extra={"generation_seed": 0},
        output_tokens=output_tokens,
        arrival_index=index,
    ))
    index += 1
    for scenario, spec in SCENARIOS.items():
        for rep in range(1, reps + 1):
            rows.append(_row(
                source=source,
                request_id=f"decision-probe-{digest}-{scenario}-{rep:02d}",
                case=f"decision_probe_{scenario}",
                metadata_extra={
                    "generation_seed": 0,
                    "kv_core_decision_probe": dict(spec["probe"]),
                    "expected_decision": spec["expected_decision"],
                    "expected_reason": spec["expected_reason"],
                },
                output_tokens=output_tokens,
                arrival_index=index,
            ))
            index += 1
    return rows


def _row(
    *,
    source: RuntimeWorkloadRow,
    request_id: str,
    case: str,
    metadata_extra: dict[str, Any],
    output_tokens: int,
    arrival_index: int,
) -> RuntimeWorkloadRow:
    metadata = dict(source.metadata)
    metadata.update({
        "workload_type": WORKLOAD_NAME,
        "scenario": "load_vs_recompute_decision_probe",
        "sample_id": request_id,
        "source_request_id": source.request_id,
        "exact_prefix": True,
        **metadata_extra,
    })
    is_seed = case == "decision_probe_seed"
    return RuntimeWorkloadRow(
        request_id=request_id,
        prompt=source.prompt,
        prefix_id=source.prefix_id,
        prefix_hash=source.prefix_hash,
        cache_key=source.cache_key,
        arrival_index=arrival_index,
        reuse_ratio=0.0 if is_seed else 1.0,
        reuse_bucket="none" if is_seed else "high",
        context_length=source.context_length,
        expected_output_tokens=output_tokens,
        batch_size=1,
        sleep_before_s=1.0,
        prefetch_lead_s=0.0,
        case=case,
        metadata=metadata,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
