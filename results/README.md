# results

Generated outputs for AstraKV-W experiments and benchmarks.

Current benchmark runs create timestamped subdirectories such as:

```text
baseline_synthetic_<timestamp>/
|-- benchmark_config.json
|-- benchmark_report.md
|-- benchmark_results.csv
|-- charts/
`-- scratch/
```

The synthetic baseline validates the measurement pipeline only. It does not
benchmark third-party runtimes, modify third-party source code, implement a
scheduler, or run CUDA kernels.
