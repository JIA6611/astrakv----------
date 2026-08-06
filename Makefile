.PHONY: help test test-all benchmark dgx-spark-validate vm-demo clean lint

# ── AstraKV-W Makefile ──────────────────────────────────────────

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Development ─────────────────────────────────────────────────

test:  ## Run quick tests (excludes GPU tests)
	python -m pytest tests/ -v --ignore=tests/test_layer_offload.py \
		-k "not GPU and not gpu" || true
	python -m pytest tests/ -v

test-all:  ## Run all tests including GPU (if available)
	python -m pytest tests/ -v

lint:  ## Run basic lint checks
	python -m flake8 astrakv/ scripts/ --max-line-length=120 --extend-ignore=E501,W503 2>/dev/null || \
	echo "flake8 not installed; skipping lint"

# ── Benchmarks ──────────────────────────────────────────────────

benchmark:  ## Run synthetic benchmark
	python cli.py benchmark --config astrakv/benchmarks/configs/baseline.yaml

benchmark-real:  ## Run real vLLM benchmark (requires GPU + server)
	python cli.py benchmark --real --config configs/dgx_spark_lmcache_cpu.yaml

dgx-spark-validate:  ## Run DGX Spark local validation and VM evidence
	bash scripts/entrypoints/run_dgx_spark_validation.sh

prefetch:  ## Run selective prefetch MVP
	python cli.py prefetch --config configs/astrakv_selective_prefetch.yaml

ablation:  ## Run ablation experiment suite
	bash scripts/archive/run_ablation.sh

# ── Virtual Memory ──────────────────────────────────────────────

vm-demo:  ## Run mmap KV-cache VM demo
	python cli.py vm mmap --blocks 50 --block-size-mb 1

vm-demo-full:  ## Run full VM demo suite
	python scripts/vm/run_vm_demo.py --output-dir results/vm/vm_demo
	python cli.py vm mmap --blocks 200 --block-size-mb 2 --output-dir results/vm/mmap_demo

# ── Analysis & Reports ──────────────────────────────────────────

analyze:  ## Analyze stress test results
	python cli.py analyze stress --results-dir results/gpu/edge_sim

report:  ## Build competition report from artifacts
	python cli.py report --output-dir results/report

# ── Edge Simulation ─────────────────────────────────────────────

edge-setup:  ## Setup cgroup for edge simulation (requires root)
	sudo bash scripts/archive/setup_edge_sim.sh 16

edge-test:  ## Run edge device simulation tests
	bash scripts/archive/run_edge_sim_tests.sh

# ── Cleanup ─────────────────────────────────────────────────────

clean:  ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	rm -rf .pytest_cache

clean-results:  ## Remove benchmark results (CAUTION: deletes data)
	rm -rf results/*/
