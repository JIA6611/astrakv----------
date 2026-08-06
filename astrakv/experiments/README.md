# experiments

Standalone experiment helpers for AstraKV-W.

These experiments are separate from the real vLLM/LMCache benchmark path. They
are intended to provide reproducible evidence for contest explanations,
ablation notes, and OS-style demonstrations without modifying third-party
runtimes.

## Files

- `vm_demo.py`: compatibility wrapper for the file-backed `mmap` VM demo. The
  reusable VM backend implementation now lives in `runtime/vm_backend.py`.

## Boundaries

- No vLLM or LMCache imports.
- No CUDA kernels.
- No model execution.
- No benchmark runner changes.
- Reusable backend logic should live in `runtime/`; experiments keep runner and
  artifact-facing code thin.

Run P2-4 with:

```powershell
python scripts\run_vm_demo.py --output-dir results\p2_4_vm_demo
```

