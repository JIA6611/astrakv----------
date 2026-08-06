# Clone Validation Record

Validation date: 2026-05-25

Scope:

- Official GitHub repositories under `third_party/`.
- Read-only validation only.
- No third-party source files were modified.
- No benchmark, runtime, or CUDA kernel was executed or implemented.

## Network Recovery

Direct HTTPS access to `github.com:443` failed initially. The local Windows system proxy was configured as `127.0.0.1:7890`, and that proxy endpoint was reachable. GitHub access was restored for Git commands by passing the proxy to Git commands:

```powershell
git -c http.proxy=http://127.0.0.1:7890 -c https.proxy=http://127.0.0.1:7890 ls-remote https://github.com/vllm-project/vllm.git HEAD
```

## Clone Results

| Project | Official repository | Local directory | Clone result | HEAD |
| --- | --- | --- | --- | --- |
| vLLM | `https://github.com/vllm-project/vllm.git` | `third_party/vllm/` | Success | `d400445` |
| LMCache | `https://github.com/LMCache/LMCache.git` | `third_party/LMCache/` | Success | `9793d8b` |
| FlashAttention | `https://github.com/Dao-AILab/flash-attention.git` | `third_party/flash-attention/` | Success | `2d5d5a1` |
| llama.cpp | `https://github.com/ggml-org/llama.cpp.git` | `third_party/llama.cpp/` | Success | `328874d` |
| SGLang | `https://github.com/sgl-project/sglang.git` | `third_party/sglang/` | Success | `0801cc0` |
| TensorRT-LLM | `https://github.com/NVIDIA/TensorRT-LLM.git` | `third_party/TensorRT-LLM/` | Success | `546a5b0` |

## Clean Worktree Check

`git status --short` was executed in every cloned repository.

| Project | Status |
| --- | --- |
| vLLM | Clean |
| LMCache | Clean |
| FlashAttention | Clean |
| llama.cpp | Clean |
| SGLang | Clean |
| TensorRT-LLM | Clean |

## Notes

- TensorRT-LLM contains very long file paths. On Windows, the first checkout failed with `Filename too long`. The incomplete checkout was removed and the repository was recloned with `core.longpaths=true`.
- The `core.longpaths=true` setting is a local Git checkout configuration and does not modify upstream source files.
- All subsequent repository worktrees were clean.

