# License and Submission Notice

## License Scope

- Source code in this repository is released under the MIT License. See `LICENSE`.
- Technical documentation, project reports, presentation materials, and demo-video materials are released under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0), unless a specific file states otherwise.

## Third-Party Boundary

AstraKV-W does not modify vLLM, LMCache, SGLang, TensorRT-LLM, FlashAttention, CUDA kernels, or model checkpoint source files as team-authored code. These systems are used as external runtime backends, reference systems, or local model artifacts. The project contribution is the surrounding control plane, evidence pipeline, policy analysis, benchmark/report tooling, VM proof-of-concept modules, and reproducible experiment organization.

The base-version and incremental-contribution statement is maintained in:

- `docs/guides/base_version_and_contribution_cn.md`

## AI Tool Disclosure

The team used OpenAI Codex / GPT-5 as an auxiliary development and documentation tool. AI assistance was used for repository reading, debugging guidance, experiment-result analysis, report drafting, figure/report organization, and compliance-review preparation. AI was not used to fabricate benchmark results or replace DGX Spark experiment execution.

AI usage records and interaction summaries are maintained in:

- `docs/ai_usage/README.md`

## Submission Note

Large generated experiment artifacts under `results/` and draft materials under `reports/` may be excluded from Git by default to avoid committing multi-GB outputs. When submitting, include either the required final artifacts through the competition platform or explicitly provide links/archives for the final report, presentation, demo video, selected evidence summaries, and reproduction instructions.
