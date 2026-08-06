# Models

This directory is used for local model artifacts.

For reproduction, download the model that matches the run you want to perform:

- Dense mainline runs: `Qwen/Qwen2.5-7B-Instruct`
- MoE route-trace analysis: `Qwen/Qwen1.5-MoE-A2.7B-Chat`

Suggested layout:

```text
models/
  Qwen2.5-7B-Instruct/
  Qwen1.5-MoE-A2.7B-Chat/
```

Typical download commands:

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
huggingface-cli download Qwen/Qwen1.5-MoE-A2.7B-Chat --local-dir models/Qwen1.5-MoE-A2.7B-Chat
```

Then point `ASTRAKV_MODEL` to the local directory you need for the current run.
