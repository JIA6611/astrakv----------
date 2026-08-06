# Text/KV Consistency Workload

- Context length: `131072`
- Block size tokens: `16`

## Analysis Buckets

| bucket | target prefix ratio | reusable blocks | mutation start block | analysis request | warmup anchor |
| --- | ---: | ---: | ---: | --- | --- |
| exact | 1.00 | 8192 | 8192 | exact-probe | exact-anchor |
| sim90 | 0.90 | 7373 | 7373 | sim90-probe | sim90-anchor |
| sim80 | 0.80 | 6554 | 6554 | sim80-probe | sim80-anchor |

## Warmup Conditions

- `cold`: no warmup requests before the analysis workload.
- `warm`: one warmup pass using the anchor prompts before the analysis workload.
- `hot`: two warmup passes using the anchor prompts before the analysis workload.

