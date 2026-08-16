# Final Benchmark Results - MOE Model Comparison

## Overview

Clean benchmark runs for 4 models on NVIDIA GB10 (DGX Spark) hardware, testing multi-turn chat performance across text, random, and reasoning datasets.

## Models Tested

| Model | Architecture | Precision | Parameters | Active Params |
|-------|-------------|-----------|------------|---------------|
| muse | Dense Transformer | bf16 | 30B | 30B |
| nemotron | Gated DeltaNet MoE | NVFP4 | 30B | 3B |
| qwen_fp8 | MoE | FP8 | 35B | 3B |
| qwen_nvfp4 | MoE | NVFP4 | 35B | 3B |

## Test Configuration

- **Concurrency Levels**: 1, 2, 4, 8
- **Output Tokens**: 128 (±32) and 512 (±128)
- **Requests per Level**: 50
- **Datasets**: text (conversational), random (random words), reasoning (logic/proofs)

## Results Directory Structure

```
/home/bala/pareto/results/final/
├── muse/
│   ├── pareto_128.html          # 128-tok Pareto report
│   └── pareto_128_rest.html     # Extended 128-tok report
├── nemotron/
│   ├── pareto_128.html          # 128-tok Pareto report
│   └── pareto_512.html          # 512-tok Pareto report
├── qwen_fp8/
│   ├── pareto_128.html          # 128-tok Pareto report
│   └── pareto_512.html          # 512-tok Pareto report
├── qwen_nvfp4/
│   ├── pareto_128.html          # 128-tok Pareto report
│   └── pareto_512.html          # 512-tok Pareto report
├── consolidated_128.html        # All models comparison (128-tok)
└── consolidated_512.html        # All models comparison (512-tok)
```

## How to View Reports

Open the HTML files in a web browser:
```bash
# Individual model reports
open /home/bala/pareto/results/final/muse/pareto_128.html
open /home/bala/pareto/results/final/nemotron/pareto_128.html
open /home/bala/pareto/results/final/qwen_fp8/pareto_128.html
open /home/bala/pareto/results/final/qwen_nvfp4/pareto_128.html

# Consolidated comparison reports
open /home/bala/pareto/results/final/consolidated_128.html
open /home/bala/pareto/results/final/consolidated_512.html
```

## Key Metrics

- **TTFT**: Time to First Token (ms)
- **ITL**: Inter-Token Latency (ms)
- **Decode Speed**: Tokens per second during decode
- **Req/s**: Request throughput
- **tok/s/W**: Energy efficiency

## Dataset Characteristics

- **Text**: Natural conversational prompts with strong token correlations
- **Random**: Random word sequences, stress-tests raw token processing
- **Reasoning**: Logic proofs and mathematical reasoning prompts

## Notes

- Router probe multiturn requires vLLM Python module (not available in host environment)
- Muse uses Dell Enterprise AI container with port 30000
- Nemotron uses Dell Enterprise AI container with port 8000
- Qwen models use generic vLLM container with port 8000
