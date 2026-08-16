# Nemotron 3.5 Lightning: Experiment Ideas for 10K-User Scale

## What We Already Have

| Dataset | What | Status |
|---------|------|--------|
| Routing entropy/Jaccard/cosine | Content-aware routing proof | Done |
| Concurrency c1-c128, 128 tok output | Throughput scaling curve | Done |
| Concurrency c1-c8, 512 tok output | Longer decode scaling | Done |
| 3 workload types | Text/random/reasoning comparison | Done |
| GPU telemetry | Power/efficiency at each level | Done |
| Cross-model comparisons | Nemotron vs Qwen vs Muse | Done |

### Existing Throughput Data (128-tok output, text dataset)

| Concurrency | Throughput (req/s) | Output tok/s | Power (W) | Efficiency (tok/s/W) |
|-------------|-------------------|--------------|-----------|---------------------|
| c1 | 0.75 | 98 | 31 | 3.1 |
| c2 | 1.11 | 144 | 34 | 4.2 |
| c4 | 1.49 | 193 | 35 | 5.5 |
| c8 | 2.11 | 275 | 37 | 7.4 |
| c16 | 3.07 | 411 | 38 | 10.8 |
| c32 | 4.58 | 576 | 40 | 14.4 |
| c64 | 5.33 | 713 | 39 | 18.2 |
| c128 | 5.22 | 698 | 39 | 18.0 |

### Existing Throughput Data (512-tok output, text dataset)

| Concurrency | Throughput (req/s) | Output tok/s | Power (W) | Efficiency (tok/s/W) |
|-------------|-------------------|--------------|-----------|---------------------|
| c1 | 0.18 | 94 | 40 | 2.4 |
| c2 | 0.27 | 139 | 43 | 3.3 |
| c4 | 0.38 | 193 | 45 | 4.3 |
| c8 | 0.54 | 275 | 46 | 5.9 |

---

## Key Gaps for 10K-User / 250K-Context Design

1. **No long-context data** — we only tested 128/512 output tokens, never measured input > ~4K tokens. 250K context is uncharted.
2. **No multi-GPU data** — everything is single GPU (GB10). Zero TP/EP scaling curves.
3. **No Mamba-specific profiling** — we know Mamba layers exist but never measured their behavior vs MoE under load.
4. **No prefix caching effectiveness data** — critical for agentic workflows where system prompts repeat.
5. **No latency distribution at scale** — we have throughput but no p95/p99 latency curves.

---

## Experiment Ideas

### Group A: Long-Context Memory & Throughput (CRITICAL for 250K)

These answer: *"Can we even serve 250K context, and at what cost?"*

| # | Experiment | What It Proves | How to Run |
|---|-----------|----------------|------------|
| A1 | **KV Cache Memory Scaling** — Send prompts of 1K, 4K, 16K, 64K, 128K, 256K tokens. Measure GPU memory utilization, available batch slots, TTFT. | How memory grows with context length. Whether 256K fits on available hardware. | `router_probe.py` modified to accept prompt length, or aiperf with padded prompts |
| A2 | **Mamba vs MoE Layer Latency by Sequence Length** — Instrument model to measure per-layer timing at 1K, 16K, 64K, 256K. | Whether Mamba layers actually stay constant-time. Where the MoE bottleneck is. | Custom vLLM profiling hook or `torch.cuda.Event` timing |
| A3 | **Chunked Prefill at 256K** — Test vLLM's chunked prefill with 256K input. Measure TTFT breakdown (prefill vs first decode). | Whether chunked prefill makes 256K inputs feasible. | aiperf with long-input dataset, `--enable-chunked-prefill` |
| A4 | **Prefix Caching Hit Rate** — Same system prompt + different user queries. Measure cache hit rate, memory saved, TTFT reduction. | Quantifies prefix caching value for agentic workflows. | vLLM metrics endpoint + custom dataset with shared prefixes |

### Group B: Concurrency & Throughput at Scale (CRITICAL for 10K users)

These answer: *"How many concurrent users can one GPU handle, and where does it break?"*

| # | Experiment | What It Proves | How to Run |
|---|-----------|----------------|------------|
| B1 | **Extended Concurrency Sweep to c256/c512** — Push beyond c128 to find the throughput cliff. | Maximum concurrent sessions per GPU. Where memory pressure causes OOM or thrashing. | `run_sweep.sh --concurrency 256` and `--concurrency 512` |
| B2 | **Long-Output Concurrency** — c1-c64 with 1024, 2048, 4096 output tokens. | How longer generations affect concurrency limits (KV cache accumulation). | `run_sweep.sh --output-tokens 2048` |
| B3 | **Mixed-Concurrency Realistic Load** — Poisson-distributed arrivals, variable output lengths (128-2048), concurrent sessions c32-c128. | Real-world throughput, not just uniform synthetic load. | Custom dataset with variable-length outputs |
| B4 | **Queue Depth & Rejection Rate** — Measure request queuing, timeout rates, and error rates at high concurrency. | System reliability under load. | vLLM server metrics + custom monitoring |

### Group C: Agentic Workflow Profiling (CRITICAL for tool-use agents)

These answer: *"How does routing and throughput change in real agent loops?"*

| # | Experiment | What It Proves | How to Run |
|---|-----------|----------------|------------|
| C1 | **Routing Drift Across Agent Turns** — Multi-turn probe with 10+ turns. Track how entropy/Jaccard changes turn-over-turn. | Whether the router stabilizes or drifts as context accumulates. | Extend `router_probe_multiturn.py` to 10+ turns |
| C2 | **Tool-Call Routing Signatures** — Inject tool-call format tokens (JSON, function definitions). Compare routing vs plain text. | Whether tool-call-heavy workflows stress different experts. | Custom dataset with function-calling format |
| C3 | **System Prompt Reuse** — Same 4K system prompt, 100 different user queries. Measure prefix cache hit rate + routing consistency. | Whether cached prefix tokens skip routing recomputation. | Custom dataset + vLLM metrics |
| C4 | **Interleaved Read/Write Patterns** — Simulate agent: generate tool call -> parse result -> generate next response. Measure throughput per phase. | Latency breakdown for real agent loops. | Custom dataset simulating tool-call-response cycles |

### Group D: Mamba-Specific Experiments (UNIQUE to Nemotron)

These answer: *"What does the hybrid architecture buy us that pure MoE doesn't?"*

| # | Experiment | What It Proves | How to Run |
|---|-----------|----------------|------------|
| D1 | **Mamba Layer Throughput Scaling** — Compare throughput at 1K vs 256K input length on Nemotron vs Qwen (pure MoE). | Quantifies Mamba's constant-time advantage for long sequences. | Side-by-side aiperf with both models |
| D2 | **Routing Entropy at Long Context** — Run routing probe with 64K, 128K, 256K input prompts. | Whether routing becomes more/less diverse at extreme context lengths. | Extended `router_probe.py` with long prompts |
| D3 | **Mamba Cache Mode Comparison** — Test `MAMBA_CACHE_MODE=align` vs alternative modes. Measure memory and throughput. | Whether cache alignment matters for performance at scale. | Different launch configs |
| D4 | **Speculative Decoding Impact** — DSpark with 3 tokens vs no speculation vs different speculation depths. | Whether speculation helps or hurts at high concurrency. | Modify SPECULATIVE_CONFIG in launch script |

### Group E: Infrastructure Sizing (for 10K-User Deployment)

These answer: *"How many GPUs/nodes do we need?"*

| # | Experiment | What It Proves | How to Run |
|---|-----------|----------------|------------|
| E1 | **Memory Budget Breakdown** — At c1, c8, c32, c64: measure model weights, KV cache, activation memory, free memory. | Exact memory allocation per concurrent session. Enables capacity planning. | vLLM metrics + `nvidia-smi` monitoring |
| E2 | **Throughput-per-GPU Curve** — Fit a curve to our c1-c128 data. Extrapolate to c256, c512. | Predict how many GPUs needed for N concurrent users. | Statistical analysis of existing data |
| E3 | **Batch Size vs Latency Pareto** — For each concurrency, plot latency p50/p95/p99 vs throughput. | SLO compliance: "Can we serve 10K users with <2s p99 latency?" | Existing data + extended concurrency |
| E4 | **Power Efficiency at Scale** — tok/s/W at each concurrency level. | Total cost of ownership for N-GPU deployment. | Existing GPU telemetry data |

---

## Recommended Priority Order

| Priority | Experiment | Why First |
|----------|-----------|-----------|
| **P0** | A1 (KV Cache Scaling) | Blocks everything — if 256K doesn't fit, the architecture changes |
| **P0** | E1 (Memory Budget) | Pairs with A1 — tells us exactly how many concurrent 256K sessions fit |
| **P1** | B1 (Extended Concurrency) | Determines single-GPU capacity ceiling |
| **P1** | A4 (Prefix Caching) | Critical for agentic workflows — could 2-3x effective capacity |
| **P1** | D1 (Mamba vs MoE Long Context) | Unique value prop — validates the hybrid architecture choice |
| **P2** | B3 (Mixed-Con realistic load) | Most realistic workload model |
| **P2** | C1 (Routing Drift) | Understanding agent-specific behavior |
| **P2** | E2 (Throughput Curve) | Enables capacity planning math |
| **P3** | D4 (Speculative Decoding) | Optimization tuning |
| **P3** | C3 (System Prompt Reuse) | Agentic workflow optimization |

---

## Quick Math: How Many GPUs for 10K Users?

From our data (128-tok output, text):
- **c1**: 0.75 req/s, 98 tok/s, 31W
- **c128**: 5.22 req/s, 698 tok/s, 39W
- **Per-session decode**: ~98 tok/s at c1, saturates around 700 tok/s at c64+

Assumptions for 10K concurrent users:
- Average 100 tokens/s output per user
- Average 256 tokens output per request
- 1 request/minute per user = 167 req/s needed
- Each session holds ~256K context KV cache

If we can fit N concurrent 256K sessions per GPU:
- Memory per session at 256K = ?? (this is what A1 tells us)
- If 1 session = 4GB KV cache -> ~30 sessions/GPU -> need ~334 GPUs
- If prefix caching saves 4GB/system-prompt -> maybe 40 sessions/GPU -> ~250 GPUs
- If Mamba saves 50% vs pure MoE -> maybe 60 sessions/GPU -> ~167 GPUs

**The A1 experiment is the single most important number.**

---

## Hardware Context

- **Current hardware**: Dell Pro Max, NVIDIA GB10 (DGX Spark, SM121), 128 GB unified memory, single GPU
- **Memory bandwidth**: ~273 GB/s LPDDR5X
- **TDP**: 140W
- **No NVLink/NVSwitch** — single GPU only
- **Model**: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 (30B total, 3B active, NVFP4)
- **Architecture**: 52 layers (29 Mamba + 23 MoE), 256 experts, top-6 routing
- **Launch**: Dell Enterprise AI container, Marlin MoE backend, FlashInfer Mamba backend, DSpark speculative decoding (3 tokens)
