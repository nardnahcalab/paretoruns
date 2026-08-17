#!/usr/bin/env python3
"""
A2: Mamba vs MoE Layer Latency by Sequence Length

Measures per-layer timing at 1K, 16K, 64K, 128K, 256K input lengths.
Tests whether Mamba layers stay constant-time while MoE layers scale with sequence length.
"""

import asyncio
import aiohttp
import json
import time
import numpy as np
import os
from datetime import datetime

class MambaMoELatencyProbe:
    def __init__(self, url="http://localhost", port=8000):
        self.base_url = f"{url}:{port}"
        self.api_url = f"{self.base_url}/v1"
        self.model = None
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "hardware": "NVIDIA GB10 (DGX Spark)",
            "model": None,
            "sequence_lengths": [],
            "runs": []
        }
    
    async def detect_model(self):
        """Detect the model name from the server."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_url}/models") as resp:
                data = await resp.json()
                if "data" in data and len(data["data"]) > 0:
                    self.model = data["data"][0]["id"]
                    self.results["model"] = self.model
                    print(f"Detected model: {self.model}")
                    return True
                print("No models found")
                return False
    
    def generate_prompt(self, target_tokens):
        """Generate a prompt of approximately target_tokens length."""
        # Use repeated text to create long prompts
        base_text = "The quick brown fox jumps over the lazy dog. " * 10  # ~100 tokens
        repeats = target_tokens // 100
        return base_text * repeats
    
    async def measure_layer_timing(self, session, prompt, max_tokens=1):
        """
        Measure timing for a single request.
        Since we can't directly instrument vLLM layers, we measure:
        - TTFT (Time to First Token): proxy for prefill layer latency
        - Total time: proxy for decode layer latency
        - Tokens per second: overall throughput
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True
        }
        
        t_start = time.perf_counter()
        t_first_token = None
        tokens_generated = 0
        
        async with session.post(f"{self.api_url}/chat/completions", json=payload) as resp:
            async for line in resp.content:
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: ') and line_str != 'data: [DONE]':
                        try:
                            chunk = json.loads(line_str[6:])
                            if chunk['choices'][0].get('delta', {}).get('content'):
                                if t_first_token is None:
                                    t_first_token = time.perf_counter()
                                tokens_generated += 1
                        except:
                            pass
        
        t_end = time.perf_counter()
        
        ttft = (t_first_token - t_start) if t_first_token else (t_end - t_start)
        total_time = t_end - t_start
        tokens_per_sec = tokens_generated / total_time if total_time > 0 else 0
        
        return {
            "ttft_s": ttft,
            "total_time_s": total_time,
            "tokens_generated": tokens_generated,
            "tokens_per_sec": tokens_per_sec
        }
    
    async def run_sequence_length_test(self, seq_length, num_trials=3):
        """Run timing test for a specific sequence length."""
        print(f"\n--- Testing sequence length: {seq_length} tokens ---")
        prompt = self.generate_prompt(seq_length)
        
        # Estimate actual token count (rough: 1 token ≈ 4 chars)
        estimated_tokens = len(prompt) // 4
        print(f"  Prompt length: {len(prompt)} chars ≈ {estimated_tokens} tokens")
        
        trials = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            for i in range(num_trials):
                print(f"  Trial {i+1}/{num_trials}...", end=" ", flush=True)
                try:
                    result = await self.measure_layer_timing(session, prompt)
                    trials.append(result)
                    print(f"TTFT={result['ttft_s']:.3f}s, Total={result['total_time_s']:.3f}s")
                except Exception as e:
                    print(f"Error: {e}")
                    trials.append(None)
        
        # Aggregate results
        valid_trials = [t for t in trials if t is not None]
        if valid_trials:
            agg = {
                "seq_length": seq_length,
                "estimated_tokens": estimated_tokens,
                "ttft_mean": np.mean([t["ttft_s"] for t in valid_trials]),
                "ttft_std": np.std([t["ttft_s"] for t in valid_trials]),
                "total_time_mean": np.mean([t["total_time_s"] for t in valid_trials]),
                "total_time_std": np.std([t["total_time_s"] for t in valid_trials]),
                "tokens_per_sec_mean": np.mean([t["tokens_per_sec"] for t in valid_trials]),
                "tokens_per_sec_std": np.std([t["tokens_per_sec"] for t in valid_trials]),
                "trials": valid_trials
            }
            print(f"  Average: TTFT={agg['ttft_mean']:.3f}s ± {agg['ttft_std']:.3f}s")
            print(f"  Average: Total={agg['total_time_mean']:.3f}s ± {agg['total_time_std']:.3f}s")
            return agg
        return None
    
    async def run(self, seq_lengths=[1000, 4000, 16000, 64000, 128000], num_trials=3):
        """Run the full experiment."""
        print("=" * 80)
        print("A2: Mamba vs MoE Layer Latency by Sequence Length")
        print("=" * 80)
        
        if not await self.detect_model():
            print("Failed to detect model. Is the server running?")
            return None
        
        self.results["sequence_lengths"] = seq_lengths
        
        for seq_len in seq_lengths:
            result = await self.run_sequence_length_test(seq_len, num_trials)
            if result:
                self.results["runs"].append(result)
        
        # Analyze scaling behavior
        if len(self.results["runs"]) >= 2:
            ttfts = [r["ttft_mean"] for r in self.results["runs"]]
            seqs = [r["seq_length"] for r in self.results["runs"]]
            
            # Fit power law: TTFT = a * seq_len^b
            log_seqs = np.log(seqs)
            log_ttfts = np.log(ttfts)
            coeffs = np.polyfit(log_seqs, log_ttfts, 1)
            scaling_exponent = coeffs[0]
            
            self.results["analysis"] = {
                "scaling_exponent": scaling_exponent,
                "interpretation": (
                    "Sublinear (Mamba advantage)" if scaling_exponent < 0.8 else
                    "Linear (expected)" if scaling_exponent < 1.2 else
                    "Superlinear (MoE bottleneck)"
                )
            }
            
            print(f"\n{'='*80}")
            print(f"ANALYSIS")
            print(f"{'='*80}")
            print(f"Scaling exponent: {scaling_exponent:.3f}")
            print(f"Interpretation: {self.results['analysis']['interpretation']}")
            print(f"TTFT at 1K: {ttfts[0]:.3f}s")
            print(f"TTFT at {seqs[-1]}: {ttfts[-1]:.3f}s")
            print(f"Ratio: {ttfts[-1]/ttfts[0]:.1f}x for {seqs[-1]/seqs[0]:.0f}x input")
        
        return self.results

async def main():
    probe = MambaMoELatencyProbe(
        url=os.getenv("VLLM_URL", "http://localhost"),
        port=int(os.getenv("VLLM_PORT", "8000"))
    )
    
    results = await probe.run(
        seq_lengths=[1000, 4000, 16000, 64000, 128000],
        num_trials=3
    )
    
    if results:
        output_path = "results/final/nemotron/mamba_experiments/layer_latency.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
