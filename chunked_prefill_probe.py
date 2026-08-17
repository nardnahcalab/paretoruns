#!/usr/bin/env python3
"""
A3: Chunked Prefill at 256K

Tests vLLM's chunked prefill with 256K input.
Measures TTFT breakdown (prefill vs first decode).
Tests whether chunked prefill makes 256K inputs feasible.
"""

import asyncio
import aiohttp
import json
import time
import os
from datetime import datetime

class ChunkedPrefillProbe:
    def __init__(self, url="http://localhost", port=8000):
        self.base_url = f"{url}:{port}"
        self.api_url = f"{self.base_url}/v1"
        self.model = None
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "hardware": "NVIDIA GB10 (DGX Spark)",
            "model": None,
            "configurations": [],
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
    
    async def measure_ttft_breakdown(self, session, prompt, max_tokens=32):
        """
        Measure TTFT breakdown:
        - Total TTFT: time from request to first token
        - Prefill phase: time to process input
        - Decode phase: time to generate first token
        
        We infer the breakdown by:
        1. Measuring TTFT with stream=True (captures first token arrival)
        2. Measuring time to generate multiple tokens to estimate decode speed
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
        token_times = []
        
        async with session.post(f"{self.api_url}/chat/completions", json=payload) as resp:
            async for line in resp.content:
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: ') and line_str != 'data: [DONE]':
                        try:
                            chunk = json.loads(line_str[6:])
                            if chunk['choices'][0].get('delta', {}).get('content'):
                                t_now = time.perf_counter()
                                if t_first_token is None:
                                    t_first_token = t_now
                                token_times.append(t_now - t_start)
                        except:
                            pass
        
        t_end = time.perf_counter()
        
        ttft = (t_first_token - t_start) if t_first_token else (t_end - t_start)
        total_time = t_end - t_start
        
        # Estimate decode time per token after first token
        if len(token_times) > 1:
            decode_times = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
            avg_decode_time = sum(decode_times) / len(decode_times)
        else:
            avg_decode_time = 0
        
        # Rough breakdown: TTFT ≈ prefill + first decode token
        # First decode token ≈ avg_decode_time
        prefill_time = ttft - avg_decode_time if ttft > avg_decode_time else ttft * 0.9
        
        return {
            "ttft_s": ttft,
            "prefill_time_s": prefill_time,
            "first_decode_token_s": avg_decode_time,
            "total_time_s": total_time,
            "tokens_generated": len(token_times),
            "prompt_length_chars": len(prompt),
            "estimated_tokens": len(prompt) // 4
        }
    
    async def run_single_config(self, seq_length, max_tokens=32, num_trials=3):
        """Run test for a single configuration."""
        print(f"\n--- Testing: seq_len={seq_length}, max_tokens={max_tokens} ---")
        prompt = self.generate_prompt(seq_length)
        estimated_tokens = len(prompt) // 4
        print(f"  Prompt: {len(prompt)} chars ≈ {estimated_tokens} tokens")
        
        trials = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            for i in range(num_trials):
                print(f"  Trial {i+1}/{num_trials}...", end=" ", flush=True)
                try:
                    result = await self.measure_ttft_breakdown(session, prompt, max_tokens)
                    trials.append(result)
                    print(f"TTFT={result['ttft_s']:.3f}s, Prefill≈{result['prefill_time_s']:.3f}s")
                except Exception as e:
                    print(f"Error: {e}")
                    trials.append(None)
        
        valid_trials = [t for t in trials if t is not None]
        if valid_trials:
            agg = {
                "seq_length": seq_length,
                "max_tokens": max_tokens,
                "estimated_input_tokens": estimated_tokens,
                "ttft_mean": sum(t["ttft_s"] for t in valid_trials) / len(valid_trials),
                "ttft_std": (sum((t["ttft_s"] - sum(t2["ttft_s"] for t2 in valid_trials)/len(valid_trials))**2 for t in valid_trials) / len(valid_trials))**0.5,
                "prefill_time_mean": sum(t["prefill_time_s"] for t in valid_trials) / len(valid_trials),
                "first_decode_token_mean": sum(t["first_decode_token_s"] for t in valid_trials) / len(valid_trials),
                "total_time_mean": sum(t["total_time_s"] for t in valid_trials) / len(valid_trials),
                "trials": valid_trials
            }
            print(f"  Average TTFT: {agg['ttft_mean']:.3f}s ± {agg['ttft_std']:.3f}s")
            return agg
        return None
    
    async def run(self):
        """Run the full experiment."""
        print("=" * 80)
        print("A3: Chunked Prefill at 256K")
        print("=" * 80)
        
        if not await self.detect_model():
            print("Failed to detect model. Is the server running?")
            return None
        
        # Test configurations
        configs = [
            {"seq_length": 4000, "max_tokens": 32},
            {"seq_length": 16000, "max_tokens": 32},
            {"seq_length": 64000, "max_tokens": 32},
            {"seq_length": 128000, "max_tokens": 32},
            {"seq_length": 256000, "max_tokens": 32},
        ]
        
        for config in configs:
            result = await self.run_single_config(config["seq_length"], config["max_tokens"], num_trials=3)
            if result:
                self.results["runs"].append(result)
        
        # Analyze scaling
        if len(self.results["runs"]) >= 2:
            ttfts = [r["ttft_mean"] for r in self.results["runs"]]
            seqs = [r["seq_length"] for r in self.results["runs"]]
            
            # Check if 256K is feasible (TTFT < 30s)
            max_ttft = max(ttfts)
            feasible = max_ttft < 30
            
            self.results["analysis"] = {
                "max_ttft_s": max_ttft,
                "feasible_256k": feasible,
                "interpretation": (
                    "256K is feasible with chunked prefill" if feasible else
                    "256K TTFT too high, needs optimization"
                )
            }
            
            print(f"\n{'='*80}")
            print(f"ANALYSIS")
            print(f"{'='*80}")
            print(f"Max TTFT: {max_ttft:.3f}s")
            print(f"256K feasible: {'Yes' if feasible else 'No'}")
            print(f"Interpretation: {self.results['analysis']['interpretation']}")
        
        return self.results

async def main():
    probe = ChunkedPrefillProbe(
        url=os.getenv("VLLM_URL", "http://localhost"),
        port=int(os.getenv("VLLM_PORT", "8000"))
    )
    
    results = await probe.run()
    
    if results:
        output_path = "results/final/nemotron/mamba_experiments/chunked_prefill.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
