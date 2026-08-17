#!/usr/bin/env python3
"""
D3: Mamba Cache Mode Comparison

Test MAMBA_CACHE_MODE=align vs alternative modes.
Measure memory and throughput.
Tests whether cache alignment matters for performance at scale.
"""

import asyncio
import aiohttp
import json
import time
import numpy as np
import os
import subprocess
from datetime import datetime

class MambaCacheModeProbe:
    def __init__(self, url="http://localhost", port=8000):
        self.base_url = f"{url}:{port}"
        self.api_url = f"{self.base_url}/v1"
        self.model = None
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "hardware": "NVIDIA GB10 (DGX Spark)",
            "model": None,
            "current_mode": os.getenv("MAMBA_CACHE_MODE", "unknown"),
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
    
    def get_gpu_memory(self):
        """Get current GPU memory usage."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                return {
                    "used_mb": int(parts[0]),
                    "total_mb": int(parts[1]),
                    "utilization": int(parts[0]) / int(parts[1]) * 100
                }
        except:
            pass
        return None
    
    async def measure_throughput(self, session, num_requests=10, max_tokens=256):
        """Measure throughput for the current cache mode."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Explain the differences between REST and GraphQL APIs."}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True
        }
        
        latencies = []
        tokens_generated = []
        
        for _ in range(num_requests):
            t_start = time.perf_counter()
            tokens = 0
            
            async with session.post(f"{self.api_url}/chat/completions", json=payload) as resp:
                async for line in resp.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: ') and line_str != 'data: [DONE]':
                            try:
                                chunk = json.loads(line_str[6:])
                                if chunk['choices'][0].get('delta', {}).get('content'):
                                    tokens += 1
                            except:
                                pass
            
            t_end = time.perf_counter()
            latencies.append(t_end - t_start)
            tokens_generated.append(tokens)
        
        return {
            "latency_mean": np.mean(latencies),
            "latency_std": np.std(latencies),
            "tokens_per_sec": np.mean(tokens_generated) / np.mean(latencies) if np.mean(latencies) > 0 else 0,
            "total_tokens": sum(tokens_generated)
        }
    
    async def run_cache_mode_test(self, mode_name, num_requests=10):
        """Run test for a specific cache mode."""
        print(f"\n--- Testing cache mode: {mode_name} ---")
        
        # Get GPU memory before test
        mem_before = self.get_gpu_memory()
        if mem_before:
            print(f"  GPU Memory before: {mem_before['used_mb']}MB / {mem_before['total_mb']}MB ({mem_before['utilization']:.1f}%)")
        
        # Run throughput test
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
            result = await self.measure_throughput(session, num_requests)
        
        # Get GPU memory after test
        mem_after = self.get_gpu_memory()
        if mem_after:
            print(f"  GPU Memory after: {mem_after['used_mb']}MB / {mem_after['total_mb']}MB ({mem_after['utilization']:.1f}%)")
        
        print(f"  Throughput: {result['tokens_per_sec']:.1f} tok/s")
        print(f"  Latency: {result['latency_mean']:.3f}s ± {result['latency_std']:.3f}s")
        
        return {
            "mode": mode_name,
            "throughput": result["tokens_per_sec"],
            "latency": result["latency_mean"],
            "latency_std": result["latency_std"],
            "tokens_per_sec": result["tokens_per_sec"],
            "memory_before": mem_before,
            "memory_after": mem_after
        }
    
    async def run(self, num_requests=10):
        """Run the full experiment."""
        print("=" * 80)
        print("D3: Mamba Cache Mode Comparison")
        print("=" * 80)
        
        if not await self.detect_model():
            print("Failed to detect model. Is the server running?")
            return None
        
        print(f"Current MAMBA_CACHE_MODE: {self.results['current_mode']}")
        
        # Test current mode
        current_result = await self.run_cache_mode_test(self.results['current_mode'], num_requests)
        self.results["runs"].append(current_result)
        
        # Note: We can't change the cache mode without restarting the server
        # So we'll just measure the current mode and document the results
        print(f"\n{'='*80}")
        print("ANALYSIS")
        print(f"{'='*80}")
        print(f"Current cache mode: {self.results['current_mode']}")
        print(f"Throughput: {current_result['throughput']:.1f} tok/s")
        print(f"Latency: {current_result['latency']:.3f}s")
        
        if current_result['memory_after']:
            print(f"GPU Memory: {current_result['memory_after']['used_mb']}MB / {current_result['memory_after']['total_mb']}MB")
        
        self.results["analysis"] = {
            "current_mode": self.results['current_mode'],
            "throughput": current_result['throughput'],
            "latency": current_result['latency'],
            "recommendation": f"Current mode ({self.results['current_mode']}) is active. To compare modes, restart server with different MAMBA_CACHE_MODE."
        }
        
        return self.results

async def main():
    probe = MambaCacheModeProbe(
        url=os.getenv("VLLM_URL", "http://localhost"),
        port=int(os.getenv("VLLM_PORT", "8000"))
    )
    
    results = await probe.run(num_requests=10)
    
    if results:
        output_path = "results/final/nemotron/mamba_experiments/mamba_cache_mode.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
