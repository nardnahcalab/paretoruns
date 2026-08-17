#!/usr/bin/env python3
"""
E4: Power Efficiency at Scale

tok/s/W at each concurrency level.
Total cost of ownership for N-GPU deployment.
"""

import asyncio
import aiohttp
import json
import time
import numpy as np
import os
import subprocess
from datetime import datetime

class PowerEfficiencyProbe:
    def __init__(self, url="http://localhost", port=8000):
        self.base_url = f"{url}:{port}"
        self.api_url = f"{self.base_url}/v1"
        self.model = None
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "hardware": "NVIDIA GB10 (DGX Spark)",
            "model": None,
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
    
    def get_gpu_power(self):
        """Get current GPU power usage."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,power.limit", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                return {
                    "power_w": float(parts[0]),
                    "power_limit_w": float(parts[1]),
                    "utilization": float(parts[0]) / float(parts[1]) * 100
                }
        except:
            pass
        return None
    
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
    
    async def measure_power_efficiency(self, session, concurrency, duration_s=30):
        """Measure power efficiency at a given concurrency level."""
        print(f"\n--- Testing concurrency {concurrency} for {duration_s}s ---")
        
        # Get initial power reading
        power_before = self.get_gpu_power()
        if power_before:
            print(f"  Power before: {power_before['power_w']:.1f}W / {power_before['power_limit_w']:.1f}W")
        
        # Run concurrent requests
        start_time = time.perf_counter()
        total_tokens = 0
        total_requests = 0
        
        async def send_request():
            nonlocal total_tokens, total_requests
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Explain the differences between REST and GraphQL APIs."}],
                "max_tokens": 256,
                "temperature": 0.7,
                "stream": True
            }
            
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
            
            total_tokens += tokens
            total_requests += 1
        
        # Create tasks
        tasks = []
        for _ in range(concurrency):
            task = asyncio.create_task(send_request())
            tasks.append(task)
        
        # Wait for all tasks
        await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        # Get final power reading
        power_after = self.get_gpu_power()
        if power_after:
            print(f"  Power after: {power_after['power_w']:.1f}W / {power_after['power_limit_w']:.1f}W")
        
        # Get memory usage
        memory = self.get_gpu_memory()
        if memory:
            print(f"  Memory: {memory['used_mb']}MB / {memory['total_mb']}MB ({memory['utilization']:.1f}%)")
        
        # Calculate metrics
        throughput = total_tokens / elapsed if elapsed > 0 else 0
        requests_per_second = total_requests / elapsed if elapsed > 0 else 0
        
        # Power efficiency (tokens per watt-second)
        avg_power = (power_before['power_w'] + power_after['power_w']) / 2 if power_before and power_after else 0
        power_efficiency = throughput / avg_power if avg_power > 0 else 0
        
        print(f"  Throughput: {throughput:.1f} tok/s")
        print(f"  Requests: {total_requests} ({requests_per_second:.2f} req/s)")
        print(f"  Avg Power: {avg_power:.1f}W")
        print(f"  Power Efficiency: {power_efficiency:.3f} tok/s/W")
        
        return {
            "concurrency": concurrency,
            "throughput": throughput,
            "requests_per_second": requests_per_second,
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "elapsed_s": elapsed,
            "avg_power_w": avg_power,
            "power_efficiency": power_efficiency,
            "memory": memory
        }
    
    async def run(self, concurrencies=[1, 4, 8, 16, 32, 64]):
        """Run the full experiment."""
        print("=" * 80)
        print("E4: Power Efficiency at Scale")
        print("=" * 80)
        
        if not await self.detect_model():
            print("Failed to detect model. Is the server running?")
            return None
        
        # Test different concurrency levels
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            for conc in concurrencies:
                result = await self.measure_power_efficiency(session, conc, duration_s=30)
                self.results["runs"].append(result)
        
        # Analyze results
        if len(self.results["runs"]) >= 2:
            throughputs = [r["throughput"] for r in self.results["runs"]]
            concs = [r["concurrency"] for r in self.results["runs"]]
            efficiencies = [r["power_efficiency"] for r in self.results["runs"]]
            
            # Find optimal efficiency
            max_efficiency_idx = np.argmax(efficiencies)
            optimal_conc = concs[max_efficiency_idx]
            max_efficiency = efficiencies[max_efficiency_idx]
            
            self.results["analysis"] = {
                "optimal_concurrency": optimal_conc,
                "max_efficiency": max_efficiency,
                "efficiency_curve": list(zip(concs, efficiencies)),
                "interpretation": (
                    f"Optimal efficiency at c{optimal_conc} with {max_efficiency:.3f} tok/s/W"
                )
            }
            
            print(f"\n{'='*80}")
            print("ANALYSIS")
            print(f"{'='*80}")
            print(f"Optimal concurrency: {optimal_conc}")
            print(f"Max efficiency: {max_efficiency:.3f} tok/s/W")
            print(f"Efficiency curve: {list(zip(concs, efficiencies))}")
        
        return self.results

async def main():
    probe = PowerEfficiencyProbe(
        url=os.getenv("VLLM_URL", "http://localhost"),
        port=int(os.getenv("VLLM_PORT", "8000"))
    )
    
    results = await probe.run(concurrencies=[1, 4, 8, 16, 32, 64])
    
    if results:
        output_path = "results/final/nemotron/mamba_experiments/power_efficiency.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
