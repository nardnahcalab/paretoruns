#!/usr/bin/env python3
"""
B3: Mixed-Concurrency Realistic Load

Poisson-distributed arrivals, variable output lengths (128-2048), concurrent sessions c32-c128.
Real-world throughput, not just uniform synthetic load.
"""

import asyncio
import aiohttp
import json
import time
import numpy as np
import os
from datetime import datetime

class MixedConcurrencyProbe:
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
    
    def generate_request(self, session_id):
        """Generate a realistic request with variable output length."""
        # Variable output lengths following a realistic distribution
        output_lengths = [128, 256, 384, 512, 768, 1024, 1536, 2048]
        weights = [0.25, 0.30, 0.15, 0.12, 0.08, 0.05, 0.03, 0.02]
        max_tokens = np.random.choice(output_lengths, p=weights)
        
        # Realistic prompts
        prompts = [
            "Explain the differences between REST and GraphQL APIs.",
            "How do I implement authentication in a FastAPI application?",
            "What are the best practices for handling errors in distributed systems?",
            "Write a Python function to find the longest palindromic substring.",
            "How should I structure a microservices architecture for an e-commerce platform?",
            "Explain the CQRS pattern and when to use it.",
            "What's the optimal way to cache database queries in a Django application?",
            "How do I implement rate limiting in an Express.js application?",
            "What are the security considerations for a REST API?",
            "How should I design a notification system that scales?",
        ]
        
        return {
            "messages": [{"role": "user", "content": np.random.choice(prompts)}],
            "max_tokens": int(max_tokens),
            "session_id": session_id
        }
    
    async def send_request(self, session, request, start_time):
        """Send a single request and measure timing."""
        payload = {
            "model": self.model,
            "messages": request["messages"],
            "max_tokens": request["max_tokens"],
            "temperature": 0.7,
            "stream": True
        }
        
        t_start = time.perf_counter()
        content = []
        tokens_generated = 0
        
        async with session.post(f"{self.api_url}/chat/completions", json=payload) as resp:
            async for line in resp.content:
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: ') and line_str != 'data: [DONE]':
                        try:
                            chunk = json.loads(line_str[6:])
                            if chunk['choices'][0].get('delta', {}).get('content'):
                                tokens_generated += 1
                                content.append(chunk['choices'][0]['delta']['content'])
                        except:
                            pass
        
        t_end = time.perf_counter()
        
        return {
            "session_id": request["session_id"],
            "max_tokens": request["max_tokens"],
            "tokens_generated": tokens_generated,
            "latency_s": t_end - t_start,
            "throughput": tokens_generated / (t_end - t_start) if (t_end - t_start) > 0 else 0,
            "arrival_offset": t_start - start_time
        }
    
    async def run_mixed_concurrency_test(self, concurrency, duration_s=60, num_sessions=100):
        """Run mixed-concurrency test with Poisson arrivals."""
        print(f"\n--- Testing concurrency {concurrency} for {duration_s}s ---")
        
        results = []
        start_time = time.perf_counter()
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            # Generate Poisson arrivals
            arrival_rate = concurrency / 10  # arrivals per second
            inter_arrival_times = np.random.exponential(1.0 / arrival_rate, num_sessions)
            
            tasks = []
            for i in range(num_sessions):
                # Calculate arrival time
                arrival_time = sum(inter_arrival_times[:i+1])
                
                # Wait until arrival time
                wait_time = arrival_time - (time.perf_counter() - start_time)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                # Create and send request
                request = self.generate_request(i)
                task = asyncio.create_task(self.send_request(session, request, start_time))
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter valid results
        valid_results = [r for r in results if isinstance(r, dict)]
        
        if valid_results:
            # Calculate metrics
            latencies = [r["latency_s"] for r in valid_results]
            throughputs = [r["throughput"] for r in valid_results]
            tokens_generated = [r["tokens_generated"] for r in valid_results]
            
            return {
                "concurrency": concurrency,
                "duration_s": duration_s,
                "num_requests": len(valid_results),
                "latency_mean": np.mean(latencies),
                "latency_std": np.std(latencies),
                "latency_p50": np.percentile(latencies, 50),
                "latency_p95": np.percentile(latencies, 95),
                "latency_p99": np.percentile(latencies, 99),
                "throughput_mean": np.mean(throughputs),
                "throughput_std": np.std(throughputs),
                "total_tokens": sum(tokens_generated),
                "avg_tokens_per_request": np.mean(tokens_generated),
                "requests_per_second": len(valid_results) / duration_s,
                "results": valid_results
            }
        return None
    
    async def run(self):
        """Run the full experiment."""
        print("=" * 80)
        print("B3: Mixed-Concurrency Realistic Load")
        print("=" * 80)
        
        if not await self.detect_model():
            print("Failed to detect model. Is the server running?")
            return None
        
        # Test different concurrency levels
        concurrencies = [32, 64, 128]
        
        for conc in concurrencies:
            result = await self.run_mixed_concurrency_test(conc, duration_s=60, num_sessions=conc*2)
            if result:
                self.results["runs"].append(result)
                print(f"\nResults for c{conc}:")
                print(f"  Requests: {result['num_requests']}")
                print(f"  Latency: {result['latency_mean']:.2f}s ± {result['latency_std']:.2f}s")
                print(f"  Throughput: {result['throughput_mean']:.1f} tok/s")
                print(f"  RPS: {result['requests_per_second']:.2f}")
        
        # Analyze scaling
        if len(self.results["runs"]) >= 2:
            throughputs = [r["throughput_mean"] for r in self.results["runs"]]
            concs = [r["concurrency"] for r in self.results["runs"]]
            
            # Fit scaling curve
            if len(concs) >= 2:
                coeffs = np.polyfit(concs, throughputs, 1)
                scaling_slope = coeffs[0]
                
                self.results["analysis"] = {
                    "scaling_slope": scaling_slope,
                    "interpretation": (
                        "Linear scaling" if 0.8 < scaling_slope < 1.2 else
                        "Sublinear scaling" if scaling_slope < 0.8 else
                        "Superlinear scaling"
                    )
                }
                
                print(f"\n{'='*80}")
                print("ANALYSIS")
                print(f"{'='*80}")
                print(f"Scaling slope: {scaling_slope:.3f}")
                print(f"Interpretation: {self.results['analysis']['interpretation']}")
        
        return self.results

async def main():
    probe = MixedConcurrencyProbe(
        url=os.getenv("VLLM_URL", "http://localhost"),
        port=int(os.getenv("VLLM_PORT", "8000"))
    )
    
    results = await probe.run()
    
    if results:
        output_path = "results/final/nemotron/mamba_experiments/mixed_concurrency.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
