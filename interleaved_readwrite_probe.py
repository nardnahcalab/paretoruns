#!/usr/bin/env python3
"""
C4: Interleaved Read/Write Patterns

Simulate agent: generate tool call -> parse result -> generate next response.
Measure throughput per phase.
Tests latency breakdown for real agent loops.
"""

import asyncio
import aiohttp
import json
import time
import numpy as np
import os
from datetime import datetime

class InterleavedReadWriteProbe:
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
    
    def create_agent_loop_messages(self, step):
        """Create messages for an agent loop step."""
        if step == 0:
            # Initial user request
            return [
                {"role": "system", "content": "You are a helpful assistant with access to tools."},
                {"role": "user", "content": "Search for active users and update their status"}
            ]
        elif step == 1:
            # After tool call result
            return [
                {"role": "system", "content": "You are a helpful assistant with access to tools."},
                {"role": "user", "content": "Search for active users and update their status"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_database",
                            "arguments": json.dumps({"query": "users", "status": "active"})
                        }
                    }
                ]},
                {"role": "tool", "content": json.dumps({"results": [{"id": "1", "name": "Alice"}, {"id": "2", "name": "Bob"}]})}
            ]
        elif step == 2:
            # After second tool call
            return [
                {"role": "system", "content": "You are a helpful assistant with access to tools."},
                {"role": "user", "content": "Search for active users and update their status"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_database",
                            "arguments": json.dumps({"query": "users", "status": "active"})
                        }
                    }
                ]},
                {"role": "tool", "content": json.dumps({"results": [{"id": "1", "name": "Alice"}, {"id": "2", "name": "Bob"}]})},
                {"role": "assistant", "content": "I found 2 active users. Now I'll update their status."},
                {"role": "assistant", "content": None, "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "update_record",
                            "arguments": json.dumps({"id": "1", "data": {"status": "inactive"}})
                        }
                    }
                ]},
                {"role": "tool", "content": json.dumps({"success": True})}
            ]
        else:
            # Final response
            return [
                {"role": "system", "content": "You are a helpful assistant with access to tools."},
                {"role": "user", "content": "Search for active users and update their status"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_database",
                            "arguments": json.dumps({"query": "users", "status": "active"})
                        }
                    }
                ]},
                {"role": "tool", "content": json.dumps({"results": [{"id": "1", "name": "Alice"}, {"id": "2", "name": "Bob"}]})},
                {"role": "assistant", "content": "I found 2 active users. Now I'll update their status."},
                {"role": "assistant", "content": None, "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "update_record",
                            "arguments": json.dumps({"id": "1", "data": {"status": "inactive"}})
                        }
                    }
                ]},
                {"role": "tool", "content": json.dumps({"success": True})},
                {"role": "assistant", "content": "I've successfully updated Alice's status to inactive. Bob remains active."}
            ]
    
    async def measure_phase(self, session, messages, phase_name, max_tokens=256):
        """Measure a single phase of the agent loop."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
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
                            delta = chunk['choices'][0].get('delta', {})
                            if delta.get('content'):
                                tokens_generated += 1
                                content.append(delta['content'])
                        except:
                            pass
        
        t_end = time.perf_counter()
        
        return {
            "phase": phase_name,
            "latency_s": t_end - t_start,
            "tokens_generated": tokens_generated,
            "throughput": tokens_generated / (t_end - t_start) if (t_end - t_start) > 0 else 0
        }
    
    async def run_agent_loop_test(self, num_loops=5):
        """Run the interleaved read/write test."""
        print(f"\n--- Running {num_loops} agent loops ---")
        
        all_results = []
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
            for loop_idx in range(num_loops):
                print(f"\n  Agent Loop {loop_idx + 1}/{num_loops}:")
                loop_results = []
                
                # Simulate 4-step agent loop
                for step in range(4):
                    messages = self.create_agent_loop_messages(step)
                    phase_name = f"step_{step}"
                    
                    result = await self.measure_phase(session, messages, phase_name)
                    loop_results.append(result)
                    
                    print(f"    Step {step}: {result['latency_s']:.3f}s, {result['tokens_generated']} tokens")
                
                all_results.append(loop_results)
        
        return all_results
    
    async def run(self, num_loops=5):
        """Run the full experiment."""
        print("=" * 80)
        print("C4: Interleaved Read/Write Patterns")
        print("=" * 80)
        
        if not await self.detect_model():
            print("Failed to detect model. Is the server running?")
            return None
        
        # Run test
        all_results = await self.run_agent_loop_test(num_loops)
        
        if all_results:
            # Analyze results
            step_latencies = [[] for _ in range(4)]
            step_throughputs = [[] for _ in range(4)]
            
            for loop_results in all_results:
                for step_idx, step_result in enumerate(loop_results):
                    step_latencies[step_idx].append(step_result["latency_s"])
                    step_throughputs[step_idx].append(step_result["throughput"])
            
            # Calculate averages
            avg_latencies = [np.mean(lats) for lats in step_latencies]
            avg_throughputs = [np.mean(tputs) for tputs in step_throughputs]
            
            # Total loop time
            total_loop_time = sum(avg_latencies)
            
            analysis = {
                "num_loops": num_loops,
                "avg_step_latencies": avg_latencies,
                "avg_step_throughputs": avg_throughputs,
                "total_loop_time": total_loop_time,
                "read_phase_time": avg_latencies[1],  # Step 1: process tool result
                "write_phase_time": avg_latencies[2],  # Step 2: generate tool call
                "read_write_ratio": avg_latencies[1] / avg_latencies[2] if avg_latencies[2] > 0 else 1.0,
                "interpretation": (
                    "Read phase dominates" if avg_latencies[1] > avg_latencies[2] * 1.2 else
                    "Write phase dominates" if avg_latencies[2] > avg_latencies[1] * 1.2 else
                    "Balanced read/write"
                )
            }
            
            self.results["analysis"] = analysis
            self.results["runs"] = all_results
            
            print(f"\n{'='*80}")
            print("ANALYSIS")
            print(f"{'='*80}")
            print(f"Total loop time: {total_loop_time:.3f}s")
            print(f"Step latencies: {[f'{l:.3f}s' for l in avg_latencies]}")
            print(f"Read phase: {analysis['read_phase_time']:.3f}s")
            print(f"Write phase: {analysis['write_phase_time']:.3f}s")
            print(f"Read/Write ratio: {analysis['read_write_ratio']:.2f}x")
            print(f"Interpretation: {analysis['interpretation']}")
        
        return self.results

async def main():
    probe = InterleavedReadWriteProbe(
        url=os.getenv("VLLM_URL", "http://localhost"),
        port=int(os.getenv("VLLM_PORT", "8000"))
    )
    
    results = await probe.run(num_loops=5)
    
    if results:
        output_path = "results/final/nemotron/mamba_experiments/interleaved_readwrite.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
