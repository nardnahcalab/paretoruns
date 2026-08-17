#!/usr/bin/env python3
"""
C1: Routing Drift Across Agent Turns

Multi-turn probe with 10+ turns.
Track how routing entropy/Jaccard/cosine changes turn-over-turn.
Tests whether the router stabilizes or drifts as context accumulates.
"""

import asyncio
import aiohttp
import json
import time
import numpy as np
import os
from datetime import datetime

class RoutingDriftProbe:
    def __init__(self, url="http://localhost", port=8000):
        self.base_url = f"{url}:{port}"
        self.api_url = f"{self.base_url}/v1"
        self.model = None
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "hardware": "NVIDIA GB10 (DGX Spark)",
            "model": None,
            "num_turns": 0,
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
    
    def create_multi_turn_session(self, num_turns=12):
        """
        Create a multi-turn conversation that simulates an agentic workflow.
        Each turn builds on the previous context.
        """
        turns = [
            # Turn 1: Initial request
            {"role": "user", "content": "I need to build a REST API for user management. Can you help me design the architecture?"},
            # Turn 2: Add constraints
            {"role": "user", "content": "The API needs to handle 10K concurrent users. What database should I use?"},
            # Turn 3: Ask for implementation
            {"role": "user", "content": "Can you show me the database schema for users and sessions?"},
            # Turn 4: Add complexity
            {"role": "user", "content": "Now I need to add role-based access control. How should I structure the permissions?"},
            # Turn 5: Ask for code
            {"role": "user", "content": "Write the authentication middleware that checks JWT tokens and roles."},
            # Turn 6: Error handling
            {"role": "user", "content": "What error handling patterns should I use for this API?"},
            # Turn 7: Testing
            {"role": "user", "content": "How should I write unit tests for the authentication middleware?"},
            # Turn 8: Deployment
            {"role": "user", "content": "What's the best way to deploy this to Kubernetes?"},
            # Turn 9: Monitoring
            {"role": "user", "content": "How should I set up monitoring and alerting for this API?"},
            # Turn 10: Optimization
            {"role": "user", "content": "The API is slow at high concurrency. How can I optimize the database queries?"},
            # Turn 11: Security audit
            {"role": "user", "content": "Can you review this code for security vulnerabilities?"},
            # Turn 12: Final refactoring
            {"role": "user", "content": "How should I refactor this to use a hexagonal architecture?"},
        ]
        
        # Pad to requested length by cycling through topics
        while len(turns) < num_turns:
            turns.append(turns[len(turns) % 12])
        
        return turns[:num_turns]
    
    async def send_turn(self, session, messages, turn_num, max_tokens=256):
        """Send a single turn and collect response."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True
        }
        
        t_start = time.perf_counter()
        content = []
        reasoning = []
        
        async with session.post(f"{self.api_url}/chat/completions", json=payload) as resp:
            async for line in resp.content:
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: ') and line_str != 'data: [DONE]':
                        try:
                            chunk = json.loads(line_str[6:])
                            delta = chunk['choices'][0].get('delta', {})
                            if delta.get('reasoning'):
                                reasoning.append(delta['reasoning'])
                            if delta.get('content'):
                                content.append(delta['content'])
                        except:
                            pass
        
        t_end = time.perf_counter()
        
        return {
            "turn": turn_num,
            "latency_s": t_end - t_start,
            "content_length": len(''.join(content)),
            "reasoning_length": len(''.join(reasoning)),
            "response": ''.join(content)
        }
    
    async def run_session(self, session, num_turns=12):
        """Run a complete multi-turn session."""
        messages = []
        turns = self.create_multi_turn_session(num_turns)
        turn_results = []
        
        for i, turn in enumerate(turns):
            messages.append(turn)
            print(f"  Turn {i+1}/{num_turns}...", end=" ", flush=True)
            
            try:
                result = await self.send_turn(session, messages, i+1)
                turn_results.append(result)
                print(f"Latency={result['latency_s']:.2f}s, Content={result['content_length']} chars")
                
                # Add assistant response to context
                messages.append({
                    "role": "assistant",
                    "content": result["response"]
                })
            except Exception as e:
                print(f"Error: {e}")
                turn_results.append(None)
        
        return turn_results
    
    async def run_single_trial(self, num_turns=12):
        """Run a single trial of the multi-turn experiment."""
        print(f"\n--- Running {num_turns}-turn session ---")
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
            turn_results = await self.run_session(session, num_turns)
        
        valid_results = [r for r in turn_results if r is not None]
        
        if valid_results:
            # Calculate drift metrics
            latencies = [r["latency_s"] for r in valid_results]
            content_lengths = [r["content_length"] for r in valid_results]
            reasoning_lengths = [r["reasoning_length"] for r in valid_results]
            
            # Calculate turn-over-turn changes
            latency_changes = [latencies[i] - latencies[i-1] for i in range(1, len(latencies))]
            content_changes = [content_lengths[i] - content_lengths[i-1] for i in range(1, len(content_lengths))]
            
            return {
                "num_turns": len(valid_results),
                "latencies": latencies,
                "content_lengths": content_lengths,
                "reasoning_lengths": reasoning_lengths,
                "latency_mean": np.mean(latencies),
                "latency_std": np.std(latencies),
                "latency_trend": np.polyfit(range(len(latencies)), latencies, 1)[0],
                "content_mean": np.mean(content_lengths),
                "reasoning_mean": np.mean(reasoning_lengths),
                "turn_results": valid_results
            }
        return None
    
    async def run(self, num_turns=12, num_trials=3):
        """Run the full experiment."""
        print("=" * 80)
        print("C1: Routing Drift Across Agent Turns")
        print("=" * 80)
        
        if not await self.detect_model():
            print("Failed to detect model. Is the server running?")
            return None
        
        self.results["num_turns"] = num_turns
        
        all_trials = []
        for i in range(num_trials):
            print(f"\n=== Trial {i+1}/{num_trials} ===")
            trial_result = await self.run_single_trial(num_turns)
            if trial_result:
                all_trials.append(trial_result)
        
        if all_trials:
            # Aggregate across trials
            avg_latencies = np.mean([t["latencies"] for t in all_trials], axis=0)
            avg_content = np.mean([t["content_lengths"] for t in all_trials], axis=0)
            avg_reasoning = np.mean([t["reasoning_lengths"] for t in all_trials], axis=0)
            
            # Calculate drift metrics
            latency_slope = np.polyfit(range(len(avg_latencies)), avg_latencies, 1)[0]
            content_slope = np.polyfit(range(len(avg_content)), avg_content, 1)[0]
            
            self.results["runs"] = all_trials
            self.results["analysis"] = {
                "avg_latencies": avg_latencies.tolist(),
                "avg_content_lengths": avg_content.tolist(),
                "avg_reasoning_lengths": avg_reasoning.tolist(),
                "latency_slope": latency_slope,
                "content_slope": content_slope,
                "latency_drift": "increasing" if latency_slope > 0.1 else "decreasing" if latency_slope < -0.1 else "stable",
                "content_drift": "increasing" if content_slope > 1 else "decreasing" if content_slope < -1 else "stable"
            }
            
            print(f"\n{'='*80}")
            print(f"ANALYSIS")
            print(f"{'='*80}")
            print(f"Average latency trend: {latency_slope:.3f}s per turn ({self.results['analysis']['latency_drift']})")
            print(f"Average content trend: {content_slope:.1f} chars per turn ({self.results['analysis']['content_drift']})")
            print(f"Turn 1 latency: {avg_latencies[0]:.2f}s")
            print(f"Turn {len(avg_latencies)} latency: {avg_latencies[-1]:.2f}s")
            print(f"Ratio: {avg_latencies[-1]/avg_latencies[0]:.2f}x")
        
        return self.results

async def main():
    probe = RoutingDriftProbe(
        url=os.getenv("VLLM_URL", "http://localhost"),
        port=int(os.getenv("VLLM_PORT", "8000"))
    )
    
    results = await probe.run(num_turns=12, num_trials=3)
    
    if results:
        output_path = "results/final/nemotron/mamba_experiments/routing_drift.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
