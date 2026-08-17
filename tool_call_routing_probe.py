#!/usr/bin/env python3
"""
C2: Tool-Call Routing Signatures

Inject tool-call format tokens (JSON, function definitions).
Compare routing vs plain text.
Tests whether tool-call-heavy workflows stress different experts.
"""

import asyncio
import aiohttp
import json
import time
import numpy as np
import os
from datetime import datetime

class ToolCallRoutingProbe:
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
    
    def create_tool_call_messages(self, tool_type="simple"):
        """Create messages with tool-call format."""
        tools = {
            "simple": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "City name"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                            },
                            "required": ["location"]
                        }
                    }
                }
            ],
            "complex": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_database",
                        "description": "Search for records in the database",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "filters": {
                                    "type": "object",
                                    "properties": {
                                        "date_range": {"type": "object"},
                                        "category": {"type": "string"},
                                        "status": {"type": "string", "enum": ["active", "inactive", "pending"]}
                                    }
                                },
                                "pagination": {
                                    "type": "object",
                                    "properties": {
                                        "page": {"type": "integer"},
                                        "per_page": {"type": "integer"}
                                    }
                                }
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "update_record",
                        "description": "Update a database record",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "data": {"type": "object"},
                                "version": {"type": "integer"}
                            },
                            "required": ["id", "data"]
                        }
                    }
                }
            ]
        }
        
        return [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": "Search for active users created in the last 30 days"},
            {"role": "assistant", "content": None, "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search_database",
                        "arguments": json.dumps({
                            "query": "users",
                            "filters": {
                                "date_range": {"start": "2026-07-18", "end": "2026-08-17"},
                                "status": "active"
                            }
                        })
                    }
                }
            ]},
            {"role": "tool", "content": json.dumps({"results": [{"id": "1", "name": "Alice"}, {"id": "2", "name": "Bob"}]})},
            {"role": "user", "content": "Now update the first user's status to inactive"}
        ]
    
    def create_plain_text_messages(self):
        """Create plain text messages (no tools)."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Search for active users created in the last 30 days"},
            {"role": "assistant", "content": "I found 2 active users created in the last 30 days: Alice and Bob."},
            {"role": "user", "content": "Now update the first user's status to inactive"}
        ]
    
    async def measure_response(self, session, messages, label, max_tokens=256):
        """Measure response for a given message set."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True
        }
        
        t_start = time.perf_counter()
        content = []
        tool_calls = []
        
        async with session.post(f"{self.api_url}/chat/completions", json=payload) as resp:
            async for line in resp.content:
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: ') and line_str != 'data: [DONE]':
                        try:
                            chunk = json.loads(line_str[6:])
                            delta = chunk['choices'][0].get('delta', {})
                            if delta.get('content'):
                                content.append(delta['content'])
                            if delta.get('tool_calls'):
                                tool_calls.extend(delta['tool_calls'])
                        except:
                            pass
        
        t_end = time.perf_counter()
        
        return {
            "label": label,
            "latency_s": t_end - t_start,
            "content_length": len(''.join(content)),
            "has_tool_calls": len(tool_calls) > 0,
            "content": ''.join(content)[:200]
        }
    
    async def run_tool_call_test(self, num_trials=3):
        """Run tool-call routing test."""
        print(f"\n--- Testing tool-call vs plain text routing ---")
        
        results = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
            # Test simple tool calls
            for i in range(num_trials):
                print(f"  Trial {i+1}/{num_trials}...")
                
                # Tool call messages
                tool_messages = self.create_tool_call_messages("simple")
                tool_result = await self.measure_response(session, tool_messages, "simple_tool")
                results.append(tool_result)
                
                # Plain text messages
                plain_messages = self.create_plain_text_messages()
                plain_result = await self.measure_response(session, plain_messages, "plain_text")
                results.append(plain_result)
                
                print(f"    Tool: {tool_result['latency_s']:.3f}s, Content: {tool_result['content_length']} chars")
                print(f"    Plain: {plain_result['latency_s']:.3f}s, Content: {plain_result['content_length']} chars")
        
        return results
    
    async def run(self, num_trials=3):
        """Run the full experiment."""
        print("=" * 80)
        print("C2: Tool-Call Routing Signatures")
        print("=" * 80)
        
        if not await self.detect_model():
            print("Failed to detect model. Is the server running?")
            return None
        
        # Run tests
        results = await self.run_tool_call_test(num_trials)
        
        # Analyze results
        tool_results = [r for r in results if r["label"] == "simple_tool"]
        plain_results = [r for r in results if r["label"] == "plain_text"]
        
        if tool_results and plain_results:
            tool_latencies = [r["latency_s"] for r in tool_results]
            plain_latencies = [r["latency_s"] for r in plain_results]
            
            tool_content = [r["content_length"] for r in tool_results]
            plain_content = [r["content_length"] for r in plain_results]
            
            analysis = {
                "tool_latency_mean": np.mean(tool_latencies),
                "plain_latency_mean": np.mean(plain_latencies),
                "latency_ratio": np.mean(tool_latencies) / np.mean(plain_latencies) if np.mean(plain_latencies) > 0 else 1.0,
                "tool_content_mean": np.mean(tool_content),
                "plain_content_mean": np.mean(plain_content),
                "content_ratio": np.mean(tool_content) / np.mean(plain_content) if np.mean(plain_content) > 0 else 1.0,
                "interpretation": (
                    "Tool calls cause different routing" if np.mean(tool_latencies) / np.mean(plain_latencies) > 1.2 else
                    "Tool calls have similar routing"
                )
            }
            
            self.results["analysis"] = analysis
            self.results["runs"] = results
            
            print(f"\n{'='*80}")
            print("ANALYSIS")
            print(f"{'='*80}")
            print(f"Tool call latency: {analysis['tool_latency_mean']:.3f}s")
            print(f"Plain text latency: {analysis['plain_latency_mean']:.3f}s")
            print(f"Latency ratio: {analysis['latency_ratio']:.2f}x")
            print(f"Tool content: {analysis['tool_content_mean']:.0f} chars")
            print(f"Plain content: {analysis['plain_content_mean']:.0f} chars")
            print(f"Content ratio: {analysis['content_ratio']:.2f}x")
            print(f"Interpretation: {analysis['interpretation']}")
        
        return self.results

async def main():
    probe = ToolCallRoutingProbe(
        url=os.getenv("VLLM_URL", "http://localhost"),
        port=int(os.getenv("VLLM_PORT", "8000"))
    )
    
    results = await probe.run(num_trials=3)
    
    if results:
        output_path = "results/final/nemotron/mamba_experiments/tool_call_routing.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
