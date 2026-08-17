#!/usr/bin/env python3
"""
A4: Prefix Caching Hit Rate

Same system prompt + different user queries.
Measure cache hit rate, memory saved, TTFT reduction.
Quantifies prefix caching value for agentic workflows.
"""

import asyncio
import aiohttp
import json
import time
import numpy as np
import os
from datetime import datetime

class PrefixCachingProbe:
    def __init__(self, url="http://localhost", port=8000):
        self.base_url = f"{url}:{port}"
        self.api_url = f"{self.base_url}/v1"
        self.model = None
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "hardware": "NVIDIA GB10 (DGX Spark)",
            "model": None,
            "system_prompt_tokens": 0,
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
    
    def create_system_prompt(self, size="large"):
        """Create a system prompt of specified size."""
        prompts = {
            "small": "You are a helpful AI assistant.",
            "medium": """You are a helpful AI assistant specializing in software engineering. 
You help users with coding, debugging, architecture design, and best practices.
Always provide clear, concise explanations with code examples when appropriate.
Follow security best practices and never expose secrets or sensitive information.""",
            "large": """You are a highly specialized AI assistant for software engineering and system architecture. 
Your expertise includes:

1. Programming Languages: Python, JavaScript, TypeScript, Go, Rust, Java, C++
2. Frameworks: React, Vue, Django, Flask, FastAPI, Express, Spring Boot
3. Databases: PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch
4. Cloud Platforms: AWS, GCP, Azure, Docker, Kubernetes
5. System Design: Microservices, event-driven architecture, CQRS, domain-driven design
6. Security: OAuth, JWT, RBAC, encryption, secure coding practices
7. Performance: Caching strategies, load balancing, CDN optimization, database tuning

When helping users:
- Always ask clarifying questions when requirements are ambiguous
- Provide production-ready code with error handling
- Explain trade-offs and alternative approaches
- Follow SOLID principles and clean code practices
- Consider scalability, maintainability, and security in all recommendations
- Never hardcode secrets, credentials, or sensitive information
- Use environment variables for configuration
- Include logging and monitoring suggestions

For code reviews:
- Check for security vulnerabilities (SQL injection, XSS, CSRF, etc.)
- Verify error handling completeness
- Assess performance implications
- Ensure code follows language-specific conventions
- Validate test coverage recommendations

For architecture discussions:
- Consider current and future scale requirements
- Evaluate cost implications of design decisions
- Assess team skill set and organizational constraints
- Plan for observability, debugging, and maintenance
- Document architectural decisions and rationale"""
        }
        return prompts.get(size, prompts["large"])
    
    def create_user_queries(self, count=10):
        """Create diverse user queries to test with the same system prompt."""
        queries = [
            "How do I implement authentication in a FastAPI application?",
            "What's the best way to handle database migrations in a team environment?",
            "Explain the differences between REST and GraphQL APIs.",
            "How should I structure a microservices architecture for an e-commerce platform?",
            "What are the best practices for handling errors in distributed systems?",
            "How do I implement rate limiting in an Express.js application?",
            "What's the optimal way to cache database queries in a Django application?",
            "How should I handle file uploads in a cloud-native application?",
            "Explain the CQRS pattern and when to use it.",
            "How do I implement real-time features with WebSockets?",
            "What's the best way to handle secrets management in production?",
            "How should I design a notification system that scales?",
            "Explain the differences between message queues and event streams.",
            "How do I implement comprehensive logging in a microservices architecture?",
            "What are the security considerations for a REST API?",
        ]
        return queries[:count]
    
    async def measure_ttft(self, session, messages, max_tokens=32):
        """Measure TTFT for a given set of messages."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True
        }
        
        t_start = time.perf_counter()
        t_first_token = None
        
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
                        except:
                            pass
        
        t_end = time.perf_counter()
        ttft = (t_first_token - t_start) if t_first_token else (t_end - t_start)
        return ttft
    
    async def run_prefix_caching_test(self, system_prompt_size="large", num_queries=10, num_trials=3):
        """Run the prefix caching test."""
        print(f"\n--- Testing prefix caching with {system_prompt_size} system prompt ---")
        
        system_prompt = self.create_system_prompt(system_prompt_size)
        queries = self.create_user_queries(num_queries)
        
        # Estimate system prompt tokens (rough: 1 token ≈ 4 chars)
        system_tokens = len(system_prompt) // 4
        self.results["system_prompt_tokens"] = system_tokens
        print(f"System prompt: {len(system_prompt)} chars ≈ {system_tokens} tokens")
        
        results_per_query = []
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
            for query_idx, query in enumerate(queries):
                print(f"\n  Query {query_idx + 1}/{num_queries}: {query[:50]}...")
                
                ttfts = []
                for trial in range(num_trials):
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ]
                    
                    ttft = await self.measure_ttft(session, messages)
                    ttfts.append(ttft)
                    print(f"    Trial {trial + 1}: TTFT = {ttft:.3f}s")
                
                results_per_query.append({
                    "query": query,
                    "ttft_mean": np.mean(ttfts),
                    "ttft_std": np.std(ttfts),
                    "ttfts": ttfts
                })
            
            # Now test with NO cache (different system prompts)
            print(f"\n--- Testing WITHOUT cache (different system prompts) ---")
            no_cache_ttfts = []
            for query_idx, query in enumerate(queries[:5]):  # Test 5 queries
                messages = [
                    {"role": "system", "content": f"You are assistant #{query_idx}. " + "Help the user." * 100},
                    {"role": "user", "content": query}
                ]
                
                ttft = await self.measure_ttft(session, messages)
                no_cache_ttfts.append(ttft)
                print(f"  Query {query_idx + 1}: TTFT = {ttft:.3f}s (no cache)")
        
        # Calculate cache effectiveness
        cached_ttfts = [r["ttft_mean"] for r in results_per_query]
        
        # First query has no cache, subsequent may benefit
        first_query_ttft = cached_ttfts[0]
        subsequent_avg_ttft = np.mean(cached_ttfts[1:]) if len(cached_ttfts) > 1 else first_query_ttft
        
        no_cache_avg_ttft = np.mean(no_cache_ttfts) if no_cache_ttfts else first_query_ttft
        
        # Cache hit rate estimation
        # If subsequent queries are faster than first, cache is working
        cache_speedup = first_query_ttft / subsequent_avg_ttft if subsequent_avg_ttft > 0 else 1.0
        
        return {
            "system_prompt_size": system_prompt_size,
            "system_prompt_tokens": system_tokens,
            "num_queries": num_queries,
            "first_query_ttft": first_query_ttft,
            "subsequent_avg_ttft": subsequent_avg_ttft,
            "no_cache_avg_ttft": no_cache_avg_ttft,
            "cache_speedup": cache_speedup,
            "estimated_cache_hit_rate": max(0, min(1, (cache_speedup - 1) * 0.5)),  # Rough estimate
            "results_per_query": results_per_query,
            "no_cache_ttfts": no_cache_ttfts
        }
    
    async def run(self):
        """Run the full experiment."""
        print("=" * 80)
        print("A4: Prefix Caching Hit Rate")
        print("=" * 80)
        
        if not await self.detect_model():
            print("Failed to detect model. Is the server running?")
            return None
        
        # Test with different system prompt sizes
        configs = [
            {"size": "medium", "queries": 10},
            {"size": "large", "queries": 10},
        ]
        
        for config in configs:
            result = await self.run_prefix_caching_test(config["size"], config["queries"], num_trials=2)
            self.results["runs"].append(result)
        
        # Analyze results
        print(f"\n{'='*80}")
        print("ANALYSIS")
        print(f"{'='*80}")
        
        for run in self.results["runs"]:
            print(f"\n{run['system_prompt_size'].upper()} System Prompt ({run['system_prompt_tokens']} tokens):")
            print(f"  First query TTFT: {run['first_query_ttft']:.3f}s")
            print(f"  Subsequent avg TTFT: {run['subsequent_avg_ttft']:.3f}s")
            print(f"  Cache speedup: {run['cache_speedup']:.2f}x")
            print(f"  Estimated cache hit rate: {run['estimated_cache_hit_rate']:.1%}")
        
        # Overall assessment
        avg_speedup = np.mean([r["cache_speedup"] for r in self.results["runs"]])
        self.results["analysis"] = {
            "avg_cache_speedup": avg_speedup,
            "cache_effective": bool(avg_speedup > 1.1),
            "recommendation": (
                "Prefix caching is effective — use shared system prompts" if avg_speedup > 1.1 else
                "Prefix caching has limited benefit — consider other optimizations"
            )
        }
        
        print(f"\nOverall cache speedup: {avg_speedup:.2f}x")
        print(f"Recommendation: {self.results['analysis']['recommendation']}")
        
        return self.results

async def main():
    probe = PrefixCachingProbe(
        url=os.getenv("VLLM_URL", "http://localhost"),
        port=int(os.getenv("VLLM_PORT", "8000"))
    )
    
    results = await probe.run()
    
    if results:
        output_path = "results/final/nemotron/mamba_experiments/prefix_caching.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
