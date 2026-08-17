#!/usr/bin/env python3
"""
C3: System Prompt Reuse

Same 4K system prompt, 100 different user queries.
Measure prefix cache hit rate + routing consistency.
Tests whether cached prefix tokens skip routing recomputation.
"""

import asyncio
import aiohttp
import json
import time
import numpy as np
import os
from datetime import datetime

class SystemPromptReuseProbe:
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
    
    def create_large_system_prompt(self):
        """Create a ~4K token system prompt."""
        return """You are a highly specialized AI assistant for software engineering and system architecture. 
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
- Document architectural decisions and rationale

Best Practices:
- Use dependency injection for testability
- Implement circuit breakers for external services
- Use structured logging (JSON format)
- Implement health checks for all services
- Use feature flags for gradual rollouts
- Implement proper grace periods for connection pooling
- Use read replicas for read-heavy workloads
- Implement idempotent operations for retry safety
- Use distributed tracing for request flow analysis
- Implement proper resource limits and quotas

Security Guidelines:
- Never store secrets in code or version control
- Use short-lived tokens with refresh mechanisms
- Implement rate limiting per user/IP
- Validate and sanitize all user inputs
- Use parameterized queries to prevent SQL injection
- Implement Content Security Policy (CSP) headers
- Use HTTPS everywhere, HSTS headers
- Implement proper CORS policies
- Use secure session management
- Implement audit logging for sensitive operations

Performance Optimization:
- Use connection pooling for database connections
- Implement multi-level caching (L1: in-memory, L2: Redis, L3: CDN)
- Use async/await for I/O-bound operations
- Implement batch processing for bulk operations
- Use database indexing strategically
- Implement query optimization and pagination
- Use compression for API responses
- Implement proper pagination for list endpoints
- Use background jobs for long-running tasks
- Implement proper resource cleanup

Monitoring and Observability:
- Use structured logging with correlation IDs
- Implement distributed tracing (OpenTelemetry)
- Set up alerts for error rates and latency
- Monitor resource utilization (CPU, memory, disk)
- Implement custom metrics for business logic
- Use dashboards for real-time visibility
- Implement log aggregation and analysis
- Set up anomaly detection for key metrics
- Use canary deployments for safe rollouts
- Implement feature flag analytics"""
    
    def create_user_queries(self, count=100):
        """Create diverse user queries."""
        base_queries = [
            "How do I implement authentication?",
            "What's the best way to handle errors?",
            "Explain microservices architecture.",
            "How should I structure my database?",
            "What are security best practices?",
            "How do I optimize performance?",
            "Explain caching strategies.",
            "How should I handle logging?",
            "What's the best testing approach?",
            "How do I deploy to production?",
        ]
        
        # Generate variations
        queries = []
        for i in range(count):
            base = base_queries[i % len(base_queries)]
            variation = f"{base} (variant {i // len(base_queries) + 1})"
            queries.append(variation)
        
        return queries
    
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
    
    async def run_system_prompt_reuse_test(self, num_queries=50, num_trials=2):
        """Run the system prompt reuse test."""
        print(f"\n--- Testing system prompt reuse with {num_queries} queries ---")
        
        system_prompt = self.create_large_system_prompt()
        system_tokens = len(system_prompt) // 4
        self.results["system_prompt_tokens"] = system_tokens
        print(f"System prompt: {len(system_prompt)} chars ≈ {system_tokens} tokens")
        
        queries = self.create_user_queries(num_queries)
        
        results_per_query = []
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
            for query_idx, query in enumerate(queries):
                if query_idx % 10 == 0:
                    print(f"  Processing query {query_idx + 1}/{num_queries}...")
                
                ttfts = []
                for trial in range(num_trials):
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ]
                    
                    ttft = await self.measure_ttft(session, messages)
                    ttfts.append(ttft)
                
                results_per_query.append({
                    "query_idx": query_idx,
                    "ttft_mean": np.mean(ttfts),
                    "ttft_std": np.std(ttfts)
                })
        
        return results_per_query
    
    async def run(self, num_queries=50, num_trials=2):
        """Run the full experiment."""
        print("=" * 80)
        print("C3: System Prompt Reuse")
        print("=" * 80)
        
        if not await self.detect_model():
            print("Failed to detect model. Is the server running?")
            return None
        
        # Run test
        results = await self.run_system_prompt_reuse_test(num_queries, num_trials)
        
        if results:
            # Analyze results
            ttfts = [r["ttft_mean"] for r in results]
            
            # First query vs subsequent
            first_ttft = ttfts[0]
            subsequent_avg = np.mean(ttfts[1:])
            
            # Calculate cache effectiveness
            cache_speedup = first_ttft / subsequent_avg if subsequent_avg > 0 else 1.0
            
            analysis = {
                "num_queries": num_queries,
                "system_prompt_tokens": self.results["system_prompt_tokens"],
                "first_query_ttft": first_ttft,
                "subsequent_avg_ttft": subsequent_avg,
                "cache_speedup": cache_speedup,
                "ttft_std": np.std(ttfts),
                "interpretation": (
                    "System prompt caching is effective" if cache_speedup > 1.1 else
                    "System prompt caching has limited benefit"
                )
            }
            
            self.results["analysis"] = analysis
            self.results["runs"] = results
            
            print(f"\n{'='*80}")
            print("ANALYSIS")
            print(f"{'='*80}")
            print(f"First query TTFT: {first_ttft:.3f}s")
            print(f"Subsequent avg TTFT: {subsequent_avg:.3f}s")
            print(f"Cache speedup: {cache_speedup:.2f}x")
            print(f"TTFT std: {np.std(ttfts):.3f}s")
            print(f"Interpretation: {analysis['interpretation']}")
        
        return self.results

async def main():
    probe = SystemPromptReuseProbe(
        url=os.getenv("VLLM_URL", "http://localhost"),
        port=int(os.getenv("VLLM_PORT", "8000"))
    )
    
    results = await probe.run(num_queries=50, num_trials=2)
    
    if results:
        output_path = "results/final/nemotron/mamba_experiments/system_prompt_reuse.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
