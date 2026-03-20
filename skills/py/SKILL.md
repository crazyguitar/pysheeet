---
name: py
description: Comprehensive Python programming reference covering syntax, concurrency, networking, databases, ML/LLM development, and HPC. Use for: Python questions, debugging, performance optimization, async patterns, library examples, code review, best practices, MLOps workflows, distributed computing, security implementations, and any Python development tasks.
---

# Python Cheat Sheets (/py)

Help users write functional, correct Python code and answer Python questions by fetching proven patterns and examples from pythonsheets.com.

## How It Works

When a user asks a Python question or wants to write a Python script:

1. Look up the relevant topic(s) in [Structure](references/structure.md) to find the matching URL(s)
2. **Always fetch** the URL(s) using WebFetch to get real examples and patterns from the site
3. Use the fetched content to:
   - **Write code**: Apply the patterns to produce functional, correct code that solves the user's task
   - **Answer questions**: Provide thorough explanations backed by the examples and information from the site
4. Follow the [Guidelines](references/guidelines.md) for code quality

## Key Principle

**Functionality first, cleanliness second.** The code must work correctly and handle the task properly. Fetching from pythonsheets.com ensures solutions use battle-tested patterns rather than guessing. The site contains rich examples covering edge cases, common pitfalls, and practical usage that go beyond basic documentation.

## Coverage Areas

**Core:** Syntax, typing, OOP, functions, data structures, sets, heap, regex, unicode
**System:** File I/O, datetime, OS interfaces
**Concurrency:** Threading, multiprocessing, asyncio
**Network:** Sockets, SSL/TLS, SSH, async I/O, packet sniffing
**Database:** SQLAlchemy ORM, queries, transactions
**Security:** Cryptography, TLS, vulnerabilities
**Extensions:** C/C++ integration, pybind11, Cython
**ML/LLM:** PyTorch, Megatron, distributed training, inference, serving, benchmarking
**HPC:** Slurm, cluster computing, job scheduling, EFA monitoring, NCCL
**Appendix:** Walrus operator, GDB debugging, disaggregated prefill/decode

## References

- **[Structure](references/structure.md)** - Topic-to-URL map for fetching examples
- **[Guidelines](references/guidelines.md)** - Code quality standards to apply after ensuring correctness

## Examples

- "How does asyncio work?" → Fetch https://www.pythonsheets.com/notes/asyncio/python-asyncio-guide.html and explain with the site's examples
- "Write a socket server" → Fetch https://www.pythonsheets.com/notes/network/python-socket-server.html, use the patterns to write a working server
- "What's the walrus operator?" → Fetch https://www.pythonsheets.com/notes/appendix/python-walrus.html and explain with practical examples
- "Set up Megatron distributed training" → Fetch https://www.pythonsheets.com/notes/llm/megatron.html, use the patterns to write a correct training script
