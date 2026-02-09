
.. raw:: html

    <h1 align="center">
    <br>
      <a href="https://www.pythonsheets.com"><img src="docs/_static/logo.png" alt="pysheeet" width=200"></a>
    </h1>
    <p align="center">
      <a href="https://github.com/crazyguitar/pysheeet/actions">
        <img src="https://github.com/crazyguitar/pysheeet/actions/workflows/pythonpackage.yml/badge.svg" alt="Build Status">
      </a>
      <a href="https://coveralls.io/github/crazyguitar/pysheeet?branch=master">
        <img src="https://coveralls.io/repos/github/crazyguitar/pysheeet/badge.svg?branch=master" alt="Coverage">
      </a>
      <a href="https://raw.githubusercontent.com/crazyguitar/pysheeet/master/LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License MIT">
      </a>
      <a href="https://doi.org/10.5281/zenodo.15529042">
        <img src="https://zenodo.org/badge/52760178.svg" alt="DOI">
      </a>
    </p>

Introduction
=============

This project was started to bring together useful Python code snippets that make
coding faster, easier, and more enjoyable. You can explore all the cheat sheets at
`Pysheeet <https://www.pythonsheets.com/>`_. Contributions are always welcome—feel
free to fork the repo and submit a pull request to help it grow!

/py Skill
=========

**pysheeet** includes Claude Skills integration for intelligent access to Python cheat sheets directly within Claude Code and Claude.ai.

Installation
------------

.. code-block:: bash

    # Install globally for Claude Code
    mkdir -p ~/.claude/skills
    cp -r skills ~/.claude/skills/py

Usage
-----

Once installed, simply ask Claude any Python question and the ``/py`` skill will automatically provide relevant examples and documentation from this cheat sheet collection.

Examples:

- "How do I create async HTTP clients?"
- "Show me SQLAlchemy query patterns"
- "Best practices for Python threading"
- "Benchmark LLM serving with Slurm"


What's New In Python 3
======================

This part only provides a quick glance at some important features in Python 3.
If you're interested in all of the most important features, please read the
official document, `What’s New in Python <https://docs.python.org/3/whatsnew/index.html>`_.

- `New in Python3 <docs/notes/python-new-py3.rst>`_


Cheat Sheet
===========

Core Python fundamentals including data types, functions, classes, and commonly
used patterns for everyday programming tasks.

- `From Scratch <docs/notes/basic/python-basic.rst>`_
- `Future <docs/notes/basic/python-future.rst>`_
- `Typing <docs/notes/basic/python-typing.rst>`_
- `Class <docs/notes/basic/python-object.rst>`_
- `Function <docs/notes/basic/python-func.rst>`_
- `Unicode <docs/notes/basic/python-unicode.rst>`_
- `List <docs/notes/basic/python-list.rst>`_
- `Set <docs/notes/basic/python-set.rst>`_
- `Dictionary <docs/notes/basic/python-dict.rst>`_
- `Heap <docs/notes/basic/python-heap.rst>`_
- `Generator <docs/notes/basic/python-generator.rst>`_
- `Regular expression <docs/notes/basic/python-rexp.rst>`_


System
======

Date/time handling, file I/O, and operating system interfaces.

- `Datetime <docs/notes/os/python-date.rst>`_ - Timestamps, formatting, parsing, timezones, timedelta
- `Files and I/O <docs/notes/os/python-io.rst>`_ - Reading, writing, pathlib, shutil, tempfile
- `Operating System <docs/notes/os/python-os.rst>`_ - Processes, environment, system calls


Concurrency
===========

Threading, multiprocessing, and concurrent.futures for parallel execution.
Covers synchronization primitives, process pools, and bypassing the GIL.

- `Threading <docs/notes/concurrency/python-threading.rst>`_ - Threads, locks, semaphores, events, conditions
- `Multiprocessing <docs/notes/concurrency/python-multiprocessing.rst>`_ - Processes, pools, shared memory, IPC
- `concurrent.futures <docs/notes/concurrency/python-futures.rst>`_ - Executors, futures, callbacks


Asyncio
=======

Asynchronous programming with Python's ``asyncio`` module. Covers coroutines,
event loops, tasks, networking, and advanced patterns.

- `A Hitchhiker's Guide to Asynchronous Programming <docs/notes/asyncio/python-asyncio-guide.rst>`_ - Design philosophy and evolution
- `Asyncio Basics <docs/notes/asyncio/python-asyncio-basic.rst>`_ - Coroutines, tasks, gather, timeouts
- `Asyncio Networking <docs/notes/asyncio/python-asyncio-server.rst>`_ - TCP/UDP servers, HTTP, SSL/TLS
- `Asyncio Advanced <docs/notes/asyncio/python-asyncio-advanced.rst>`_ - Synchronization, queues, subprocesses


C/C++ Extensions
================

Native extensions for performance-critical code. Covers modern pybind11 (used by
PyTorch, TensorFlow), ctypes, cffi, Cython, and the traditional Python C API.
Also includes a guide for Python developers learning modern C++ syntax.

- `ctypes <docs/notes/extension/python-ctypes.rst>`_ - Load shared libraries without compilation
- `Python C API <docs/notes/extension/python-capi.rst>`_ - Traditional C extension reference
- `Modern C/C++ Extensions <docs/notes/extension/python-cext-modern.rst>`_ - pybind11, Cython
- `Learn C++ from Python <docs/notes/extension/cpp-from-python.rst>`_ - Modern C++ for Python developers


Security
========

Modern cryptographic practices and common security vulnerabilities. Covers
encryption, TLS/SSL, and why legacy patterns are dangerous.

- `Modern Cryptography <docs/notes/security/python-crypto.rst>`_ - AES-GCM, RSA-OAEP, Ed25519, Argon2
- `TLS/SSL and Certificates <docs/notes/security/python-tls.rst>`_ - HTTPS servers, certificate generation
- `Common Vulnerabilities <docs/notes/security/python-vulnerability.rst>`_ - Padding oracle, injection, timing attacks


Network
=======

Low-level network programming with Python sockets. Covers TCP/UDP communication,
server implementations, asynchronous I/O, SSL/TLS encryption, and packet analysis.

- `Socket Basics <docs/notes/network/python-socket.rst>`_
- `Socket Servers <docs/notes/network/python-socket-server.rst>`_
- `Async Socket I/O <docs/notes/network/python-socket-async.rst>`_
- `SSL/TLS Sockets <docs/notes/network/python-socket-ssl.rst>`_
- `Packet Sniffing <docs/notes/network/python-socket-sniffer.rst>`_
- `SSH and Tunnels <docs/notes/network/python-ssh.rst>`_


Database
========

Database access with SQLAlchemy, Python's most popular ORM. Covers connection
management, raw SQL, object-relational mapping, and common query patterns.

- `SQLAlchemy Basics <docs/notes/database/python-sqlalchemy.rst>`_
- `SQLAlchemy ORM <docs/notes/database/python-sqlalchemy-orm.rst>`_
- `SQLAlchemy Query Recipes <docs/notes/database/python-sqlalchemy-query.rst>`_


LLM
===

Large Language Models (LLM) training, inference, and optimization. Covers PyTorch
for model development, distributed training across GPUs, and vLLM/SGLang for
high-performance LLM inference and serving.

- `PyTorch <docs/notes/llm/pytorch.rst>`_ - Tensors, autograd, neural networks, training loops
- `Distributed Training <docs/notes/llm/distributed.rst>`_ - Multi-GPU training, DDP, FSDP, DeepSpeed
- `LLM Serving <docs/notes/llm/llm-serving.rst>`_ - vLLM and SGLang for production inference with TP/PP/DP/EP
- `LLM Benchmark <docs/notes/llm/llm-bench.rst>`_ - Benchmark suite for measuring serving performance


HPC
===

High-Performance Computing tools for cluster management and job scheduling.
Covers Slurm workload manager for distributed computing and GPU clusters.

- `Slurm <docs/notes/hpc/slurm.rst>`_


Appendix
=========

Supplementary topics covering Python internals, debugging techniques, and
language features that don't fit elsewhere.

- `PEP 572 and the walrus operator <docs/notes/appendix/python-walrus.rst>`_
- `Python Interpreter in GNU Debugger <docs/notes/appendix/python-gdb.rst>`_

PDF Version
============

`pdf`_

.. _pdf: https://media.readthedocs.org/pdf/pysheeet/latest/pysheeet.pdf

How to run the server
=======================

.. code-block:: bash

    $ virtualenv venv
    $ . venv/bin/activate
    $ pip install -r requirements.txt
    $ make
    $ python app.py

    # URL: localhost:5000
