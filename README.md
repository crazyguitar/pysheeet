
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

Plugin
======

**pysheeet** is available as a Claude Code plugin. Once installed, Claude
automatically uses the cheat sheets to answer Python questions — just ask
naturally and the skill triggers based on context.

Installation
------------

**As a Claude Code plugin (recommended):**

.. code-block:: bash

    # Step 1: Add the marketplace
    claude plugin marketplace add crazyguitar/pysheeet

    # Step 2: Install the plugin
    claude plugin install pysheeet@pysheeet

**Local testing (single session only):**

.. code-block:: bash

    claude --plugin-dir /path/to/pysheeet

**Manual installation (requires cloning the repo):**

.. code-block:: bash

    git clone https://github.com/crazyguitar/pysheeet.git
    mkdir -p ~/.claude/skills
    cp -r pysheeet/skills/py ~/.claude/skills/py

Python Interview Cheatsheet
===========================

Curated Python interview questions indexed by topic — each question links
directly to the section of the cheat sheet that answers it. Use it for quick
review before an interview, or to drill down on a specific area (GIL, asyncio,
decorators, MRO, and more).

- `Python Interview Cheatsheet <docs/notes/interview/index.rst>`_

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
- `Megatron <docs/notes/llm/megatron.rst>`_ - NVIDIA Megatron training/fine-tuning framework with enroot/pyxis
- `LLM Serving <docs/notes/llm/llm-serving.rst>`_ - vLLM and SGLang for production inference with TP/PP/DP/EP
- `LLM Benchmark <docs/notes/llm/llm-bench.rst>`_ - Benchmark suite for measuring serving performance


HPC
===

High-Performance Computing tools for cluster management and job scheduling.
Covers Slurm workload manager and Ray for distributed computing on GPU clusters.

- `Slurm <docs/notes/hpc/slurm.rst>`_
- `Ray Cluster <docs/notes/hpc/ray.rst>`_


Blog
====

Supplementary topics covering Python internals, debugging techniques, and
language features that don't fit elsewhere.

- `NVSHMEM Multi-NIC Support with AWS EFA <docs/notes/appendix/nvshmem-multi-nic.rst>`_
- `Is Disaggregated Prefill/Decode a Silver Bullet for LLM Serving? <docs/notes/appendix/disaggregated-prefill-decode.rst>`_
- `Monitoring EFA with NCCL GIN and Nsys <docs/notes/appendix/megatron-efa-monitoring.rst>`_
- `GPU-Initiated Networking for NCCL on AWS <docs/notes/appendix/nccl-gin.rst>`_
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


## 🌐 Web Resources & Aesthetic Symbols Index
- [SYM 1D47D](https://scholarly-type-fonts-40.pages.dev/symbol/sym-1d47d/)
- [SYM 268A](https://neon-hacker-text-25.pages.dev/symbol/sym-268a/)
- [ROBLOX NAMES](https://theeduplaycampen.pages.dev/ru/roblox-names/)
- [SYM 1D49B](https://vintage-angel-symbols-66.pages.dev/symbol/sym-1d49b/)
- [SYM 26A6](https://futuristic-gaming-fonts-52.pages.dev/symbol/sym-26a6/)
- [SYM 1D456](https://anime-sparkle-text-22.pages.dev/symbol/sym-1d456/)
- [SYM 265D](https://sleek-bio-fonts-25.pages.dev/symbol/sym-265d/)
- [SYM 2764 FE0F](https://clean-line-emojis-77.pages.dev/symbol/sym-2764-fe0f/)
- [HEARTS](https://kawaii-kaomoji-hub-96.pages.dev/hearts/)
- [FREEFIRE NAMES](https://scholarly-cross-symbols-35.pages.dev/es/freefire-names/)
- [SYM 1D49D](https://vintage-library-rune-80.pages.dev/symbol/sym-1d49d/)
- [SYM 26FF](https://dark-poetry-fonts-30.pages.dev/symbol/sym-26ff/)
- [ARROWS LINES](https://vintage-library-rune-80.pages.dev/vi/arrows-lines/)
- [SYM 1F61D](https://pastel-chibi-emotes-23.pages.dev/symbol/sym-1f61d/)
- [SYM 1D42B](https://minimal-star-symbols-32.pages.dev/symbol/sym-1d42b/)
- [COQUETTE BOW RIBBON](https://pastel-chibi-emotes-23.pages.dev/symbol/coquette-bow-ribbon/)
- [SYM 1F62B](https://gothic-bio-fonts-13.pages.dev/symbol/sym-1f62b/)
- [SYM 26E9](https://manga-emotion-symbols-69.pages.dev/symbol/sym-26e9/)
- [GAMING WEAPONS](https://sleek-dot-symbols-31.pages.dev/vi/gaming-weapons/)
- [SYM 1F917](https://gothic-bio-fonts-13.pages.dev/symbol/sym-1f917/)
- [MUSIC WEATHER](https://pastel-chibi-emotes-23.pages.dev/vi/music-weather/)
- [EIGHT POINTED STAR](https://pastel-chibi-emotes-23.pages.dev/symbol/eight-pointed-star/)
- [SYM 1F499](https://vintage-library-rune-80.pages.dev/symbol/sym-1f499/)
- [SYM 2678](https://anime-sparkle-text-58.pages.dev/symbol/sym-2678/)
- [SYM 260D](https://ribbon-bow-unicode-18.pages.dev/symbol/sym-260d/)
- [SYM 1D435](https://nordic-minimal-fonts-67.pages.dev/symbol/sym-1d435/)
- [ARROWS LINES](https://cyber-clan-tags-38.pages.dev/es/arrows-lines/)
- [SYM 2631](https://cyber-clan-tags-75.pages.dev/symbol/sym-2631/)
- [SYM 1F60E](https://clean-mono-fonts-64.pages.dev/symbol/sym-1f60e/)
- [ARROWS LINES](https://zen-spacing-text-68.pages.dev/vi/arrows-lines/)
- [SYM 1F63D](https://vintage-library-rune-80.pages.dev/symbol/sym-1f63d/)
- [SYM 1D4A0](https://minimal-star-symbols-26.pages.dev/symbol/sym-1d4a0/)
- [LEFT BLACK LENTICULAR BRACKET](https://pastel-chibi-emotes-23.pages.dev/symbol/left-black-lenticular-bracket/)
- [SEA STARFISH OCEAN](https://manga-emotion-symbols-69.pages.dev/symbol/sea-starfish-ocean/)
- [HIGH VOLTAGE LIGHTNING](https://balletcore-unicode-67.pages.dev/symbol/high-voltage-lightning/)
- [SYM 26B5](https://clean-mono-fonts-64.pages.dev/symbol/sym-26b5/)
- [CROSSED SWORDS](https://balletcore-unicode-67.pages.dev/symbol/crossed-swords/)
- [SYM 263B](https://anime-sparkle-text-58.pages.dev/symbol/sym-263b/)
- [SYM 260C](https://cyber-clan-tags-38.pages.dev/symbol/sym-260c/)
- [SYM 2681](https://clean-mono-fonts-64.pages.dev/symbol/sym-2681/)
- [SYM 1D404](https://angelic-bow-symbols-76.pages.dev/symbol/sym-1d404/)
- [SYM 26C1](https://matrix-glitch-text-59.pages.dev/symbol/sym-26c1/)
- [HEAVY STAR](https://balletcore-unicode-67.pages.dev/symbol/heavy-star/)
- [SAGITTARIUS ZODIAC ARCHER](https://zen-spacing-text-68.pages.dev/symbol/sagittarius-zodiac-archer/)
- [DISCORD STATUS](https://clean-aesthetic-fonts-73.pages.dev/ja/discord-status/)
- [SYM 1F614](https://matrix-glitch-text-59.pages.dev/symbol/sym-1f614/)
- [BRACKETS](https://balletcore-unicode-67.pages.dev/ru/brackets/)
- [DAGGER BLADE](https://balletcore-unicode-67.pages.dev/symbol/dagger-blade/)
- [SYM 1F49F](https://soft-pink-fonts-41.pages.dev/symbol/sym-1f49f/)
- [SYM 2728](https://zen-aesthetic-fonts-87.pages.dev/symbol/sym-2728/)
- [SYM 267A](https://anime-sparkle-text-58.pages.dev/symbol/sym-267a/)
- [SEA STARFISH OCEAN](https://zen-spacing-text-68.pages.dev/symbol/sea-starfish-ocean/)
- [LATIN CROSS FAITH](https://balletcore-unicode-67.pages.dev/symbol/latin-cross-faith/)
- [RIGHTWARDS PAIRED HARPOON](https://matrix-glitch-text-59.pages.dev/symbol/rightwards-paired-harpoon/)
- [KAOMOJI](https://gothic-bio-fonts-69.pages.dev/ja/kaomoji/)
- [SYM 26CC](https://vintage-library-rune-80.pages.dev/symbol/sym-26cc/)
- [SYM 1F49F](https://baroque-crown-unicode-60.pages.dev/symbol/sym-1f49f/)
- [SYM 2636](https://clean-mono-fonts-64.pages.dev/symbol/sym-2636/)
- [SYM 2668](https://minimal-star-symbols-25.pages.dev/symbol/sym-2668/)
- [SYM 1D442](https://matrix-glitch-text-59.pages.dev/symbol/sym-1d442/)
- [SYM 1D461](https://anime-sparkle-text-58.pages.dev/symbol/sym-1d461/)
- [HIGH VOLTAGE LIGHTNING](https://scholarly-cross-symbols-35.pages.dev/symbol/high-voltage-lightning/)
- [FREEFIRE NAMES](https://baroque-unicode-decor-43.pages.dev/freefire-names/)
- [SYM 267D](https://witchy-runic-text-71.pages.dev/symbol/sym-267d/)
- [SYM 1D490](https://angelic-ribbon-text-78.pages.dev/symbol/sym-1d490/)
- [SYM 263A](https://anime-sparkle-text-58.pages.dev/symbol/sym-263a/)
- [SYM 1F49A](https://anime-sparkle-text-81.pages.dev/symbol/sym-1f49a/)
- [SYM 1D442](https://angelic-bow-symbols-76.pages.dev/symbol/sym-1d442/)
- [COQUETTE BOW RIBBON](https://dark-literary-kaomoji-13.pages.dev/symbol/coquette-bow-ribbon/)
- [SYM 1F603](https://mecha-crosshair-tags-20.pages.dev/symbol/sym-1f603/)
- [SYM 26A2](https://baroque-crown-unicode-60.pages.dev/symbol/sym-26a2/)
- [SYM 267E](https://anime-sparkle-text-22.pages.dev/symbol/sym-267e/)
- [SYM 1FAE3](https://sleek-line-unicode-29.pages.dev/symbol/sym-1fae3/)
- [SYM 2616](https://clean-sparkle-text-75.pages.dev/symbol/sym-2616/)
- [SYM 2641](https://neon-gamer-symbols-64.pages.dev/symbol/sym-2641/)
- [SYM 1D411](https://baroque-unicode-decor-43.pages.dev/symbol/sym-1d411/)
- [SYM 1FAE8](https://gothic-bio-fonts-13.pages.dev/symbol/sym-1fae8/)
- [SYM 26D6](https://kawaii-kaomoji-hub-77.pages.dev/symbol/sym-26d6/)
- [STARS](https://pastel-chibi-emotes-23.pages.dev/ru/stars/)
- [SYM 1D490](https://matrix-glitch-text-59.pages.dev/symbol/sym-1d490/)
- [CIRCLED STAR](https://scholarly-runes-text-68.pages.dev/symbol/circled-star/)
- [SYM 1F607](https://kawaii-kaomoji-hub-77.pages.dev/symbol/sym-1f607/)
- [SYM 26E5](https://classic-literature-symbols-64.pages.dev/symbol/sym-26e5/)
- [SYM 1F48C](https://zen-spacing-text-68.pages.dev/symbol/sym-1f48c/)
- [SYM 268A](https://scholarly-cross-symbols-35.pages.dev/symbol/sym-268a/)
- [SYM 1D47F](https://anime-sparkle-text-58.pages.dev/symbol/sym-1d47f/)
- [SYM 1D40D](https://sleek-line-unicode-29.pages.dev/symbol/sym-1d40d/)
- [LIBRA ZODIAC SCALES](https://zen-arrow-symbols-99.pages.dev/symbol/libra-zodiac-scales/)
- [SYM 26AB](https://coquette-aesthetic-symbols-84.pages.dev/symbol/sym-26ab/)
- [BRACKETS](https://chibi-emoticon-lab-65.pages.dev/es/brackets/)
- [BEAMED EIGHTH NOTES](https://neon-gamer-symbols-64.pages.dev/symbol/beamed-eighth-notes/)
- [SYM 26A8](https://clean-mono-fonts-64.pages.dev/symbol/sym-26a8/)
- [INSTAGRAM BIO](https://raven-gothic-kaomoji-25.pages.dev/ru/instagram-bio/)
- [SYM 1D455](https://academic-rune-text-25.pages.dev/symbol/sym-1d455/)
- [SYM 1D499](https://neon-glitch-fonts-20.pages.dev/symbol/sym-1d499/)
- [SYM 1F497](https://zen-arrow-symbols-99.pages.dev/symbol/sym-1f497/)
- [SYM 268F](https://gothic-bio-fonts-13.pages.dev/symbol/sym-268f/)
- [HEARTS](https://dark-literary-kaomoji-13.pages.dev/pt/hearts/)
- [SYM 1F63F](https://vintage-library-rune-80.pages.dev/symbol/sym-1f63f/)
- [SYM 1F62E 200D 1F4A8](https://chibi-emoticon-lab-65.pages.dev/symbol/sym-1f62e-200d-1f4a8/)
- [SYM 26A6](https://neon-glitch-fonts-20.pages.dev/symbol/sym-26a6/)
- [ROBLOX NAMES](https://neon-gamer-symbols-64.pages.dev/ru/roblox-names/)
- [SYM 2731](https://chibi-emoticon-lab-65.pages.dev/symbol/sym-2731/)
- [SYM 1D406](https://gothic-bio-fonts-69.pages.dev/symbol/sym-1d406/)
- [SINGLE EIGHTH MUSICAL NOTE](https://coquette-aesthetic-symbols-51.pages.dev/symbol/single-eighth-musical-note/)
- [SYM 1D410](https://sleek-bio-fonts-25.pages.dev/symbol/sym-1d410/)
- [SYM 1F929](https://baroque-unicode-decor-43.pages.dev/symbol/sym-1f929/)
- [MANGA EMOTION SYMBOLS 69.PAGES.DEV](https://manga-emotion-symbols-69.pages.dev/)
- [SYM 2614](https://pastel-manga-symbols-57.pages.dev/symbol/sym-2614/)
- [SYM 26FD](https://angelic-bow-symbols-76.pages.dev/symbol/sym-26fd/)
- [SYM 1D485](https://modern-bullet-symbols-45.pages.dev/symbol/sym-1d485/)
- [SYM 26AE](https://nordic-minimal-fonts-67.pages.dev/symbol/sym-26ae/)
- [NATURE FLOWERS](https://kawaii-kaomoji-hub-88.pages.dev/ru/nature-flowers/)
- [SYM 2636](https://minimal-star-symbols-54.pages.dev/symbol/sym-2636/)
- [SYM 2611](https://anime-sparkle-text-58.pages.dev/symbol/sym-2611/)
- [SYM 2748](https://pastel-manga-symbols-57.pages.dev/symbol/sym-2748/)
- [SYM 26A7](https://clean-mono-fonts-64.pages.dev/symbol/sym-26a7/)
- [TIBETAN LOTUS BLOSSOM](https://chibi-emoticon-world-87.pages.dev/symbol/tibetan-lotus-blossom/)
- [SYM 26BA](https://cyber-clan-tags-38.pages.dev/symbol/sym-26ba/)
- [SYM 1D431](https://minimal-star-symbols-54.pages.dev/symbol/sym-1d431/)
- [SYM 1D45D](https://gothic-bio-fonts-81.pages.dev/symbol/sym-1d45d/)
- [SYM 2657](https://scholarly-vintage-symbols-48.pages.dev/symbol/sym-2657/)
- [SYM 1F62D](https://minimal-star-symbols-54.pages.dev/symbol/sym-1f62d/)
- [SYM 26AE](https://mecha-synth-kaomoji-92.pages.dev/symbol/sym-26ae/)
- [SPARKLE DOT FLARE](https://dark-literary-kaomoji-13.pages.dev/symbol/sparkle-dot-flare/)
- [GEMINI ZODIAC TWINS](https://dark-literary-kaomoji-13.pages.dev/symbol/gemini-zodiac-twins/)
- [QUARTER MUSICAL NOTE](https://angelic-ribbon-text-78.pages.dev/symbol/quarter-musical-note/)
- [SYM 1D421](https://dolly-angel-fonts-14.pages.dev/symbol/sym-1d421/)
- [SCORPIO ZODIAC SCORPION](https://zen-arrow-symbols-99.pages.dev/symbol/scorpio-zodiac-scorpion/)
- [FREEFIRE NAMES](https://neon-glitch-fonts-20.pages.dev/freefire-names/)
