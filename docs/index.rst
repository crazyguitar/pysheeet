.. python-cheatsheet documentation master file, created by
   sphinx-quickstart on Sun Feb 28 09:26:04 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
    :description lang=en: Comprehensive Python cheat sheet with practical code snippets, examples, and tutorials for Python developers. Learn Python basics, advanced topics, databases, networking, and more.
    :keywords: Python, Python Cheat Sheet, Python Tutorial, Python Examples, Python Code Snippets, Programming, Development, Python Reference, Python Guide


Python Cheat Sheet - Complete Guide with Code Examples
======================================================

Welcome to **pysheeet** - your ultimate Python cheat sheet! This comprehensive resource contains practical Python
code snippets, examples, and tutorials to make coding easier and more efficient for Python developers of all levels.
From basic Python syntax to advanced topics like databases, networking, and multitasking, this cheat sheet serves as your
complete Python reference guide. Ideal for beginners learning Python fundamentals and experienced developers seeking
quick code examples. Whether you're learning Python for web development, data science, automation, or general programming,
you'll find practical examples that save you time and improve your coding efficiency.

Contributions are always welcome—feel free to share ideas for new snippets, improvements, or clearer explanations!
If you'd like to contribute, `fork pysheeet on GitHub`_.
If there is any question or suggestion, please create an issue on `GitHub Issues`_.

.. _fork pysheeet on GitHub: https://github.com/crazyguitar/pysheeet
.. _GitHub Issues: https://github.com/crazyguitar/pysheeet/issues

Plugin
------

**pysheeet** is available as a `Claude Code <https://claude.com/claude-code>`_ plugin. Once installed,
Claude automatically uses the cheat sheets to answer Python questions.

.. code-block:: bash

    # Step 1: Add the marketplace
    claude plugin marketplace add crazyguitar/pysheeet

    # Step 2: Install the plugin
    claude plugin install pysheeet@pysheeet

For local testing and manual installation, see the main `README <https://github.com/crazyguitar/pysheeet/blob/master/README.rst>`_.

What's New In Python 3
----------------------

The official document, `What's New In Python`_, displays all of the most
important changes. However, if you're too busy to read the whole changes,
this part provides a brief glance of new features in Python 3.

.. _What's New In Python: https://docs.python.org/3/whatsnew/index.html

.. toctree::
   :maxdepth: 1

   notes/python-new-py3

Blog
----

This website also covers advanced topics related to LLM training, inference,
HPC networking, distributed systems, GPU computing, and Python debugging
techniques in the `blog </notes/appendix/index.html>`_.

.. toctree::
   :maxdepth: 1

   notes/appendix/index

Python Cheat Sheet
------------------

This section focuses on commonly used Python code snippets. The cheat sheet
covers not only core Python features but also essential data structures,
algorithms, and frequently used modules to help programmers efficiently tackle
everyday tasks.

.. toctree::
   :maxdepth: 1

   notes/basic/index
   notes/os/index
   notes/concurrency/index
   notes/asyncio/index
   notes/network/index
   notes/database/index
   notes/security/index
   notes/extension/index
   notes/llm/index
   notes/hpc/index
