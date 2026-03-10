.. meta::
    :description lang=en: Design philosophy of pep 572, the walrus operator
    :keywords: Python3, PEP 572, walrus operator


PEP 572 and The Walrus Operator
===============================

:Date: 2025-08-30

Abstract
--------

`PEP 572`_ is one of the most contentious proposals in Python3 history because
assigning a value within an expression seems unnecessary. Also, it is ambiguous
for developers to distinguish the difference between **the walrus operator**
(``:=``) and the equal operator (``=``). Even though sophisticated developers
can use "``:=``" smoothly, they may concern the readability of their code. To
better understand the usage of "``:=``," this article discusses its design
philosophy and what kind of problems it tries to solve.


Introduction
------------

For C/C++ developer, assigning a function return to a variable is common due
to error code style handling. Managing function errors includes two steps;
one is to check the return value; another is to check ``errno``. For example,

.. code-block:: cpp

    #include <stdio.h>
    #include <unistd.h>
    #include <string.h>
    #include <errno.h>

    int main(int argc, char *argv[]) {
        int rc = -1;

        // assign access return to rc and check its value
        if ((rc = access("hello_walrus", R_OK)) == -1) {
            fprintf(stderr, "%s", strerror(errno));
            goto end;
        }
        rc = 0;
    end:
        return rc;
    }

In this case, ``access`` will assign its return value to the variable ``rc``
first. Then, the program will compare the ``rc`` value with ``-1`` to check
whether the execution of ``access`` is successful or not. However, Python did
not allow assigning values to variables within an expression before 3.8. To fix
this problem, therefore, PEP 572 introduced the walrus operator for developers.
The following Python snippet is equal to the previous C example.

.. code-block:: python

    >>> import os
    >>> from ctypes import *
    >>> libc = CDLL("libc.dylib", use_errno=True)
    >>> access = libc.access
    >>> path = create_string_buffer(b"hello_walrus")
    >>> if (rc := access(path, os.R_OK)) == -1:
    ...     errno = get_errno()
    ...     print(os.strerror(errno), file=sys.stderr)
    ...
    No such file or directory


Why ``:=`` ?
------------

Developers may confuse the difference between "``:=``" and  "``=``." In fact, they
serve the same purpose, assigning somethings to variables. Why Python introduced
"``:=``" instead of using "``=``"? What is the benefit of using "``:=``"? One
reason is to reinforce the visual recognition due to a common mistake made by
C/C++ developers. For instance,

.. code-block:: cpp

    int rc = access("hello_walrus", R_OK);

    // rc is unintentionally assigned to -1
    if (rc = -1) {
        fprintf(stderr, "%s", strerror(errno));
        goto end;
    }

Rather than comparison, the variable, ``rc``, is mistakenly assigned to -1. To
prevent this error, some people advocate using `Yoda conditions`_ within an
expression.

.. code-block:: cpp

    int rc = access("hello_walrus", R_OK);

    // -1 = rc will raise a compile error
    if (-1 == rc) {
        fprintf(stderr, "%s", strerror(errno));
        goto end;
    }

However, Yoda style is not readable enough like Yoda speaks non-standardized
English. Also, unlike C/C++ can detect assigning error during the compile-time
via compiler options (e.g., -Wparentheses), it is difficult for Python interpreter
to distinguish such mistakes throughout the runtime. Thus, the final result
of PEP 572 was to use a new syntax as a solution to implement *assignment
expressions*.

The walrus operator was not the first solution for PEP 572. The original proposal
used ``EXPR as NAME`` to assign values to variables. Unfortunately, there are
some rejected reasons in this solution and other solutions as well. After
intense debates, the final decision was ``:=``.

Scopes
------

Unlike other expressions, which a variable is bound to a scope, an assignment
expression belongs to the current scope. The purpose of this design is to
allow a compact way to write code.

.. code-block:: python3

    >>> if not (env := os.environ.get("HOME")):
    ...     raise KeyError("env HOME does not find!")
    ...
    >>> print(env)
    /root

In PEP 572, another benefit is to conveniently capture a "witness" for an
``any()`` or an ``all()`` expression. Although capturing function inputs can
assist an interactive debugger, the advantage is not so obvious, and examples
lack readability. Therefore, this benefit does not discuss here. Note that
other languages (e.g., C/C++ or Go) may bind an assignment to a scope. Take
Golang as an example.

.. code-block:: go

    package main

    import (
        "fmt"
        "os"
    )

    func main() {
        if env := os.Getenv("HOME"); env == "" {
            panic(fmt.Sprintf("Home does not find"))
        }
        fmt.Print(env) // <--- compile error: undefined: env
    }

Pitfalls
--------

Although an assigning expression allows writing compact code, there are many
pitfalls when a developer uses it in a list comprehension. A common ``SyntaxError``
is to rebind iteration variables.

.. code-block:: python3

    >>> [i := i+1 for i in range(5)]  # invalid

However, updating an iteration variable will reduce readability and introduce
bugs. Even if Python 3.8 did not implement the walrus operator, a programmer
should avoid reusing iteration variables within a scope.

Another pitfall is Python prohibits using assignment expressions within a
comprehension under a class scope.

.. code-block:: python3

    >>> class Example:
    ...     [(j := i) for i in range(5)] # invalid
    ...

This limitation was from `bpo-3692`_. The interpreter's behavior is
unpredictable when a class declaration contains a list comprehension. To avoid
this corner case, assigning expression is invalid under a class.

.. code-block:: python3

    >>> class Foo:
    ...     a = [1, 2, 3]
    ...     b = [4, 5, 6]
    ...     c = [i for i in zip(a, b)]  # b is defined
    ...
    >>> class Bar:
    ...     a = [1,2,3]
    ...     b = [4,5,6]
    ...     c = [x * y for x in a for y in b] # b is undefined
    ...
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "<stdin>", line 4, in Bar
      File "<stdin>", line 4, in <listcomp>
    NameError: name 'b' is not defined

Conclusion
----------

The reason why the walrus operator (``:=``) is so controversial is that code
readability may decrease. In fact, in the discussion `mail thread <https://mail.python.org/pipermail/python-ideas/2018-March/049409.html>`_,
the author of PEP 572, Christoph Groth, had considered using "``=``" to implement
inline assignment like C/C++. Without judging "``:=``" is ugly, many developers
argue that distinguishing the functionality between "``:=``" and "``=``" is
difficult because they serve the same purpose, but behaviors are not consistent.
Also, writing compact code is not persuasive enough because smaller is not
always better. However, in some cases, the walrus operator can enhance
readability (if you understand how to use ``:=``). For example,

.. code-block:: python3

    buf = b""
    while True:
        data = read(1024)
        if not data:
            break
        buf += data

By using ``:=``, the previous example can be simplified.

.. code-block:: python3

    buf = b""
    while (data := read(1024)):
        buf += data

`Python document`_ and GitHub `issue-8122`_ provides many great examples about
improving code readability by "``:=``". However, using the walrus operator
should be careful. Some cases, such as ``foo(x := 3, cat='vector')``, may
introduce new bugs if developers are not aware of scopes. Although PEP 572
may be risky for developers to write buggy code, an in-depth understanding of
design philosophy and useful examples will help us use it to write readable
code at the right time.

References
----------

1. `PEP 572 - Assignment Expressions`_
2. `What’s New In Python 3.8`_
3. `PEP 572 and decision-making in Python`_
4. `The PEP 572 endgame`_
5. `Use assignment expression in stdlib (combined PR)`_
6. `Improper scope in list comprehension, when used in class declaration`_

.. _PEP 572: https://www.python.org/dev/peps/pep-0572/
.. _PEP 572 - Assignment Expressions: https://www.python.org/dev/peps/pep-0572/
.. _What’s New In Python 3.8: https://docs.python.org/3/whatsnew/3.8.html
.. _PEP 572 and decision-making in Python: https://lwn.net/Articles/757713/
.. _The PEP 572 endgame: https://lwn.net/Articles/759558/
.. _Use assignment expression in stdlib (combined PR): https://github.com/python/cpython/pull/8122/files
.. _improper scope in list comprehension, when used in class declaration: https://bugs.python.org/issue3692
.. _Yoda conditions: https://en.wikipedia.org/wiki/Yoda_conditions
.. _bpo-3692: https://bugs.python.org/issue3692
.. _Python document: https://docs.python.org/3/whatsnew/3.8.html#assignment-expressions
.. _issue-8122: https://github.com/python/cpython/pull/8122/files
