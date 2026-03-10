.. meta::
    :description lang=en: Python interpreter in GNU Debugger (GDB)
    :keywords: Python, Python3, GDB

==================================
Python Interpreter in GNU Debugger
==================================

:Date: 2025-08-30

Abstract
--------

The GNU Debugger (GDB) is the most powerful debugging tool for developers to
troubleshoot errors in their code. However, it is hard for beginners to learn,
and that is why many programmers prefer to insert ``print`` to examine runtime
status. Fortunately, `GDB Text User Interface (TUI)`_ provides a way for
developers to review their source code and debug simultaneously. More
excitingly, In GDB 7, **Python Interpreter** was built into GDB. This feature
offers more straightforward ways to customize GDB printers and commands through
the Python library. By discussing examples, this article tries to explore
advanced debugging techniques via Python to develop tool kits for GDB.

Introduction
------------

Troubleshooting software bugs is a big challenge for developers. While GDB
provides many “debug commands” to inspect programs’ runtime status, its
non-intuitive usages impede programmers to use it to solve problems. Indeed,
mastering GDB is a long-term process. However, a quick start is not complicated;
you must unlearn what you have learned like Yoda. To better understand how to
use Python in GDB, this article will focus on discussing Python interpreter in
GDB.

Define Commands
---------------

GDB supports customizing commands by using ``define``. It is useful to run a
batch of commands to troubleshoot at the same time. For example, a developer
can display the current frame information by defining a ``sf`` command.

.. code-block:: bash

    # define in .gdbinit
    define sf
      where        # find out where the program is
      info args    # show arguments
      info locals  # show local variables
    end

However, writing a user-defined command may be inconvenient due to limited APIs.
Fortunately, by interacting with Python interpreter in GDB, developers can
utilize Python libraries to establish their debugging tool kits readily. The
following sections show how to use Python to simplify debugging processes.

Dump Memory
-----------

Inspecting a process’s memory information is an effective way to troubleshoot
memory issues. Developers can acquire memory contents by ``info proc mappings``
and ``dump memory``. To simplify these steps, defining a customized command is
useful. However, the implementation is not straightforward by using pure GDB
syntax. Even though GDB supports conditions, processing output is not intuitive.
To solve this problem, using Python API in GDB would be helpful because Python
contains many useful operations for handling strings.

.. code-block:: python

    # mem.py
    import gdb
    import time
    import re

    class DumpMemory(gdb.Command):
        """Dump memory info into a file."""

        def __init__(self):
            super().__init__("dm", gdb.COMMAND_USER)

        def get_addr(self, p, tty):
            """Get memory addresses."""
            cmd = "info proc mappings"
            out = gdb.execute(cmd, tty, True)
            addrs = []
            for l in out.split("\n"):
                if re.match(f".*{p}*", l):
                    s, e, *_ = l.split()
                    addrs.append((s, e))
            return addrs

        def dump(self, addrs):
            """Dump memory result."""
            if not addrs:
                return

            for s, e in addrs:
                f = int(time.time() * 1000)
                gdb.execute(f"dump memory {f}.bin {s} {e}")

        def invoke(self, args, tty):
            try:
                # cat /proc/self/maps
                addrs = self.get_addr(args, tty)
                # dump memory
                self.dump(addrs)
            except Exception as e:
                print("Usage: dm [pattern]")

    DumpMemory()


Running the ``dm`` command will invoke ``DumpMemory.invoke``. By sourcing
or implementing Python scripts in *.gdbinit*, developers can utilize
user-defined commands to trace bugs when a program is running. For example, the
following steps show how to invoke ``DumpMemory`` in GDB.

.. code-block:: bash

    (gdb) start
    ...
    (gdb) source mem.py  # source commands
    (gdb) dm stack       # dump stack to ${timestamp}.bin
    (gdb) shell ls       # ls current dir
    1577283091687.bin  a.cpp  a.out  mem.py

Dump JSON
---------

Parsing JSON is helpful when a developer is inspecting a JSON string in a
running program. GDB can parse a ``std::string`` via ``gdb.parse_and_eval``
and return it as a ``gdb.Value``. By processing ``gdb.Value``, developers can
pass a JSON string into Python ``json`` API and print it in a pretty format.

.. code-block:: python

    # dj.py
    import gdb
    import re
    import json

    class DumpJson(gdb.Command):
        """Dump std::string as a styled JSON."""

        def __init__(self):
            super().__init__("dj", gdb.COMMAND_USER)

        def get_json(self, args):
            """Parse std::string to JSON string."""
            ret = gdb.parse_and_eval(args)
            typ = str(ret.type)
            if re.match("^std::.*::string", typ):
                return json.loads(str(ret))
            return None

        def invoke(self, args, tty):
            try:
                # string to json string
                s = self.get_json(args)
                # json string to object
                o = json.loads(s)
                print(json.dumps(o, indent=2))
            except Exception as e:
                print(f"Parse json error! {args}")

    DumpJson()

The command ``dj`` displays a more readable JSON format in GDB. This command
helps improve visual recognization when a JSON string large. Also, by using
this command, it can detect or monitor whether a ``std::string`` is JSON or
not.

.. code-block:: bash

    (gdb) start
    (gdb) list
    1       #include <string>
    2
    3       int main(int argc, char *argv[])
    4       {
    5           std::string json = R"({"foo": "FOO","bar": "BAR"})";
    6           return 0;
    7       }
    ...
    (gdb) ptype json
    type = std::string
    (gdb) p json
    $1 = "{\"foo\": \"FOO\",\"bar\": \"BAR\"}"
    (gdb) source dj.py
    (gdb) dj json
    {
      "foo": "FOO",
      "bar": "BAR"
    }

Highlight Syntax
----------------

Syntax highlighting is useful for developers to trace source code or to
troubleshoot issues. By using `Pygments`_, applying color to the source is easy
without defining ANSI escape code manually. The following example shows how to
apply color to the ``list`` command output.

.. code-block:: python

    import gdb

    from pygments import highlight
    from pygments.lexers import CLexer
    from pygments.formatters import TerminalFormatter

    class PrettyList(gdb.Command):
        """Print source code with color."""

        def __init__(self):
            super().__init__("pl", gdb.COMMAND_USER)
            self.lex = CLexer()
            self.fmt = TerminalFormatter()

        def invoke(self, args, tty):
            try:
                out = gdb.execute(f"l {args}", tty, True)
                print(highlight(out, self.lex, self.fmt))
            except Exception as e:
                print(e)

    PrettyList()

Tracepoints
-----------

Although a developer can insert ``printf``, ``std::cout``, or ``syslog`` to
inspect functions, printing messages is not an effective way to debug when a
project is enormous. Developers may waste their time in building source code
and may acquire little information. Even worse, the output may become too much
to detect problems. In fact, inspecting functions or variables do not require
to embed *print functions* in code. By writing a Python script with GDB API,
developers can customize watchpoints to trace issues dynamically at runtime.
For example, by implementing a ``gdb.Breakpoint`` and a ``gdb.Command``, it is
useful for developers to acquire essential information, such as parameters,
call stacks, or memory usage.

.. code-block:: python

    # tp.py
    import gdb

    tp = {}

    class Tracepoint(gdb.Breakpoint):
        def __init__(self, *args):
            super().__init__(*args)
            self.silent = True
            self.count = 0

        def stop(self):
            self.count += 1
            frame = gdb.newest_frame()
            block = frame.block()
            sym_and_line = frame.find_sal()
            framename = frame.name()
            filename = sym_and_line.symtab.filename
            line = sym_and_line.line
            # show tracepoint info
            print(f"{framename} @ {filename}:{line}")
            # show args and vars
            for s in block:
                if not s.is_argument and not s.is_variable:
                    continue
                typ = s.type
                val = s.value(frame)
                size = typ.sizeof
                name = s.name
                print(f"\t{name}({typ}: {val}) [{size}]")
            # do not stop at tracepoint
            return False

    class SetTracepoint(gdb.Command):
        def __init__(self):
            super().__init__("tp", gdb.COMMAND_USER)

        def invoke(self, args, tty):
            try:
                global tp
                tp[args] = Tracepoint(args)
            except Exception as e:
                print(e)

    def finish(event):
        for t, p in tp.items():
            c = p.count
            print(f"Tracepoint '{t}' Count: {c}")

    gdb.events.exited.connect(finish)
    SetTracepoint()

Instead of inserting ``std::cout`` at the beginning of functions, using a
tracepoint at a function's entry point provides useful information to inspect
arguments, variables, and stacks. For instance, by setting a tracepoint at
``fib``, it is helpful to examine memory usage, stack, and the number of calls.

.. code-block:: cpp

    int fib(int n)
    {
        if (n < 2) {
            return 1;
        }
        return fib(n-1) + fib(n-2);
    }

    int main(int argc, char *argv[])
    {
        fib(3);
        return 0;
    }

The following output shows the result of an inspection of the function ``fib``.
In this case, tracepoints display all information a developer needs, including
arguments' value, recursive flow, and variables' size. By using tracepoints,
developers can acquire more useful information comparing with ``std::cout``.

.. code-block:: bash

    (gdb) source tp.py
    (gdb) tp main
    Breakpoint 1 at 0x647: file a.cpp, line 12.
    (gdb) tp fib
    Breakpoint 2 at 0x606: file a.cpp, line 3.
    (gdb) r
    Starting program: /root/a.out
    main @ a.cpp:12
            argc(int: 1) [4]
            argv(char **: 0x7fffffffe788) [8]
    fib @ a.cpp:3
            n(int: 3) [4]
    fib @ a.cpp:3
            n(int: 2) [4]
    fib @ a.cpp:3
            n(int: 1) [4]
    fib @ a.cpp:3
            n(int: 0) [4]
    fib @ a.cpp:3
            n(int: 1) [4]
    [Inferior 1 (process 5409) exited normally]
    Tracepoint 'main' Count: 1
    Tracepoint 'fib' Count: 5

Profiling
---------

Without inserting timestamps, profiling is still feasible through tracepoints.
By using a ``gdb.FinishBreakpoint`` after a ``gdb.Breakpoint``, GDB sets a
temporary breakpoint at the return address of a frame for developers to get
the current timestamp and to calculate the time difference. Note that profiling
via GDB is not precise. Other tools, such as `Linux perf`_ or `Valgrind`_,
provide more useful and accurate information to trace performance issues.

.. code-block:: python

    import gdb
    import time

    class EndPoint(gdb.FinishBreakpoint):
        def __init__(self, breakpoint, *a, **kw):
            super().__init__(*a, **kw)
            self.silent = True
            self.breakpoint = breakpoint

        def stop(self):
            # normal finish
            end = time.time()
            start, out = self.breakpoint.stack.pop()
            diff = end - start
            print(out.strip())
            print(f"\tCost: {diff}")
            return False

    class StartPoint(gdb.Breakpoint):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.silent = True
            self.stack = []

        def stop(self):
            start = time.time()
            # start, end, diff
            frame = gdb.newest_frame()
            sym_and_line = frame.find_sal()
            func = frame.function().name
            filename = sym_and_line.symtab.filename
            line = sym_and_line.line
            block = frame.block()

            args = []
            for s in block:
                if not s.is_argument:
                    continue
                name = s.name
                typ = s.type
                val = s.value(frame)
                args.append(f"{name}: {val} [{typ}]")

            # format
            out = ""
            out += f"{func} @ {filename}:{line}\n"
            for a in args:
                out += f"\t{a}\n"

            # append current status to a breakpoint stack
            self.stack.append((start, out))
            EndPoint(self, internal=True)
            return False

    class Profile(gdb.Command):
        def __init__(self):
            super().__init__("prof", gdb.COMMAND_USER)

        def invoke(self, args, tty):
            try:
                StartPoint(args)
            except Exception as e:
                print(e)

    Profile()


The following output shows the profiling result by setting a tracepoint at the
function ``fib``. It is convenient to inspect the function's performance and
stack at the same time.

.. code-block:: bash

    (gdb) source prof.py
    (gdb) prof fib
    Breakpoint 1 at 0x606: file a.cpp, line 3.
    (gdb) r
    Starting program: /root/a.out
    fib(int) @ a.cpp:3
            n: 1 [int]
            Cost: 0.0007786750793457031
    fib(int) @ a.cpp:3
            n: 0 [int]
            Cost: 0.002572298049926758
    fib(int) @ a.cpp:3
            n: 2 [int]
            Cost: 0.008517265319824219
    fib(int) @ a.cpp:3
            n: 1 [int]
            Cost: 0.0014069080352783203
    fib(int) @ a.cpp:3
            n: 3 [int]
            Cost: 0.01870584487915039

Pretty Print
------------

Although ``set print pretty on`` in GDB offers a better format to inspect
variables, developers may require to parse variables' value for readability.
Take the system call ``stat`` as an example. While it provides useful information
to examine file attributes, the output values, such as the permission, may not
be readable for debugging. By implementing a user-defined pretty print,
developers can parse ``struct stat`` and output information in a readable format.

.. code-block:: python

    import gdb
    import pwd
    import grp
    import stat
    import time

    from datetime import datetime


    class StatPrint:
        def __init__(self, val):
            self.val = val

        def get_filetype(self, st_mode):
            if stat.S_ISDIR(st_mode):
                return "directory"
            if stat.S_ISCHR(st_mode):
                return "character device"
            if stat.S_ISBLK(st_mode):
                return "block device"
            if stat.S_ISREG:
                return "regular file"
            if stat.S_ISFIFO(st_mode):
                return "FIFO"
            if stat.S_ISLNK(st_mode):
                return "symbolic link"
            if stat.S_ISSOCK(st_mode):
                return "socket"
            return "unknown"

        def get_access(self, st_mode):
            out = "-"
            info = ("r", "w", "x")
            perm = [
                (stat.S_IRUSR, stat.S_IWUSR, stat.S_IXUSR),
                (stat.S_IRGRP, stat.S_IRWXG, stat.S_IXGRP),
                (stat.S_IROTH, stat.S_IWOTH, stat.S_IXOTH),
            ]
            for pm in perm:
                for c, p in zip(pm, info):
                    out += p if st_mode & c else "-"
            return out

        def get_time(self, st_time):
            tv_sec = int(st_time["tv_sec"])
            return datetime.fromtimestamp(tv_sec).isoformat()

        def to_string(self):
            st = self.val
            st_ino = int(st["st_ino"])
            st_mode = int(st["st_mode"])
            st_uid = int(st["st_uid"])
            st_gid = int(st["st_gid"])
            st_size = int(st["st_size"])
            st_blksize = int(st["st_blksize"])
            st_blocks = int(st["st_blocks"])
            st_atim = st["st_atim"]
            st_mtim = st["st_mtim"]
            st_ctim = st["st_ctim"]

            out = "{\n"
            out += f"Size: {st_size}\n"
            out += f"Blocks: {st_blocks}\n"
            out += f"IO Block: {st_blksize}\n"
            out += f"Inode: {st_ino}\n"
            out += f"Access: {self.get_access(st_mode)}\n"
            out += f"File Type: {self.get_filetype(st_mode)}\n"
            out += f"Uid: ({st_uid}/{pwd.getpwuid(st_uid).pw_name})\n"
            out += f"Gid: ({st_gid}/{grp.getgrgid(st_gid).gr_name})\n"
            out += f"Access: {self.get_time(st_atim)}\n"
            out += f"Modify: {self.get_time(st_mtim)}\n"
            out += f"Change: {self.get_time(st_ctim)}\n"
            out += "}"
            return out

    p = gdb.printing.RegexpCollectionPrettyPrinter("sp")
    p.add_printer("stat", "^stat$", StatPrint)

    o = gdb.current_objfile()
    gdb.printing.register_pretty_printer(o, p)

By sourcing the previous Python script, the ``PrettyPrinter`` can recognize
``struct stat`` and output a readable format for developers to inspect file
attributes. Without inserting functions to parse and print ``struct stat``, it
is a more convenient way to acquire a better output from Python API.

.. code-block:: bash

    (gdb) list 15
    10          struct stat st;
    11
    12          if ((rc = stat("./a.cpp", &st)) < 0) {
    13              perror("stat failed.");
    14              goto end;
    15          }
    16
    17          rc = 0;
    18       end:
    19          return rc;
    (gdb) source st.py
    (gdb) b 17
    Breakpoint 1 at 0x762: file a.cpp, line 17.
    (gdb) r
    Starting program: /root/a.out

    Breakpoint 1, main (argc=1, argv=0x7fffffffe788) at a.cpp:17
    17          rc = 0;
    (gdb) p st
    $1 = {
    Size: 298
    Blocks: 8
    IO Block: 4096
    Inode: 1322071
    Access: -rw-rw-r--
    File Type: regular file
    Uid: (0/root)
    Gid: (0/root)
    Access: 2019-12-28T15:53:17
    Modify: 2019-12-28T15:53:01
    Change: 2019-12-28T15:53:01
    }

Note that developers can disable a user-defined pretty-print via the command
``disable``. For example, the previous Python script registers a pretty printer
under the global pretty-printers. By calling ``disable pretty-print``, the
printer ``sp`` will be disabled.

.. code-block:: bash

    (gdb) disable pretty-print global sp
    1 printer disabled
    1 of 2 printers enabled
    (gdb) i pretty-print
    global pretty-printers:
      builtin
        mpx_bound128
      sp [disabled]
        stat

Additionally, developers can exclude a printer in the current GDB debugging
session if it is no longer required. The following snippet shows how to delete
the ``sp`` printer through ``gdb.pretty_printers.remove``.

.. code-block:: bash

    (gdb) python
    >import gdb
    >for p in gdb.pretty_printers:
    >    if p.name == "sp":
    >        gdb.pretty_printers.remove(p)
    >end
    (gdb) i pretty-print
    global pretty-printers:
      builtin
        mpx_bound128

Conclusion
----------

Integrating Python interpreter into GDB offers many flexible ways to
troubleshoot issues. While many integrated development environments (IDEs) may
embed GDB to debug visually, GDB allows developers to implement their commands
and parse variables’ output at runtime. By using debugging scripts, developers
can monitor and record necessary information without modifying their code.
Honestly, inserting or enabling debugging code blocks may change a program’s
behaviors, and developers should get rid of this bad habit. Also, when a problem
is reproduced, GDB can attach that process and examine its status without stopping
it. Obviously, debugging via GDB is inevitable if a challenging issue emerges.
Thanks to integrating Python into GDB, developing a script to troubleshoot becomes
more accessible that leads to developers establishing their debugging methods
diversely.


Reference
---------

1. `Extending GDB using Python`_
2. `gcc/gcc/gdbhooks.py`_
3. `gdbinit/Gdbinit`_
4. `cyrus-and/gdb-dashboard`_
5. `hugsy/gef`_
6. `sharkdp/stack-inspector`_
7. `gdb Debugging Full Example (Tutorial)`_

.. _Pygments: https://pygments.org/
.. _Extending GDB using Python: https://sourceware.org/gdb/onlinedocs/gdb/Python.html
.. _gcc/gcc/gdbhooks.py: https://github.com/gcc-mirror/gcc/blob/master/gcc/gdbhooks.py
.. _hugsy/gef: https://github.com/hugsy/gef
.. _cyrus-and/gdb-dashboard: https://github.com/cyrus-and/gdb-dashboard
.. _gdbinit/Gdbinit: https://github.com/gdbinit/Gdbinit
.. _sharkdp/stack-inspector: https://github.com/sharkdp/stack-inspector
.. _GDB Text User Interface (TUI): https://sourceware.org/gdb/onlinedocs/gdb/TUI.html
.. _Linux perf: https://github.com/torvalds/linux/tree/master/tools/perf
.. _Valgrind: https://valgrind.org/
.. _gdb Debugging Full Example (Tutorial): http://www.brendangregg.com/blog/2016-08-09/gdb-example-ncurses.html
