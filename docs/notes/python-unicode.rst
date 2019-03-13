.. meta::
    :description lang=en: Collect useful snippets of unicode
    :keywords: Python, Python3, Python Unicode, Python Unicode Cheat Sheet

=======
Unicode
=======

The main goal of this cheat sheet is to collect some common snippets which are
related to Unicode. In Python 3, strings are represented by Unicode instead of
bytes. Further information can be found on PEP `3100 <https://www.python.org/dev/peps/pep-3100>`_

**ASCII** code is the most well-known standard which defines numeric codes
for characters. The numeric values only define 128 characters originally,
so ASCII only contains control codes, digits, lowercase letters, uppercase
letters, etc. However, it is not enough for us to represent characters such as
accented characters, Chinese characters, or emoji existed around the world.
Therefore, **Unicode** was developed to solve this issues. It defines the
*code point* to represent various characters like ASCII but the number of
characters is up to 1,111,998.

.. contents:: Table of Contents
    :backlinks: none

String
------

In Python 2, strings are represented in *bytes*, not *Unicode*. Python provides
different types of string such as Unicode string, raw string, and so on.
In this case, if we want to declare a Unicode string, we add ``u`` prefix for
string literals.

.. code-block:: python

    >>> s = 'Café'  # byte string
    >>> s
    'Caf\xc3\xa9'
    >>> type(s)
    <type 'str'>
    >>> u = u'Café' # unicode string
    >>> u
    u'Caf\xe9'
    >>> type(u)
    <type 'unicode'>

In Python 3, strings are represented in *Unicode*. If we want to represent a
byte string, we add the ``b`` prefix for string literals. Note that the early
Python versions (3.0-3.2) do not support the ``u`` prefix. In order to ease
the pain to migrate Unicode aware applications from Python 2, Python 3.3 once
again supports the ``u`` prefix for string literals. Further information can
be found on PEP `414 <https://www.python.org/dev/peps/pep-0414>`_

.. code-block:: python

    >>> s = 'Café'
    >>> type(s)
    <class 'str'>
    >>> s
    'Café'
    >>> s.encode('utf-8')
    b'Caf\xc3\xa9'
    >>> s.encode('utf-8').decode('utf-8')
    'Café'

Characters
----------

Python 2 takes all string characters as bytes. In this case, the length of
strings may be not equivalent to the number of characters. For example,
the length of ``Café`` is 5, not 4 because ``é`` is encoded as a 2 bytes
character.

.. code-block:: python

    >>> s= 'Café'
    >>> print([_c for _c in s])
    ['C', 'a', 'f', '\xc3', '\xa9']
    >>> len(s)
    5
    >>> s = u'Café'
    >>> print([_c for _c in s])
    [u'C', u'a', u'f', u'\xe9']
    >>> len(s)
    4

Python 3 takes all string characters as Unicode code point. The lenght of
a string is always equivalent to the number of characters.

.. code-block:: python

    >>> s = 'Café'
    >>> print([_c for _c in s])
    ['C', 'a', 'f', 'é']
    >>> len(s)
    4
    >>> bs = bytes(s, encoding='utf-8')
    >>> print(bs)
    b'Caf\xc3\xa9'
    >>> len(bs)
    5

Unicode Code Point
------------------

`ord <https://docs.python.org/3/library/functions.html#ord>`_ is a powerful
built-in function to get a Unicode code point from a given character.
Consequently, If we want to check a Unicode code point of a character, we can
use ``ord``.

.. code-block:: python

    >>> s = u'Café'
    >>> for _c in s: print('U+%04x' % ord(_c))
    ...
    U+0043
    U+0061
    U+0066
    U+00e9
    >>> u = '中文'
    >>> for _c in u: print('U+%04x' % ord(_c))
    ...
    U+4e2d
    U+6587


Encoding
--------

A *Unicode code point* transfers to a *byte string* is called encoding.

.. code-block:: python

    >>> s = u'Café'
    >>> type(s.encode('utf-8'))
    <class 'bytes'>

Decodeing
---------

A *byte string* transfers to a *Unicode code point* is called encoding.

.. code-block:: python

    >>> s = bytes('Café', encoding='utf-8')
    >>> s.decode('utf-8')
    'Café'

Unicode Normalization
---------------------

Some characters can be represented in two similar form. For example, the
character, ``é`` can be written as ``e ́`` (Canonical Decomposition) or ``é``
(Canonical Composition). In this case, we may acquire unexpected results when we
are comparing two strings even though they look alike. Therefore, we can
normalize a Unicode form to solve the issue.

.. code-block:: python

    # python 3
    >>> u1 = 'Café'       # unicode string
    >>> u2 = 'Cafe\u0301'
    >>> u1, u2
    ('Café', 'Café')
    >>> len(u1), len(u2)
    (4, 5)
    >>> u1 == u2
    False
    >>> u1.encode('utf-8') # get u1 byte string
    b'Caf\xc3\xa9'
    >>> u2.encode('utf-8') # get u2 byte string
    b'Cafe\xcc\x81'
    >>> from unicodedata import normalize
    >>> s1 = normalize('NFC', u1)  # get u1 NFC format
    >>> s2 = normalize('NFC', u2)  # get u2 NFC format
    >>> s1 == s2
    True
    >>> s1.encode('utf-8'), s2.encode('utf-8')
    (b'Caf\xc3\xa9', b'Caf\xc3\xa9')
    >>> s1 = normalize('NFD', u1)  # get u1 NFD format
    >>> s2 = normalize('NFD', u2)  # get u2 NFD format
    >>> s1, s2
    ('Café', 'Café')
    >>> s1 == s2
    True
    >>> s1.encode('utf-8'), s2.encode('utf-8')
    (b'Cafe\xcc\x81', b'Cafe\xcc\x81')


Avoid ``UnicodeDecodeError``
----------------------------

Python raises `UnicodeDecodeError` when byte strings cannot decode to Unicode
code points. If we want to avoid this exception, we can pass *replace*,
*backslashreplace*, or *ignore* to errors argument in `decode <https://docs.python.org/3/library/stdtypes.html#bytes.decode>`_.

.. code-block:: python

    >>> u = b"\xff"
    >>> u.decode('utf-8', 'strict')
        Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
    >>> # use U+FFFD, REPLACEMENT CHARACTER
    >>> u.decode('utf-8', "replace")
    '\ufffd'
    >>> # inserts a \xNN escape sequence
    >>> u.decode('utf-8', "backslashreplace")
    '\\xff'
    >>> # leave the character out of the Unicode result
    >>> u.decode('utf-8', "ignore")
    ''

Long String
-----------

The following snippet shows common ways to declare a multi-line string in
Python.

.. code-block:: python

    # original long string
    s = 'This is a very very very long python string'

    # Single quote with an escaping backslash
    s = "This is a very very very " \
        "long python string"

    # Using brackets
    s = (
        "This is a very very very "
        "long python string"
    )

    # Using ``+``
    s = (
        "This is a very very very " +
        "long python string"
    )

    # Using triple-quote with an escaping backslash
    s = '''This is a very very very \
    long python string'''


String Formatting 
-----------------

.. code-block:: python

    # Using ``%`` operator
    >>> print("This is a %s %s" % ('long', 'string'))
    >>> 'This is a long string' 

    # using new ``format`` function with positional argument
    >>> print("This is a {0} {1}".format(long, string))

    # Using ``f-string`` new in Python 3.6
    >>> long = 'long'
    >>> string = 'string'
    >>> f'This is a {long} {string}'
    >>> 'This is a long string'

    # Using ``Standard Template String`` 
    >>> from string import Template
    >>> s = Template('This is a $long $string')
    >>> s.substitute(long='long', string='string')
