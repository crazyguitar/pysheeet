.. meta::
    :description lang=en: Collect useful snippets of unicode
    :keywords: Python, Python3, Python Unicode, Python Unicode Cheat Sheet

=======
Unicode
=======

.. contents:: Table of Contents
    :backlinks: none


Encode: unicode code point to bytes
------------------------------------

.. code-block:: python

    >>> s = u'Café'
    >>> type(s.encode('utf-8'))
    <class 'bytes'>

Decode: bytes to unicode code point
------------------------------------

.. code-block:: python

    >>> s = bytes('Café', encoding='utf-8')
    >>> s.decode('utf-8')
    'Café'

Get unicode code point
-----------------------

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

python2 ``str`` is equivalent to byte string
---------------------------------------------

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


python3 ``str`` is equivalent to unicode string
-------------------------------------------------

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


python2 take ``str`` char as byte character
--------------------------------------------

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

python3 take ``str`` char as unicode character
-----------------------------------------------

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


unicode normalization
----------------------

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


Avoid UnicodeDecodeError
-------------------------

.. code-block:: python

    # raise a UnicodeDecodeError

    >>> u = b"0xff"
    >>> u.decode('utf-8')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte

    # raise a UnicodeDecodeError

    >>> u.decode('utf-8', "strict")
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte

    # use U+FFFD, REPLACEMENT CHARACTER

    >>> u.decode('utf-8', "replace")
    '\ufffd'

    # inserts a \xNN escape sequence

    >>> u.decode('utf-8', "backslashreplace")
    '\\xff'

    # leave the character out of the Unicode result

    >>> u.decode('utf-8', "ignore")
    ''

Long String
-----------

Original long string

.. code-block:: python

    # original long string
    >>> s = 'This is a very very very long python string'
    >>> s
    'This is a very very very long python string'

Single quote with an escaping backslash

.. code-block:: python

    >>> s = "This is a very very very " \
    ...     "long python string"
    >>> s
    'This is a very very very long python string'

Using brackets

.. code-block:: python

    >>> s = ("This is a very very very "
    ...      "long python string")
    >>> s
    'This is a very very very long python string'

Using ``+``

.. code-block:: python

    >>> s = ("This is a very very very " +
    ...      "long python string")
    >>> s
    'This is a very very very long python string'

Using triple-quote with an escaping backslash

.. code-block:: python

    >>> s = '''This is a very very very \
    ... long python string'''
    >>> s
    'This is a very very very long python string'


String Formatting 
-----------------

... code-block:: python 

    # using ``%`` operator
    >>> print("This is a %s %s" % ('long', 'string'))
    >>> 'This is a long string' 

    # using new ``format`` function  with kwargs
    >>> print("This is a {long} {string}".format(long='long', string='string'))

    # using new ``format`` function with positional argument
    >>> print("This is a {0} {1}".format(long, string))

Using ``f-string``

... code-block:: python
    >>> long = 'long'
    >>> string = 'string'
    >>> f'This is a {long} {string}'
    >>> 'This is a long string'

Using ``Standard Template String`` 

... code-block:: python 
    >>> from string import Template
    >>> s = Template('This is a $long $string')
    >>> s.substitute(long='long', string='string')
    >>> 'This is a long string'