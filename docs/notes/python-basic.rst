=======================
Python basic cheatsheet
=======================

Python Naming Styles
--------------------

.. code-block:: python

    # see: PEP8

    # class: camel style only
    MyClass

    # func, module, package
    var_underscore_separate

    # for public use
    var

    # for internal use
    _var

    # convention to avoid conflict keyword
    var_

    # for private use in class
    __var

    # for protect use in class
    _var_

    # "magic" method or attributes
    # ex: __init__, __file__, __main__
    __var__

    # for "internal" use throwaway variable
    # usually used in loop
    # ex: [_ for _ in range(10)]
    # or variable not used
    # for _, a in [(1,2),(3,4)]: print a
    _

Check object attributes
-----------------------

.. code-block:: python

    # example of check list attributes
    >>> dir(list)
    ['__add__', '__class__', ...]

Define a function __doc__
-------------------------

.. code-block:: python

    # Define a function document
    >>> def Example():
    ...   """ This is an example function """
    ...   print "Example function"
    ...
    >>> Example.__doc__
    ' This is an example function '

    # Or using help function
    >>> help(Example)

Check instance type
-------------------

.. code-block:: python

    >>> ex = 10
    >>> isinstance(ex,int)
    True

Check, Get, Set attribute
-------------------------

.. code-block:: python

    >>> class Example(object):
    ...   def __init__(self):
    ...     self.name = "ex"
    ...   def printex(self):
    ...     print "This is an example"
    ...

    # Check object has attributes
    # hasattr(obj, 'attr')
    >>> ex = Example()
    >>> hasattr(ex,"name")
    True
    >>> hasattr(ex,"printex")
    True
    >>> hasattr(ex,"print")
    False

    # Get object attribute
    # getattr(obj, 'attr')
    >>> getattr(ex,'name')
    'ex'

    # Set object attribute
    # setattr(obj, 'attr', value)
    >>> setattr(ex,'name','example')
    >>> ex.name
    'example'

Check inheritance
-----------------

.. code-block:: python

    >>> class Example(object):
    ...   def __init__(self):
    ...     self.name = "ex"
    ...   def printex(self):
    ...     print "This is an Example"
    ...
    >>> issubclass(Example, object)
    True

Check all global variables
--------------------------

.. code-block:: python

    # globals() return a dictionary
    # {'variable name': variable value}
    >>> globals()
    {'args': (1, 2, 3, 4, 5), ...}

Check "callable"
----------------

.. code-block:: python

    >>> a = 10
    >>> def fun():
    ...   print "I am callable"
    ...
    >>> callable(a)
    False
    >>> callable(fun)
    True

Get function/class name
-----------------------

.. code-block:: python

    >>> class ExampleClass(object):
    ...   pass
    ...
    >>> def example_function():
    ...   pass
    ...
    >>> ex = ExampleClass()
    >>> ex.__class__.__name__
    'ExampleClass'
    >>> example_function.__name__
    'example_function'


``__new__`` & ``__init__``
--------------------------

.. code-block:: python

    # __init__ will invoke
    >>> class ClassA(object):
    ...     def __new__(cls, arg):
    ...         print '__new__ ' + arg
    ...         return object.__new__(cls, arg)
    ...     def __init__(self, arg):
    ...         print '__init__ ' + arg
    ...
    >>> o = ClassA("Hello")
    __new__ Hello
    __init__ Hello

    # init won't be invoke
    >>> class ClassB(object):
    ...     def __new__(cls, arg):
    ...         print '__new__ ' + arg
    ...         return object
    ...     def __init__(self, arg):
    ...         print '__init__ ' + arg
    ...
    >>> o = ClassB("Hello")
    __new__ Hello

Representations of your class behave
------------------------------------

.. code-block:: python

    >>> class Example(object):
    ...    def __str__(self):
    ...       return "Example __str__"
    ...    def __repr__(self):
    ...       return "Example __repr__"
    ...
    >>> print str(Example())
    Example __str__
    >>> Example()
    Example __repr__

Get list item "SMART"
---------------------

.. code-block:: python

    >>> a = [1, 2, 3, 4, 5]
    >>> a[0]
    1
    >>>a[-1]
    5
    >>> a[0:]
    [1, 2, 3, 4, 5]
    >>> a[:-1]
    [1, 2, 3, 4]

    # a[start:end:step]
    >>> a[0:-1:2]
    [1, 3]

    # using slice object
    # slice(start,end,step)
    >>> s = slice(0, -1, 2)
    >>> a[s]
    [1, 3]

    # Get index and item in loop
    >>> a = range(3)
    >>> for idx, item in enumerate(a):
    ...   print (idx,item),
    ...
    (0, 0) (1, 1) (2, 2)

    # Transfer two list into tuple list
    >>> a = [1, 2, 3, 4, 5]
    >>> b = [2, 4, 5, 6, 8]
    >>> zip(a, b)
    [(1, 2), (2, 4), (3, 5), (4, 6), (5, 8)]

    # with filter
    >>> [x for x in range(5) if x > 1]
    [2, 3, 4]
    >>> l = ['1', '2', 3, 'Hello', 4]
    >>> predicate = lambda x: isinstance(x, int)
    >>> filter(predicate, l)
    [3, 4]

Get dictionary item "SMART"
---------------------------

.. code-block:: python

    # get dictionary all keys
    >>> a = {"1":1, "2":2, "3":3}
    >>> b = {"2":2, "3":3, "4":4}
    >>> a.keys()
    ['1', '3', '2']

    # get dictionary key and value as tuple
    >>> a.items()
    [('1', 1), ('3', 3), ('2', 2)]

    # find same key between two dictionary
    >>> [_ for _ in a.keys() if _ in b.keys()]
    ['3', '2']
    # better way
    >>> c = set(a).intersection(set(b))
    >>> list(c)
    ['3', '2']
    # or
    >>> [_ for _ in a if _ in b]
    ['3', '2']

    # update dictionary
    >>> a.update(b)
    >>> a
    {'1': 1, '3': 3, '2': 2, '4': 4}

Set a list/dict "SMART"
-----------------------

.. code-block:: python

    # get a list with init value
    >>> ex = [0] * 10
    >>> ex
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # extend two list
    >>> a = [1, 2, 3]; b = ['a', 'b']
    >>> a + b
    [1, 2, 3, 'a', 'b']

    # using list comprehension
    >>> [x for x in range(10)]
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> fn = lambda x: x**2
    >>> [fn(x) for x in range(5)]
    [0, 1, 4, 9, 16]
    >>> {'{0}'.format(x): x for x in range(3)}
    {'1': 1, '0': 0, '2': 2}

    # using builtin function "map"
    >>> map(fn, range(5))
    [0, 1, 4, 9, 16]

NamedTuple
----------

.. code-block:: python

    # namedtuple(typename, field_names)
    # replace define class without method
    >>> from collections import namedtuple
    >>> Example = namedtuple("Example",'a b c')
    >>> e = Example(1, 2, 3)
    >>> print e.a, e[1], e[1] + e.b
    1 2 4

Delegating Iteration (__iter__)
--------------------------------

.. code-block:: python

    # __iter__ return an iterator object
    # Be careful: list is an "iterable" object not an "iterator"
    >>> class Example(object):
    ...    def __init__(self,list_):
    ...       self._list = list_
    ...    def __iter__(self):
    ...      return iter(self._list)
    ...
    >>> ex = Example([1, 2, 3, 4, 5])
    >>> for _ in ex: print _,
    ...
    1 2 3 4 5

Using Generator as Iterator
---------------------------

.. code-block:: python

    # see: PEP289
    >>> a = (_ for _ in range(10))
    >>> for _ in a: print _,
    ...
    0 1 2 3 4 5 6 7 8 9

    # equivalent to
    >>> def generator():
    ...   for _ in range(10):
    ...     yield _
    ...
    >>> for _ in generator(): print _,
    ...
    0 1 2 3 4 5 6 7 8 9

Emulating a list
----------------

.. code-block:: python

    >>> class EmuList(object):
    ...   def __init__(self, list_):
    ...     self._list = list_
    ...   def __repr__(self):
    ...     return "EmuList: " + repr(self._list)
    ...   def append(self, item):
    ...     self._list.append(item)
    ...   def remove(self, item):
    ...     self._list.remove(item)
    ...   def __len__(self):
    ...     return len(self._list)
    ...   def __getitem__(self, sliced):
    ...     return self._list[sliced]
    ...   def __setitem__(self, sliced, val):
    ...     self._list[sliced] = val
    ...   def __delitem__(self, sliced):
    ...     del self._list[sliced]
    ...   def __contains__(self, item):
    ...     return item in self._list
    ...   def __iter__(self):
    ...     return iter(self._list)
    ...
    >>> emul = EmuList(range(5))
    >>> emul
    EmuList: [0, 1, 2, 3, 4]
    >>> emul[1:3]  #  __getitem__
    [1, 2]
    >>> emul[0:4:2]  #  __getitem__
    [0, 2]
    >>> len(emul)  #  __len__
    5
    >>> emul.append(5)
    >>> emul
    EmuList: [0, 1, 2, 3, 4, 5]
    >>> emul.remove(2)
    >>> emul
    EmuList: [0, 1, 3, 4, 5]
    >>> emul[3] = 6  # __setitem__
    >>> emul
    EmuList: [0, 1, 3, 6, 5]
    >>> 0 in emul  # __contains__
    True


Emulating a dictionary
----------------------

.. code-block:: python

    >>> class EmuDict(object):
    ...   def __init__(self, dict_):
    ...     self._dict = dict_
    ...   def __repr__(self):
    ...     return "EmuDict: " + repr(self._dict)
    ...   def __getitem__(self, key):
    ...     return self._dict[key]
    ...   def __setitem__(self, key, val):
    ...     self._dict[key] = val
    ...   def __delitem__(self, key):
    ...     del self._dict[key]
    ...   def __contains__(self, key):
    ...     return key in self._dict
    ...   def __iter__(self):
    ...     return iter(self._dict.keys())
    ... 
    >>> _ = {"1":1, "2":2, "3":3}
    >>> emud = EmuDict(_)
    >>> emud  # __repr__
    EmuDict: {'1': 1, '2': 2, '3': 3}
    >>> emud['1']  # __getitem__
    1
    >>> emud['5'] = 5  # __setitem__
    >>> emud
    EmuDict: {'1': 1, '2': 2, '3': 3, '5': 5}
    >>> del emud['2']  # __delitem__
    >>> emud
    EmuDict: {'1': 1, '3': 3, '5': 5}
    >>> for _ in emud: print emud[_],  # __iter__
    ... 
    1 3 5
    >>> '1' in emud  # __contains__
    True


Emulating a matrix multiplication
----------------------------------

.. code-block:: python

    # PEP 465 - "@" represent matrix multiplication
    #
    # Need Python-3.5 or above
    >>> class Arr:
    ...     def __init__(self, *arg):
    ...         self._arr = arg
    ...     def __matmul__(self, other):
    ...         if not isinstance(other, Arr):
    ...             raise TypeError
    ...         if len(self) != len(other):
    ...             raise ValueError
    ...         return sum([x*y for x, y in zip(self._arr, other._arr)])
    ...     def __imatmul__(self, other):
    ...         if not isinstance(other, Arr):
    ...             raise TypeError
    ...         if len(self) != len(other):
    ...             raise ValueError
    ...         res = sum([x*y for x, y in zip(self._arr, other._arr)])
    ...         self._arr = [res]
    ...         return self
    ...     def __len__(self):
    ...         return len(self._arr)
    ...     def __str__(self):
    ...         return self.__repr__()
    ...     def __repr__(self):
    ...         return "Arr({})".format(repr(self._arr))
    ...
    >>> a = Arr(9, 5, 2, 7)
    >>> b = Arr(5, 5, 6, 6)
    >>> a @ b  # __matmul__
    124
    >>> a @= b  # __imatmul__
    >>> a
    Arr([124])


Decorator
---------

.. code-block:: python

    # see: PEP318
    >>> def decor(func):
    ...   @functools.wraps(func)
    ...   def wrapper(*args,**kwargs):
    ...     print "wrapper"
    ...     func()
    ...     print "-------"
    ...   return wrapper
    ...
    >>> @decor
    ... def example():
    ...   print "Example"
    ...
    >>> example()
    wrapper
    Example
    -------

    # equivalent to
    >>> def example():
    ...   print "Example"
    ...
    >>> example = decor(example)
    >>> example()
    wrapper
    Example
    -------

Decorator with arguments
------------------------

.. code-block:: python

    >>> def example(val):
    ...   def decor(func):
    ...     @functools.wraps(func)
    ...     def wrapper(*args,**kargs):
    ...       print "Val is {0}".format(val)
    ...       func()
    ...     return wrapper
    ...   return decor
    ...
    >>> @example(10)
    ... def undecor():
    ...   print "This is undecor func"
    ...
    >>> undecor()
    Val is 10
    This is undecor func

    # equivalent to
    >>> def undecor():
    ...   print "This is undecor func"
    ...
    >>> d = example(10)
    >>> undecor = d(undecor)
    >>> undecor()
    Val is 10
    This is undecor func

for: exp else: exp
------------------

.. code-block:: python

    # see document: More Control Flow Tools
    # forloop’s else clause runs when no break occurs
    >>> for _ in range(5):
    ...   print _,
    ... else:
    ...   print "\nno break occur"
    ...
    0 1 2 3 4
    no break occur
    >>> for _ in range(5):
    ...   if _ % 2 ==0:
    ...     print "break occur"
    ...     break
    ... else:
    ...   print "else not occur"
    ...
    break occur

    # above statement equivalent to
    flag = False
    for _ in range(5):
        if _ % 2 == 0:
            flag = True
            print "break occur"
            break
    if flag == False:
        print "else not occur"

try: exp else: exp
------------------

.. code-block:: python

    # No exception occur will go into else.
    >>> try:
    ...   print "No exception"
    ... except:
    ...   pass
    ... else:
    ...   print "No exception occur"
    ...
    No exception
    No exception occur

Lambda function
---------------

.. code-block:: python

    >>> fn = lambda x: x**2
    >>> fn(3)
    9
    >>> (lambda x:x**2)(3)
    9
    >>> (lambda x: [x*_ for _ in range(5)])(2)
    [0, 2, 4, 6, 8]
    >>> (lambda x: x if x>3 else 3)(5)
    5

    # multiline lambda example
    >>> (lambda x:
    ... True
    ... if x>0
    ... else
    ... False)(3)
    True

Option arguments - (\*args, \*\*kwargs)
---------------------------------------

.. code-block:: python

    >>> def example(a,b=None,*args,**kwargs):
    ...   print a, b
    ...   print args
    ...   print kwargs
    ...
    >>> example(1,"var",2,3,word="hello")
    1 var
    (2, 3)
    {'word': 'hello'}
    >>> _args = (1,2,3,4,5)
    >>> _kwargs = {"1":1,"2":2,"3":3}
    >>> example(1,"var",*_args,**_kwargs)
    1 var
    (1, 2, 3, 4, 5)
    {'1': 1, '3': 3, '2': 2}

``type()`` declare (create) a ``class``
----------------------------------------

.. code-block:: python

    >>> def fib(self, n):
    ...     if n <= 2:
    ...         return 1
    ...     return fib(self, n-1) + fib(self, n-2)
    ...
    >>> Fib = type('Fib', (object,), {'val': 10,
    ...                               'fib': fib})
    >>> f = Fib()
    >>> f.val
    10
    >>> f.fib(f.val)
    55

    # equal to
    >>> class Fib(object):
    ...     val = 10
    ...     def fib(self, n):
    ...         if n <=2:
    ...             return 1
    ...         return self.fib(n-1)+self.fib(n-2)
    ...
    >>> f = Fib()
    >>> f.val
    10
    >>> f.fib(f.val)
    55


Callable object
---------------

.. code-block:: python

    >>> class calobj(object):
    ...   def example(self):
    ...     print "I am callable!"
    ...   def __call__(self):
    ...     self.example()
    ...
    >>> ex = calobj()
    >>> ex()
    I am callable!

Context Manager - "with" statement
----------------------------------

.. code-block:: python

    # replace try: ... finally: ...
    # see: PEP343
    # common use in open and close

    import socket

    class Socket(object):
        def __init__(self,host,port):
            self.host = host
            self.port = port
        def __enter__(self):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((self.host,self.port))
            sock.listen(5)
            self.sock = sock
            return self.sock

        def __exit__(self,*exc_info):
            if exc_info[0] is not None:
                import traceback
                traceback.print_exception(*exc_info)
            self.sock.close()

    if __name__=="__main__":
        host = 'localhost'
        port 5566
        with Socket(host,port) as s:
            while True:
                conn, addr = s.accept()
                msg = conn.recv(1024)
                print msg
                conn.send(msg)
                conn.close()

Using @contextmanager
---------------------

.. code-block:: python

    from contextlib import contextmanager

    @contextmanager
    def opening(filename, mode='r'):
       f = open(filename, mode)
       try:
          yield f
       finally:
          f.close()

    with opening('example.txt','r') as fd:
       fd.read()

Using "with" statement open file
--------------------------------

.. code-block:: python

    >>> with open("/etc/passwd",'r') as f:
    ...    content = f.read()

Property - Managed attributes
-----------------------------

.. code-block:: python

    >>> class example(object):
    ...     def __init__(self,value):
    ...        self._val = value
    ...     @property
    ...     def val(self):
    ...         return self._val
    ...     @val.setter
    ...     def val(self,value):
    ...         if not isintance(value,int):
    ...             raise TypeError("Expect int")
    ...         self._val = value
    ...     @val.deleter
    ...     def val(self):
    ...         del self._val
    ...
    >>> ex = example(123)
    >>> ex.val = "str"
    Traceback (most recent call last):
      File "", line 1, in
      File "test.py", line 12, in val
        raise TypeError("Expect int")
    TypeError: Expect int

Computed attributes - Using property
------------------------------------

Concept: Attribute's value is not store in memory. Computing the value only
when we need.

.. code-block:: python

    >>> class example(object):
    ...   @property
    ...   def square3(self):
    ...     return 2**3
    ...
    >>> ex = example()
    >>> ex.square3
    8

Descriptor - manage attributes
------------------------------

.. code-block:: python

    >>> class Integer(object):
    ...   def __init__(self,name):
    ...     self._name = name
    ...   def __get__(self,inst,cls):
    ...     if inst is None:
    ...       return self
    ...     else:
    ...       return inst.__dict__[self._name]
    ...   def __set__(self,inst,value):
    ...     if not isinstance(value,int):
    ...       raise TypeError("Expected INT")
    ...     inst.__dict__[self._name] = value
    ...   def __delete__(self,inst):
    ...     del inst.__dict__[self._name]
    ...
    >>> class example(object):
    ...   x = Integer('x')
    ...   def __init__(self,val):
    ...     self.x = val
    ...
    >>> ex = example(1)
    >>> ex = example("str")
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "<stdin>", line 4, in __init__
      File "<stdin>", line 11, in __set__
    TypeError: Expected an int

@staticmethod, @classmethod
---------------------------

.. code-block:: python

    # @classmethod: bound to class
    # @staticmethod: like python function but in class
    >>> class example(object):
    ...   @classmethod
    ...   def clsmethod(cls):
    ...     print "I am classmethod"
    ...   @staticmethod
    ...   def stmethod():
    ...     print "I am staticmethod"
    ...   def instmethod(self):
    ...     print "I am instancemethod"
    ...
    >>> ex = example()
    >>> ex.clsmethod()
    I am classmethod
    >>> ex.stmethod()
    I am staticmethod
    >>> ex.instmethod()
    I am instancemethod
    >>> example.clsmethod()
    I am classmethod
    >>> example.stmethod()
    I am staticmethod
    >>> example.instmethod()
    Traceback (most recent call last):
      File "", line 1, in
    TypeError: unbound method instmethod() ...

Abstract method - Metaclass
---------------------------

.. code-block:: python

    # usually using in define methods but not implement
    >>> from abc import ABCMeta, abstractmethod
    >>> class base(object):
    ...   __metaclass__ = ABCMeta
    ...   @abstractmethod
    ...   def absmethod(self):
    ...     """ Abstract method """
    ...
    >>> class example(base):
    ...   def absmethod(self):
    ...     print "abstract"
    ...
    >>> ex = example()
    >>> ex.absmethod()
    abstract

    # another better way to define a meta class
    >>> class base(object):
    ...   def absmethod(self):
    ...     raise NotImplementedError
    ...
    >>> class example(base):
    ...   def absmethod(self):
    ...     print "abstract"
    ...
    >>> ex = example()
    >>> ex.absmethod()
    abstract

Common Use "Magic"
------------------

.. code-block:: python

    # see python document: data model
    # For command class
    __main__
    __name__
    __file__
    __module__
    __all__
    __dict__
    __class__
    __doc__
    __init__(self, [...)
    __str__(self)
    __repr__(self)
    __del__(self)

    # For Descriptor
    __get__(self, instance, owner)
    __set__(self, instance, value)
    __delete__(self, instance)

    # For Context Manager
    __enter__(self)
    __exit__(self, exc_ty, exc_val, tb)

    # Emulating container types
    __len__(self)
    __getitem__(self, key)
    __setitem__(self, key, value)
    __delitem__(self, key)
    __iter__(self)
    __contains__(self, value)

    # Controlling Attribute Access
    __getattr__(self, name)
    __setattr__(self, name, value)
    __delattr__(self, name)
    __getattribute__(self, name)

    # Callable object
    __call__(self, [args...])

    # Compare related
    __cmp__(self, other)
    __eq__(self, other)
    __ne__(self, other)
    __lt__(self, other)
    __gt__(self, other)
    __le__(self, other)
    __ge__(self, other)

    # arithmetical operation related
    __add__(self, other)
    __sub__(self, other)
    __mul__(self, other)
    __div__(self, other)
    __mod__(self, other)
    __and__(self, other)
    __or__(self, other)
    __xor__(self, other)
