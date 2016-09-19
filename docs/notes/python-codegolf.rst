===========================================
Python Obfuscation and Code Golf Cheatsheet
===========================================

Variable Naming and Formatting
------------------------------

.. code-block:: python

    # Join newlines with semicolons.
    >>> x = 5
    >>> y = 5

    >>> x = 5;y = 5


    # Reduce whitespace and newlines wherever possible.
    >>> i = 5
    
    >>> i=5


    >>> for i in range(20):
    >>>     print(i)

    >>> for i in range(20):print(i)


    >>> if a==3:
    >>>     print a

    >>> if a==3:print a


    # Use single-letter variable names.
    # Characters like o, O, and _ are good for obfuscation.
    >>> first = 1
    >>> second = 2
    >>> third = 3

    >>> o=1;O=2;_=3


    # Store built-in functions as variables if they will be used multiple times.
    >>> a=range(10);b=range(25);c=range(45);d=range(75)
    
    >>> r=range;a=r(10);b=r(25);c=r(45)


Built-In Operators
-----------------------

.. code-block:: python

    # Use `backquotes` to convert an integer to a string (only for Python 2)
    >>> x = 123456789
    >>> `x`
    '123456789'


    # Reverse sequences and numbers using [::-1]
    >>> [1, 2, 3, 4][::-1]
    [4, 3, 2, 1]

    >>> "Hello world!"[::-1]
    '!dlrow olleH'


Variable Assignmnet
_______________________

.. code-block:: python

    # Assign repeated variables with a single statement.
    >>> a = 0
    >>> b = 0
    >>> c = 0

    >>> a=b=c=0


    # Use unpacking to assign multiple quantities
    >>> a = range(10)
    >>> b = range(25)
    >>> c = range(75)

    >>> a,b,c=map(range,[10,25,75])


    >>> a,b,c='1','2','3'

    >>> a,b,c='123'


Iteration
-----------------------

.. code-block:: python

    # Use map() to condense list operations using built-ins
    >>> b = [float(i) for i in a]
    
    >>> b=map(float,a)


    # Use a single space for indentation
    >>> for i in range(5):
    >>>     for j in range(5):
    >>>         print(i+j)

    >>> for i in range(5):
    >>>  for j in range(5):
    >>>   print(i+j)


    # Condense nested loops using moduli
    >>> for i in range(a):
    >>>  for j in range(b):
    >>>   print(a,b)

    >>> for x in range(a*b):
    >>>  print(x/b,x%b)


Conditionals
-------------------------

.. code-block:: python

    # Condense serial equalities:
    >>> if i==4 and j==4:
    >>>  print("Both are 4.")

    >>> if i==j==4:
    >>>  print("Both are 4.")

