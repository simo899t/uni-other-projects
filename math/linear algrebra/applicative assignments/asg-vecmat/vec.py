def getitem(v: 'Vec', k: int) -> int:
    """
    Return the value of entry k in vector v.
    Entries start from zero

    return

    >>> v = Vec([1,2,3,4])
    >>> v[2]
    3
    >>> v[0]
    1
    """
    return v.store[k]


def setitem(v: 'Vec', k: int, val: int) -> None:
    """
    Set the element of v with index k to be val.
    The function should only set the value of elements already in the vector.
    It cannot extend the vector.

    >>> v = Vec([1,2,3])
    >>> v[2] = 4
    >>> v[2]
    4
    >>> v[0] = 3
    >>> v[0]
    3
    >>> v[1] = 0
    >>> v[1]
    0
    """
    v.store[k] = val


def equal(u: 'Vec', v: 'Vec') -> bool:
    """
    Checks if two vectors are equal. (both in length and entries)

    >>> Vec([1,2,3]) == Vec([1,2,3])
    True
    >>> Vec([0,0,0]) == Vec([0,0,0])
    True

    """
    return (u.size == v.size and all (u[i] == v[i] for i in range(u.size)))


def add(u: 'Vec', v: 'Vec') -> 'Vec':
    """
    Returns a vector that is the sum of the two vectors.
    """
    assert u.size == v.size, "incompatible sizes. must be the same"
    return Vec([u[i] + v[i] for i in range(u.size)])


def dot(u: 'Vec', v: 'Vec') -> int:
    """
    Returns the dot product of the two vectors.

    >>> u1 = Vec([1, 2])
    >>> u2 = Vec([1, 2])
    >>> u1*u2
    5
    >>> u1 == Vec([1, 2])
    True
    >>> u2 == Vec([1, 2])
    True

    """
    assert u.size == v.size, "incompatible sizes. must be the same"
    return sum(u[i] * v[i] for i in range(u.size))


def scalar_mul(v: 'Vec', alpha: int) -> 'Vec':
    """
    Returns the scalar-vector product alpha times v.

    >>> zero = Vec([0, 0, 0, 0])
    >>> u = Vec([1, 2, 3, 4])
    >>> 0*u == zero
    True
    >>> 1*u == u
    True
    >>> 0.5*u == Vec([0.5, 1, 1.5, 2])
    True
    >>> u == Vec([1, 2, 3, 4])
    True
    """
    return Vec([alpha * v[i] for i in range(v.size)])


def neg(v: 'Vec') -> 'Vec':
    """
    Returns the negation of a vector.

    >>> u = Vec([1, 2, 3, 4])
    >>> -u
    Vec([-1, -2, -3, -4], 4)
    >>> u == Vec([1, 2, 3, 4])
    True
    >>> -Vec([1, 2]) == Vec([-1, -2])
    True
    """
    return Vec([-v[i] for i in range(v.size)])

###############################################################################################################################


class Vec:
    """
    A vector has two attributes:
    store - the list containing the data
    size - the size of the vector
    """

    def __init__(self, data):
        """
        Initialize a vector (as a list) with the given data.
        
        >>> v = Vec([10, 20, 30, 40])

        or 

        >>> w = Vec((1,2,3))
        """
        self.store = list(data)
        self.size = len(self.store)

    __getitem__ = getitem
    __setitem__ = setitem
    __neg__ = neg
    __rmul__ = scalar_mul  # if left arg of * is primitive, assume it's a scalar

    def __mul__(self, other):
        # If other is a vector, returns the dot product of self and other
        if isinstance(other, Vec):
            return dot(self, other)
        else:
            return scalar_mul(self, other)  # scalar multiplication

    def __truediv__(self, other):  # Scalar division
        assert other != 0, "division by zero is not allowed"
        return scalar_mul(self, 1/other)

    __add__ = add

    def __radd__(self, other):
        "Hack to allow sum(...) to work with vectors"
        if other == 0:
            return self

    def __sub__(a: 'Vec', b: 'Vec') -> 'Vec':
        "Returns a vector which is the difference of a and b."
        return a + (-b)

    __eq__ = equal

    def __str__(self):
        return f"[{', '.join(map(str, self.store))}]"

    def __repr__(self):
        "used when just typing >>> v"
        return "Vec(" + str(self.store) + ", " + str(self.size) + ")"

    def copy(self):
        "Don't make a new copy of the domain D"
        return Vec(self.store.copy())

    def __iter__(self):
        raise TypeError('%r object is not iterable' % self.__class__.__name__)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
