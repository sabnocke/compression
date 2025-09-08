import time
from itertools import tee, filterfalse
from typing import Callable, Iterable
from functools import wraps

def partition[T](predicate: Callable[[T], bool], iterable: Iterable[T]):
    """
    Use a predicate to partition entries into true entries and false entries.
    """
    t1, t2 = tee(iterable)
    return filter(predicate, t2), filterfalse(predicate, t1)


def measure(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start} seconds")
        return result
    return wrapper