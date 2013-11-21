class LazyProperty(object):
    def __init__(self, func):
        self._func = func
        self.__name__ = func.__name__

    def __get__(self, obj, klass):
        result = self._func(obj)
        obj.__dict__[self.__name__] = result
        return result
