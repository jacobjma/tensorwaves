class CallCounter(object):
    def __init__(self, func):
        self.n = 0
        self._func = func

    def func_caller(self):
        self.n += 1
        return self._func()
