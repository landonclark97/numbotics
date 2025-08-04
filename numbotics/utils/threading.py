from sys import platform
from typing import Any
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import threading
from itertools import repeat


if platform == 'darwin':
    mp.set_start_method('fork')


_GLOBALS = {}



def cpu_count():
    return mp.cpu_count()



class ResourceThreadPool:

    def __init__(self, poolsize: int = cpu_count(), per_thread_resources: list[Any] | None = None):
        self._poolsize = poolsize
        if per_thread_resources is not None:
            if len(per_thread_resources) != poolsize:
                raise ValueError("per_thread_resources must be a list of length equal to poolsize, i.e. each thread receieves one resource")
            self._per_thread_resources = [per_thread_resources.copy()]
        else:
            raise ValueError("per_thread_resources must be a list of resources equal in length to poolsize, i.e. each thread receieves one resource")

        self._executor = ThreadPoolExecutor(
            max_workers=poolsize,
            initializer=self._per_thread_init,
            initargs=self._per_thread_resources
        )
        self._all_tids = []


    def __enter__(self):
        return self
    

    def __exit__(self, exc_type, exc_value, traceback):
        global _GLOBALS
        self._executor.shutdown(wait=True)
        for tid in self._all_tids:
            del _GLOBALS[tid]
        self._all_tids = []


    def _per_thread_init(self, resources):
        global _GLOBALS
        tid = threading.get_native_id()
        _GLOBALS[tid] = resources.pop()
        self._all_tids.append(tid)


    def func_wrapper(self, args):
        global _GLOBALS
        func, args = args[0], args[1:]
        tid = threading.get_native_id()
        args = (_GLOBALS[tid], *args)
        return func(*args)


    def map(self, func, *args):
        return list(self._executor.map(self.func_wrapper, zip(repeat(func), *args)))