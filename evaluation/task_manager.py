import multiprocessing as mp
# from https://stackoverflow.com/questions/9601802/python-pool-apply-async-and-map-async-do-not-block-on-full-queue?rq=1
from threading import Semaphore
from multiprocessing import Pool


def get_num_processes(num_processes):
    if num_processes == -1:
        num_processes = mp.cpu_count()
    elif num_processes > 0:
        num_processes = min(num_processes, mp.cpu_count())
    else:
        num_processes = 0
    return num_processes


class TaskManager(object):
    def __init__(self, processes, queue_size, callback):
        self.pool = Pool(processes=get_num_processes(processes))
        self.workers = Semaphore(processes + queue_size)
        self.callback = callback

    def new_task(self, function, *args, **kwargs):
        """Start a new task, blocks if queue is full."""
        self.workers.acquire()
        self.pool.apply_async(function, args, kwargs, callback=self.task_done, error_callback=self.error_callback)

    def task_done(self, *args, **kwargs):
        """Called once task is done, releases the queue is blocked."""
        self.workers.release()
        if self.callback is not None:
            self.callback(*args, **kwargs)
    
    def error_callback(self, e):
        self.workers.release()
        print(e)

    def close(self):
        self.pool.close()
        self.pool.join()
        