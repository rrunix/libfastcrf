from threading import Thread, Lock
from tqdm import tqdm
import time
from filelock import FileLock
import multiprocessing
import sys


def get_multiprocessing_id():
    if len(multiprocessing.current_process()._identity) > 0:
        position = multiprocessing.current_process()._identity[0] + 1
    else:
        position = 0
    return position


def get_desc(experiment_method_manager):
    return experiment_method_manager.experiment_manager.dataset_name + '_' + \
               experiment_method_manager.method_name + '_' + \
               experiment_method_manager.experiment_manager.cv


class MultiprocessTQDM(Thread):

    def __init__(self, size, experiment_method_manager=None, desc=None, position=None, lock_file='research_paper/tmp/lock',
                 sleep_for=1):
        super(MultiprocessTQDM, self).__init__()
        self.size = size
        self.lock_file = lock_file
        self.updated = 0
        self.lock = Lock()
        self.running = True
        self.sleep_for = sleep_for

        position = position or get_multiprocessing_id()

        desc = desc or get_desc(experiment_method_manager)
        self.pbar = tqdm(total=size, desc=desc, position=position)

    def update(self):
        with self.lock:
            self.updated += 1

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.pbar.close()

    def run(self):
        while self.running or self.updated > 0:
            with self.lock:
                n = self.updated
                self.updated = 0

            if n > 0:
                with FileLock(self.lock_file):
                    self.pbar.update(n)
                    self.pbar.refresh()

            time.sleep(self.sleep_for)


if sys.version_info[0] == 2:
    from contextlib import contextmanager
    @contextmanager
    def multiprocessing_context(*args, **kwargs):
        pool = multiprocessing.Pool(*args, **kwargs)
        yield pool
        pool.terminate()
else:
    multiprocessing_context = multiprocessing.Pool