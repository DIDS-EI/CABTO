
import omnigibson as og
from omnigibson.simulator import _launch_app

og.app = _launch_app()
import multiprocessing as mp
import time

def simulator_process():
    print("Simulator process is started")
    time.sleep(1000000)

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    process = mp.Process(target=simulator_process)
    process.start()

    process.join()