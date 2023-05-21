from multiprocessing import Process
from client import start_client
import time
from server import NUM_CLIENTS

def run_simulation():

    # This will hold all the processes which we are going to create
    processes = []

    # Start all the clients
    for id in range(NUM_CLIENTS):
        client_process = Process(target=start_client, args=(id, ))
        client_process.start()
        processes.append(client_process)
        time.sleep(3)

    # Block until all processes are finished
    for p in processes:
        p.join()


if __name__ == "__main__":
    run_simulation()