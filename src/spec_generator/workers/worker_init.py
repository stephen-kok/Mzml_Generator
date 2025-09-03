import threading

# Global event for stopping multiprocessing pools, shared across worker processes
_stop_event_worker = None

def init_worker(stop_event: threading.Event):
    """
    Initializer for the multiprocessing.Pool, setting a global stop event
    for that worker process.
    """
    global _stop_event_worker
    _stop_event_worker = stop_event
