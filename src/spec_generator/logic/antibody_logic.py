import threading

from ..config import AntibodySimConfig
from .antibody import execute_antibody_simulation

class AntibodyTabLogic:
    def start_plot_generation(self, config: AntibodySimConfig, task_queue, callback):
        """
        Starts the plot generation in a new thread.
        """
        threading.Thread(
            target=self._worker_generate_and_plot,
            args=(config, task_queue, callback),
            daemon=True
        ).start()

    def _worker_generate_and_plot(self, config: AntibodySimConfig, task_queue, callback):
        """
        The actual worker function that runs in a thread.
        It runs the full antibody simulation and returns the data for plotting.
        """
        try:
            # The execute function handles its own logging to the queue
            result = execute_antibody_simulation(
                config=config,
                final_filepath="", # Not used when returning data
                update_queue=task_queue,
                return_data_only=True
            )
            task_queue.put(('callback', (callback, result)))

        except Exception as e:
            task_queue.put(('error', f"An error occurred during plot generation: {e}"))
            task_queue.put(('callback', (callback, None)))
