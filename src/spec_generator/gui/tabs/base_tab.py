import queue
from tkinter import ttk
import random
from ...utils.ui_helpers import ScrollableFrame
from ...config import CommonParams, LCParams

class BaseTab(ttk.Frame):
    """
    A base class for all tabs in the notebook, providing common structure.
    """
    def __init__(self, master, style, app_queue: queue.Queue, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.style = style
        self.app_queue = app_queue
        self.main_frame = ScrollableFrame(self)
        self.main_frame.pack(fill="both", expand=True)
        self.content_frame = self.main_frame.scrollable_frame

        # These widgets are expected by the main app's queue processor
        self.output_text = None
        self.progress_bar = None

        self.create_widgets()

    def create_widgets(self):
        """
        A placeholder method that subclasses must override to create their widgets.
        """
        raise NotImplementedError

    def _gather_common_params(self, params_dict, lc_params_dict):
        seed_str = params_dict['seed_var'].get().strip()
        if seed_str:
            seed = int(seed_str)
        else:
            seed = random.randint(0, 2**32 - 1)
            params_dict['seed_var'].set(str(seed))

        common = CommonParams(
            isotopic_enabled=params_dict['isotopic_enabled_var'].get(),
            resolution=float(params_dict['resolution_entry'].get()) * 1000,
            peak_sigma_mz=float(params_dict['peak_sigma_mz_entry'].get()),
            mz_step=float(params_dict['mz_step_entry'].get()),
            mz_range_start=float(params_dict['mz_range_start_entry'].get()),
            mz_range_end=float(params_dict['mz_range_end_entry'].get()),
            noise_option=params_dict['noise_option_var'].get(),
            pink_noise_enabled=params_dict['pink_noise_enabled_var'].get(),
            output_directory=params_dict['output_directory_var'].get(),
            seed=seed,
            filename_template=params_dict['filename_template_var'].get(),
        )

        lc_enabled = lc_params_dict['enabled_var'].get()
        lc = LCParams(
            enabled=lc_enabled,
            num_scans=int(lc_params_dict['num_scans_entry'].get()) if lc_enabled else 1,
            scan_interval=float(lc_params_dict['scan_interval_entry'].get()) if lc_enabled else 0.0,
            gaussian_std_dev=float(lc_params_dict['gaussian_std_dev_entry'].get()) if lc_enabled else 0.0,
            lc_tailing_factor=float(lc_params_dict['lc_tailing_factor_entry'].get()) if lc_enabled else 0.0,
        )
        return common, lc

    def get_log_widgets(self):
        """
        Returns the output text and progress bar widgets for this tab.
        This allows the main app to route messages to the correct tab.
        """
        return self.output_text, self.progress_bar

    def on_task_done(self):
        """
        Called when a background task for this tab is finished.
        Subclasses should implement this to re-enable buttons etc.
        """
        pass
