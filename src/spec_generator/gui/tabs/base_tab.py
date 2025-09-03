import queue
from tkinter import ttk, messagebox
import random
from ...utils.ui_helpers import ScrollableFrame
from ...config import CommonParams, LCParams

class BaseTab(ttk.Frame):
    """
    A base class for all tabs in the notebook, providing common structure.
    """
    def __init__(self, master, style, app_controller=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.style = style
        self.app_controller = app_controller
        self.task_queue = queue.Queue()
        self.main_frame = ScrollableFrame(self)
        self.main_frame.pack(fill="both", expand=True)
        self.content_frame = self.main_frame.scrollable_frame

        # These widgets are expected by the main app's queue processor
        self.output_text = None
        self.progress_bar = None

        self.create_widgets()
        self._process_task_queue()

    def create_widgets(self):
        """
        A placeholder method that subclasses must override to create their widgets.
        """
        raise NotImplementedError

    def _gather_common_params(self, params_dict, lc_params_dict=None):
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

        lc = None
        if lc_params_dict:
            lc_enabled = lc_params_dict['enabled_var'].get()
            lc = LCParams(
                enabled=lc_enabled,
                num_scans=int(lc_params_dict['num_scans_entry'].get()) if lc_enabled else 1,
                scan_interval=float(lc_params_dict['scan_interval_entry'].get()) if lc_enabled else 0.0,
                gaussian_std_dev=float(lc_params_dict['gaussian_std_dev_entry'].get()) if lc_enabled else 0.0,
                lc_tailing_factor=float(lc_params_dict['lc_tailing_factor_entry'].get()) if lc_enabled else 0.0,
            )
        return common, lc

    def on_task_done(self):
        """
        Called when a background task for this tab is finished.
        Subclasses should implement this to re-enable buttons etc.
        """
        pass

    def _process_task_queue(self):
        """
        Process messages from the tab-specific queue.
        """
        try:
            while True:
                msg_type, msg_data = self.task_queue.get_nowait()

                if msg_type == 'log':
                    if self.output_text:
                        self.output_text.insert("end", msg_data)
                        self.output_text.see("end")
                elif msg_type == 'clear_log':
                    if self.output_text:
                        self.output_text.delete('1.0', "end")
                elif msg_type == 'progress_set':
                    if self.progress_bar:
                        self.progress_bar["value"] = msg_data
                elif msg_type == 'progress_add':
                    if self.progress_bar:
                        self.progress_bar["value"] += msg_data
                elif msg_type == 'progress_max':
                    if self.progress_bar:
                        self.progress_bar["maximum"] = msg_data
                elif msg_type == 'error':
                    messagebox.showerror("Error", msg_data)
                    if self.progress_bar:
                        self.progress_bar["value"] = 0
                elif msg_type == 'warning':
                    messagebox.showwarning("Warning", msg_data)
                elif msg_type == 'done':
                    # First, call the specific tab's on_task_done
                    self.on_task_done()
                    # Then, show a generic message if one was provided
                    if msg_data:
                        messagebox.showinfo("Complete", msg_data)
                elif msg_type == 'preview_done':
                    if hasattr(self, 'on_preview_done'):
                        self.on_preview_done()

        except queue.Empty:
            pass
        finally:
            self.after(100, self._process_task_queue)
