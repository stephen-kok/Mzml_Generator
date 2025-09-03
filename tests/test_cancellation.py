import unittest
import time
import threading
import multiprocessing
import os
from unittest.mock import MagicMock

from spec_generator.logic.simulation import execute_simulation_and_write_mzml
from spec_generator.logic.binding import execute_binding_simulation
from spec_generator.logic.antibody import execute_antibody_simulation
from spec_generator.logic.peptide_map import execute_peptide_map_simulation
from spec_generator.config import (
    SpectrumGeneratorConfig,
    CovalentBindingConfig,
    AntibodySimConfig,
    PeptideMapSimConfig,
    CommonParams,
    LCParams,
    PeptideMapLCParams,
    Chain
)

# A long sequence to make peptide map and antibody sims take time
LONG_SEQUENCE = "QVTLRESGPALVKPTQTLTLTCTFSGFSLSTAGMSVGWIRQPPGKALEWLADIWWDDKKHYNPSLKDRLTISKDTSKNQVVLKVTNMDPADTATYYCARDMIFNFYFDVWGQGTTVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"

class TestCancellation(unittest.TestCase):
    def setUp(self):
        """Set up common parameters for tests."""
        self.output_dir = "test_output_cancel"
        os.makedirs(self.output_dir, exist_ok=True)

        self.common_params = CommonParams(
            output_directory=self.output_dir,
            filename_template="{protein_mass}.mzML",
            mz_range_start=400.0,
            mz_range_end=800.0,
            mz_step=0.1,
            peak_sigma_mz=0.1,
            isotopic_enabled=True, # Make it a bit slower
            resolution=10000,
            noise_option="No Noise",
            pink_noise_enabled=False,
            seed=42
        )
        self.lc_params = LCParams(
            enabled=True,
            num_scans=100, # A decent number of scans to allow for cancellation
            scan_interval=0.2,
            gaussian_std_dev=3.0,
            lc_tailing_factor=1.1,
        )

    def test_simulation_cancellation(self):
        """Test that a standard simulation can be cancelled."""
        config = SpectrumGeneratorConfig(
            common=self.common_params,
            lc=self.lc_params,
            protein_masses=[25000, 26000], # Multiple proteins to engage the pool
            intensity_scalars=[1.0, 1.0],
            protein_list_file=None,
            mass_inhomogeneity=0.0
        )
        stop_event = threading.Event()
        result_queue = multiprocessing.Queue()

        def target():
            result = execute_simulation_and_write_mzml(
                config, os.path.join(self.output_dir, "sim_cancel.mzML"), stop_event=stop_event
            )
            result_queue.put(result)

        thread = threading.Thread(target=target)
        thread.start()

        time.sleep(0.1)
        stop_event.set()
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive(), "Thread should have terminated after cancellation.")
        self.assertFalse(result_queue.get(), "Function should return False on cancellation.")

    def test_peptide_map_cancellation(self):
        """Test that a peptide map simulation can be cancelled."""
        pepmap_lc = PeptideMapLCParams(run_time=5, scan_interval=0.5, peak_width_seconds=20)
        config = PeptideMapSimConfig(
            common=self.common_params,
            lc=pepmap_lc,
            sequence=LONG_SEQUENCE,
            missed_cleavages=2,
            charge_state=2,
        )
        stop_event = threading.Event()
        result_queue = multiprocessing.Queue()

        def target():
            result = execute_peptide_map_simulation(
                config, os.path.join(self.output_dir, "pepmap_cancel.mzML"), stop_event=stop_event
            )
            result_queue.put(result)

        thread = threading.Thread(target=target)
        thread.start()

        time.sleep(0.2)
        stop_event.set()
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive(), "Process should have terminated.")
        self.assertFalse(result_queue.get(), "Function should return False on cancellation.")

    def test_antibody_cancellation(self):
        """Test that an antibody simulation can be cancelled."""
        config = AntibodySimConfig(
            common=self.common_params,
            lc=self.lc_params,
            chains=[
                Chain(type="HC", name="HC1", seq=LONG_SEQUENCE[:150], pyro_glu=False, k_loss=False, ptms=[]),
                Chain(type="LC", name="LC1", seq=LONG_SEQUENCE[150:250], pyro_glu=False, k_loss=False, ptms=[]),
            ],
            assembly_abundances={"HC1": 1.0, "LC1": 1.0, "HC1LC1": 1.0, "HC1HC1": 1.0, "HC1HC1LC1": 1.0, "HC1HC1LC1LC1":1.0}
        )
        stop_event = threading.Event()
        result_queue = multiprocessing.Queue()

        def target():
            result = execute_antibody_simulation(
                config, os.path.join(self.output_dir, "antibody_cancel.mzML"), stop_event=stop_event
            )
            result_queue.put(result)

        thread = threading.Thread(target=target)
        thread.start()

        time.sleep(0.1)
        stop_event.set()
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive(), "Thread should have terminated.")
        self.assertFalse(result_queue.get(), "Function should return False on cancellation.")


if __name__ == '__main__':
    # Clean up output dir before running
    if os.path.exists("test_output_cancel"):
        import shutil
        shutil.rmtree("test_output_cancel")
    unittest.main()
