import unittest
import os
import tempfile
import shutil
import numpy as np
import copy

from spec_generator.logic.binding import execute_binding_simulation
from spec_generator.config import CovalentBindingConfig, CommonParams, LCParams

class TestBinding(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.default_common = CommonParams(
            isotopic_enabled=True, resolution=120000, peak_sigma_mz=0.01,
            mz_step=0.02, mz_range_start=400.0, mz_range_end=2500.0,
            noise_option="No Noise", pink_noise_enabled=False,
            output_directory=self.test_dir, seed=123,
            filename_template="test_binding.mzML"
        )
        self.default_lc = LCParams(
            enabled=True, num_scans=1, scan_interval=0.0,
            gaussian_std_dev=5.0, lc_tailing_factor=0.0
        )
        self.default_config = CovalentBindingConfig(
            common=self.default_common, lc=self.default_lc,
            protein_avg_mass=25000.0, compound_list_file="",
            prob_binding=1.0, prob_dar2=1.0,
            total_binding_range=(50.0, 50.0), dar2_range=(20.0, 20.0)
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_execute_binding_simulation_successful_run(self):
        filepath = os.path.join(self.test_dir, "test.mzML")
        success, final_filepath = execute_binding_simulation(
            config=self.default_config, compound_mass=500.0,
            total_binding_percentage=50.0, dar2_percentage_of_bound=20.0,
            filepath=filepath
        )
        self.assertTrue(success)
        self.assertTrue(os.path.exists(final_filepath))


    def test_binding_produces_different_tic(self):
        """
        Tests that different binding scenarios produce different Total Ion Currents.
        """
        # Scenario 1: No binding
        config_no_binding = copy.deepcopy(self.default_config)
        _, spectra_no_binding = execute_binding_simulation(
            config=config_no_binding, compound_mass=500.0,
            total_binding_percentage=0.0, dar2_percentage_of_bound=0.0,
            filepath="", return_data_only=True
        )
        tic_no_binding = np.sum(spectra_no_binding[0])

        # Scenario 2: 50% binding
        config_binding = copy.deepcopy(self.default_config)
        _, spectra_binding = execute_binding_simulation(
            config=config_binding, compound_mass=500.0,
            total_binding_percentage=50.0, dar2_percentage_of_bound=20.0,
            filepath="", return_data_only=True
        )
        tic_binding = np.sum(spectra_binding[0])

        # The TIC should be different because the mass and charge distribution changes
        self.assertNotAlmostEqual(tic_no_binding, tic_binding, places=0)


if __name__ == '__main__':
    unittest.main()
