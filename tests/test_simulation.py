import unittest
import numpy as np
import os
import tempfile
import shutil
import copy

from spec_generator.logic.simulation import execute_simulation_and_write_mzml
from spec_generator.config import SpectrumGeneratorConfig, CommonParams, LCParams

class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.default_common = CommonParams(
            isotopic_enabled=True,
            resolution=120000,
            peak_sigma_mz=0.01,
            mass_dependent_peak_width=False,
            peak_width_scaling_factor=0.0,
            mz_step=0.02,
            mz_range_start=400.0,
            mz_range_end=2500.0,
            noise_option="No Noise",
            pink_noise_enabled=False,
            output_directory=self.test_dir,
            seed=123,
            filename_template="test.mzML"
        )
        self.default_lc = LCParams(
            enabled=False,
            num_scans=1,
            scan_interval=0.0,
            gaussian_std_dev=0.0,
            lc_tailing_factor=0.0
        )
        self.default_config = SpectrumGeneratorConfig(
            common=self.default_common,
            lc=self.default_lc,
            protein_list_file=None,
            protein_masses=[25000.0],
            intensity_scalars=[1.0],
            mass_inhomogeneity=0.0
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_execute_simulation_successful_run(self):
        result = execute_simulation_and_write_mzml(
            config=self.default_config,
            final_filepath=os.path.join(self.test_dir, "test.mzML"),
            update_queue=None,
            return_data_only=True
        )
        self.assertIsInstance(result, tuple)
        mz_range, spectra = result
        self.assertIsInstance(mz_range, np.ndarray)
        self.assertIsInstance(spectra, list)
        self.assertEqual(len(spectra), 1)  # One protein
        self.assertIsInstance(spectra[0], list)
        self.assertEqual(len(spectra[0]), 1)  # One scan
        self.assertIsInstance(spectra[0][0], np.ndarray)
        self.assertEqual(mz_range.shape, spectra[0][0].shape)
        # Check that some signal was generated
        self.assertTrue(np.sum(spectra[0][0]) > 0)

    def test_mass_inhomogeneity_changes_spectrum(self):
        # Use deepcopy to avoid modifying the default config
        config_inhomogeneity = copy.deepcopy(self.default_config)
        config_inhomogeneity.mass_inhomogeneity = 5.0
        config_inhomogeneity.common.seed = 42 # Use fixed seed

        _, spectra_inhomogeneity = execute_simulation_and_write_mzml(
            config=config_inhomogeneity,
            final_filepath=os.path.join(self.test_dir, "test_inhomogeneity.mzML"),
            update_queue=None,
            return_data_only=True
        )

        config_no_inhomogeneity = copy.deepcopy(self.default_config)
        config_no_inhomogeneity.common.seed = 42 # Use same fixed seed

        _, spectra_no_inhomogeneity = execute_simulation_and_write_mzml(
            config=config_no_inhomogeneity,
            final_filepath=os.path.join(self.test_dir, "test_no_inhomogeneity.mzML"),
            update_queue=None,
            return_data_only=True
        )

        # The spectra should not be identical
        self.assertFalse(np.array_equal(spectra_inhomogeneity[0][0], spectra_no_inhomogeneity[0][0]))

    def test_invalid_mass_raises_error(self):
        config = copy.deepcopy(self.default_config)
        config.protein_masses = [-100]
        # execute_simulation_and_write_mzml returns False on validation error, it doesn't raise
        result = execute_simulation_and_write_mzml(config, "test.mzML", None)
        self.assertFalse(result)

    def test_file_is_created(self):
        filepath = os.path.join(self.test_dir, "test.mzML")
        success = execute_simulation_and_write_mzml(
            config=self.default_config,
            final_filepath=filepath,
            update_queue=None,
            return_data_only=False
        )
        self.assertTrue(success)
        self.assertTrue(os.path.exists(filepath))

if __name__ == '__main__':
    unittest.main()
