import unittest
import os
import tempfile
import shutil

from spec_generator.config import PeptideMapSimConfig, PeptideMapLCParams, CommonParams
from spec_generator.logic.peptide_map import execute_peptide_map_simulation

class TestPeptideMapSimulation(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory for test outputs."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory after tests."""
        shutil.rmtree(self.test_dir)

    def test_simulation_runs_without_error(self):
        """
        Test that a full peptide map simulation runs to completion and creates a
        non-empty mzML file.
        """
        common_params = CommonParams(
            isotopic_enabled=True,
            resolution=120000,
            peak_sigma_mz=0.0,
            mz_step=0.01,
            mz_range_start=200.0,
            mz_range_end=2000.0,
            noise_option="No Noise",
            pink_noise_enabled=False,
            output_directory=self.test_dir,
            seed=123,
            filename_template="test_pepmap.mzML"
        )

        lc_params = PeptideMapLCParams(
            run_time=10.0, # Short run time for testing
            scan_interval=1.0,
            peak_width_seconds=30.0
        )

        config = PeptideMapSimConfig(
            common=common_params,
            lc=lc_params,
            sequence="TESTPEPTIDEKONEK",
            missed_cleavages=1,
            charge_state=2
        )

        output_filepath = os.path.join(self.test_dir, "test_pepmap.mzML")

        # We don't have a queue in the test environment, so we pass None
        success = execute_peptide_map_simulation(
            config=config,
            final_filepath=output_filepath,
            update_queue=None
        )

        # 1. Check that the simulation reports success
        self.assertTrue(success)

        # 2. Check that the output file was created
        self.assertTrue(os.path.exists(output_filepath))

        # 3. Check that the file is not empty
        self.assertGreater(os.path.getsize(output_filepath), 0)

    def test_return_data_only(self):
        """
        Test the 'return_data_only' flag to ensure it returns data instead of
        writing a file.
        """
        common_params = CommonParams(
            isotopic_enabled=True, resolution=120000, peak_sigma_mz=0.0,
            mz_step=0.1, mz_range_start=200.0, mz_range_end=2000.0,
            noise_option="No Noise", pink_noise_enabled=False,
            output_directory=self.test_dir, seed=123, filename_template=""
        )
        lc_params = PeptideMapLCParams(run_time=1.0, scan_interval=1.0, peak_width_seconds=30.0)
        config = PeptideMapSimConfig(
            common=common_params, lc=lc_params, sequence="TESTK",
            missed_cleavages=0, charge_state=1
        )

        result = execute_peptide_map_simulation(
            config=config,
            final_filepath="",
            update_queue=None,
            return_data_only=True
        )

        # Check that the result is a tuple (mz_range, run_data)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        # Check that run_data is in the expected format
        mz_range, final_scans = result
        self.assertIsInstance(final_scans, list)
        # final_scans is a list of numpy arrays
        import numpy as np
        self.assertIsInstance(final_scans[0], np.ndarray)
        num_scans = int(lc_params.run_time * 60 / lc_params.scan_interval)
        self.assertEqual(len(final_scans), num_scans)

if __name__ == '__main__':
    unittest.main()
