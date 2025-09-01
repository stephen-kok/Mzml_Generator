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
            final_filepath=output_filepath
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

    def test_peptide_map_content_is_correct(self):
        """
        Test that the peptide map simulation generates peaks for the correct
        digested peptides at their expected m/z values.
        """
        import numpy as np
        from pyteomics import mass

        common_params = CommonParams(
            isotopic_enabled=True, resolution=120000, peak_sigma_mz=0.0,
            mz_step=0.01, mz_range_start=50.0, mz_range_end=1000.0,
            noise_option="No Noise", pink_noise_enabled=False,
            output_directory=self.test_dir, seed=123, filename_template=""
        )
        lc_params = PeptideMapLCParams(run_time=1.0, scan_interval=1.0, peak_width_seconds=30.0)
        config = PeptideMapSimConfig(
            common=common_params, lc=lc_params, sequence="PEPTIDERKTEST",
            missed_cleavages=0, charge_state=2
        )

        mz_range, final_scans = execute_peptide_map_simulation(
            config=config, final_filepath="", return_data_only=True
        )

        # Combine all scans to get the total signal
        total_signal = np.sum(final_scans, axis=0)

        expected_peptides = ["PEPTIDER", "K", "TEST"]
        for peptide in expected_peptides:
            expected_mz = mass.calculate_mass(sequence=peptide, charge=config.charge_state)
            idx = np.abs(mz_range - expected_mz).argmin()
            self.assertGreater(
                total_signal[idx], 0,
                msg=f"Expected peak for peptide '{peptide}' at m/z ~{expected_mz:.2f}, but none was found."
            )

    def test_simulation_with_predicted_charges(self):
        """
        Test that the simulation correctly generates multiple charge states
        when the 'predict_charge' option is enabled.
        """
        import numpy as np
        from pyteomics import mass
        from spec_generator.logic.charge import predict_charge_states

        common_params = CommonParams(
            isotopic_enabled=True, resolution=120000, peak_sigma_mz=0.0,
            mz_step=0.01, mz_range_start=200.0, mz_range_end=1000.0,
            noise_option="No Noise", pink_noise_enabled=False,
            output_directory=self.test_dir, seed=123, filename_template=""
        )
        lc_params = PeptideMapLCParams(run_time=1.0, scan_interval=1.0, peak_width_seconds=30.0)

        peptide_sequence = "PEPTIDEK" # One basic residue, should give charge 2 as primary

        config = PeptideMapSimConfig(
            common=common_params, lc=lc_params, sequence=peptide_sequence,
            missed_cleavages=0, charge_state=0, predict_charge=True # charge_state is ignored
        )

        mz_range, final_scans = execute_peptide_map_simulation(
            config=config, final_filepath="", return_data_only=True
        )

        total_signal = np.sum(final_scans, axis=0)

        # Get the expected charge distribution
        expected_distribution = predict_charge_states(peptide_sequence)
        self.assertGreater(len(expected_distribution), 1, "Expected more than one charge state to be predicted.")

        # Find the intensity at the m/z for each predicted charge state
        found_intensities = {}
        for charge, rel_intensity in expected_distribution.items():
            expected_mz = mass.calculate_mass(sequence=peptide_sequence, charge=charge)
            idx = np.abs(mz_range - expected_mz).argmin()
            # Check a small window around the peak
            peak_intensity = np.sum(total_signal[max(0, idx-2):idx+3])
            self.assertGreater(peak_intensity, 0, f"No peak found for charge {charge} at m/z ~{expected_mz:.2f}")
            found_intensities[charge] = peak_intensity

        # Check that the relative intensities roughly match the prediction
        primary_charge = max(expected_distribution, key=expected_distribution.get)

        for charge, intensity in found_intensities.items():
            if charge != primary_charge:
                # The intensity of secondary peaks should be less than the primary one
                self.assertLess(intensity, found_intensities[primary_charge])


    def test_parallel_execution_matches_sequential(self):
        """
        Verify that the parallel peptide map implementation produces the exact
        same output as the original sequential implementation when the same
        seed is used.
        """
        import numpy as np
        from spec_generator.logic.peptide import digest_sequence
        from spec_generator.logic.retention_time import predict_retention_times
        from spec_generator.core.spectrum import generate_peptide_spectrum
        from spec_generator.logic.charge import predict_charge_states
        from spec_generator.core.lc import _get_lc_peak_shape

        # Use a smaller configuration to make the test faster
        common_params = CommonParams(
            isotopic_enabled=True, resolution=10000, peak_sigma_mz=0.0,
            mz_step=0.1, mz_range_start=400.0, mz_range_end=800.0, # Smaller range
            noise_option="No Noise", pink_noise_enabled=False,
            output_directory=self.test_dir, seed=42, filename_template=""
        )
        lc_params = PeptideMapLCParams(run_time=0.5, scan_interval=1.0, peak_width_seconds=10.0) # Shorter run
        config = PeptideMapSimConfig(
            common=common_params, lc=lc_params, sequence="TESTK", # Shorter sequence
            missed_cleavages=0, predict_charge=True, charge_state=0
        )

        # 1. Get the result from the parallel implementation
        mz_range_parallel, final_scans_parallel = execute_peptide_map_simulation(
            config=config, final_filepath="", return_data_only=True
        )

        # 2. Manually run the sequential logic to get the expected result
        peptides = digest_sequence(config.sequence, missed_cleavages=config.missed_cleavages)
        retention_times = predict_retention_times(peptides, total_run_time_minutes=config.lc.run_time)
        mz_range = np.arange(config.common.mz_range_start, config.common.mz_range_end, config.common.mz_step)
        peak_sigma_mz = config.common.mz_range_end / (config.common.resolution * 2.355)

        peptide_data_sequential = []
        for peptide in peptides:
            total_spectrum = np.zeros_like(mz_range)
            base_intensity = 1000.0
            if config.predict_charge:
                charge_distribution = predict_charge_states(peptide)
                for charge, rel_intensity in charge_distribution.items():
                    spectrum_for_charge = generate_peptide_spectrum(
                        peptide_sequence=peptide, mz_range=mz_range, peak_sigma_mz_float=peak_sigma_mz,
                        intensity_scalar=base_intensity * rel_intensity, resolution=config.common.resolution,
                        charge=charge
                    )
                    total_spectrum += spectrum_for_charge
            else:
                total_spectrum = generate_peptide_spectrum(
                    peptide_sequence=peptide, mz_range=mz_range, peak_sigma_mz_float=peak_sigma_mz,
                    intensity_scalar=base_intensity, resolution=config.common.resolution,
                    charge=config.charge_state
                )
            peptide_data_sequential.append({
                "sequence": peptide, "rt": retention_times[peptide], "spectrum": total_spectrum
            })

        scan_interval_minutes = config.lc.scan_interval / 60.0
        num_scans = int(config.lc.run_time / scan_interval_minutes)
        final_scans_sequential = [np.zeros_like(mz_range) for _ in range(num_scans)]
        lc_peak_scans = int(config.lc.peak_width_seconds / config.lc.scan_interval)
        apex_index = (lc_peak_scans - 1) / 2.0
        std_dev_scans = lc_peak_scans / 6
        tau = 1.0
        lc_shape = _get_lc_peak_shape(lc_peak_scans, apex_index, std_dev_scans, tau)
        for p_data in peptide_data_sequential:
            apex_scan = int(p_data["rt"] / scan_interval_minutes)
            start_scan = apex_scan - (lc_peak_scans // 2)
            for i in range(lc_peak_scans):
                scan_idx = start_scan + i
                if 0 <= scan_idx < num_scans:
                    final_scans_sequential[scan_idx] += p_data["spectrum"] * lc_shape[i]

        # 3. Compare the results
        self.assertEqual(len(final_scans_parallel), len(final_scans_sequential))
        for i in range(len(final_scans_parallel)):
            np.testing.assert_allclose(
                final_scans_parallel[i],
                final_scans_sequential[i],
                rtol=1e-7,
                atol=1e-7,
                err_msg=f"Scan {i} does not match between parallel and sequential execution."
            )


if __name__ == '__main__':
    unittest.main()
