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

    def test_charge_state_peaks_are_correct(self):
        """
        Test that the simulation generates peaks at the correct m/z values
        for a few expected charge states of a 25kDa protein.
        """
        mz_range, spectra = execute_simulation_and_write_mzml(
            config=self.default_config,
            final_filepath=os.path.join(self.test_dir, "test.mzML"),
            update_queue=None,
            return_data_only=True
        )
        spectrum = spectra[0][0] # First protein, first scan

        PROTON_MASS = 1.007276
        protein_mass = self.default_config.protein_masses[0]

        # Check for a few expected charge states
        for z in [15, 16, 17]:
            expected_mz = (protein_mass + z * PROTON_MASS) / z
            # Find the index in the m/z range that is closest to our expected m/z
            idx = np.abs(mz_range - expected_mz).argmin()

            # We expect a peak (non-zero intensity) at this index
            # This confirms the charge state was generated.
            self.assertGreater(
                spectrum[idx], 0,
                msg=f"Expected a peak for charge state {z} at m/z ~{expected_mz:.2f}, but found none."
            )

    def test_parallel_simulation_matches_sequential(self):
        """
        Verify that the parallel simulation (for both multi-protein and mass
        inhomogeneity) produces the exact same output as the original
        sequential implementation when the same seed is used.
        """
        import numpy as np
        from spec_generator.core.spectrum import generate_protein_spectrum
        from spec_generator.core.lc import apply_lc_profile_and_noise

        # Use a config with multiple proteins and mass inhomogeneity
        config = copy.deepcopy(self.default_config)
        config.protein_masses = [25000.0, 30000.0]
        config.intensity_scalars = [1.0, 0.8]
        config.mass_inhomogeneity = 2.0
        config.common.seed = 42
        config.lc.enabled = True
        config.lc.num_scans = 10 # Reduced from 50 to prevent timeout

        # 1. Get the result from the parallel implementation
        mz_range_parallel, spectra_parallel = execute_simulation_and_write_mzml(
            config=config, final_filepath="", update_queue=None, return_data_only=True
        )

        # 2. Manually run the sequential logic to get the expected result

        # Reset the random seed to ensure the mass distribution is the same
        np.random.seed(config.common.seed)

        common = config.common
        lc = config.lc
        mz_range_seq = np.arange(common.mz_range_start, common.mz_range_end + common.mz_step, common.mz_step)

        all_clean_spectra_seq = []
        for i, protein_mass in enumerate(config.protein_masses):
            if config.mass_inhomogeneity > 0:
                num_samples = 7
                mass_distribution = np.random.normal(loc=protein_mass, scale=config.mass_inhomogeneity, size=num_samples)
                total_spectrum = np.zeros_like(mz_range_seq, dtype=float)
                for sub_mass in mass_distribution:
                    spectrum = generate_protein_spectrum(
                        sub_mass, mz_range_seq, common.mz_step, common.peak_sigma_mz,
                        config.intensity_scalars[i], common.isotopic_enabled, common.resolution
                    )
                    total_spectrum += spectrum
                clean_spectrum = total_spectrum / num_samples
            else:
                clean_spectrum = generate_protein_spectrum(
                    protein_mass, mz_range_seq, common.mz_step, common.peak_sigma_mz,
                    config.intensity_scalars[i], common.isotopic_enabled, common.resolution
                )
            all_clean_spectra_seq.append(clean_spectrum)

        spectra_seq = apply_lc_profile_and_noise(
            mz_range=mz_range_seq, all_clean_spectra=all_clean_spectra_seq, num_scans=lc.num_scans,
            gaussian_std_dev=lc.gaussian_std_dev, lc_tailing_factor=lc.lc_tailing_factor,
            seed=common.seed, noise_option=common.noise_option, pink_noise_enabled=common.pink_noise_enabled,
            apex_scans=None, update_queue=None
        )

        # 3. Compare the results
        np.testing.assert_allclose(mz_range_parallel, mz_range_seq)
        combined_chromatogram_parallel = spectra_parallel[0]
        self.assertEqual(len(combined_chromatogram_parallel), len(spectra_seq))
        for i in range(len(combined_chromatogram_parallel)):
            np.testing.assert_allclose(
                combined_chromatogram_parallel[i],
                spectra_seq[i],
                rtol=1e-7,
                atol=1e-7,
                err_msg=f"Scan {i} does not match between parallel and sequential simulation."
            )


if __name__ == '__main__':
    unittest.main()
