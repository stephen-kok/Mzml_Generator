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

    def test_binding_peak_is_correct(self):
        """
        Test that the simulation generates a peak at the correct m/z for a
        fully bound species.
        """
        protein_mass = 25000.0
        compound_mass = 500.0
        config = copy.deepcopy(self.default_config)
        config.protein_avg_mass = protein_mass
        # Force 100% DAR1 binding
        config.prob_binding = 1.0
        config.prob_dar2 = 0.0
        config.total_binding_range = (100.0, 100.0)

        mz_range, spectra = execute_binding_simulation(
            config=config, compound_mass=compound_mass,
            total_binding_percentage=100.0, dar2_percentage_of_bound=0.0,
            filepath="", return_data_only=True
        )
        spectrum = spectra[0] # The simulation returns a list of spectra directly

        PROTON_MASS = 1.007276
        bound_mass = protein_mass + compound_mass

        # Check for a few expected charge states of the bound species
        for z in [15, 16, 17]:
            expected_mz = (bound_mass + z * PROTON_MASS) / z
            idx = np.abs(mz_range - expected_mz).argmin()
            self.assertGreater(
                spectrum[idx], 0,
                msg=f"Expected a peak for bound species at z={z} (m/z ~{expected_mz:.2f}), but found none."
            )


    def test_parallel_binding_matches_sequential(self):
        """
        Verify that the parallel implementation of generate_binding_spectrum
        produces the exact same output as the original sequential version.
        """
        import numpy as np
        from spec_generator.core.spectrum import generate_protein_spectrum, generate_binding_spectrum

        mz_range = np.arange(400.0, 2500.0, 0.02)

        params = {
            "protein_avg_mass": 25000.0,
            "compound_avg_mass": 500.0,
            "mz_range": mz_range,
            "mz_step_float": 0.02,
            "peak_sigma_mz_float": 0.01,
            "total_binding_percentage": 50.0,
            "dar2_percentage_of_bound": 20.0,
            "original_intensity_scalar": 1.0,
            "isotopic_enabled": True,
            "resolution": 120000
        }

        # 1. Get the result from the parallel implementation
        parallel_result = generate_binding_spectrum(**params)

        # 2. Manually run the sequential logic to get the expected result
        native_intensity_scalar = params["original_intensity_scalar"] * (100 - params["total_binding_percentage"]) / 100.0
        total_bound_intensity = params["original_intensity_scalar"] * (params["total_binding_percentage"] / 100.0)
        dar2_intensity_scalar = 0.0
        if params["total_binding_percentage"] > 0 and params["dar2_percentage_of_bound"] > 0:
            dar2_intensity_scalar = total_bound_intensity * (params["dar2_percentage_of_bound"] / 100.0)
        dar1_intensity_scalar = total_bound_intensity - dar2_intensity_scalar

        # In the new implementation, generate_protein_spectrum is called in parallel.
        # To test against the old implementation, we call it sequentially here.
        native_spectrum = generate_protein_spectrum(
            protein_avg_mass=params["protein_avg_mass"],
            mz_range=params["mz_range"],
            mz_step_float=params["mz_step_float"],
            peak_sigma_mz_float=params["peak_sigma_mz_float"],
            intensity_scalar=native_intensity_scalar,
            isotopic_enabled=params["isotopic_enabled"],
            resolution=params["resolution"]
        )
        dar1_spectrum = generate_protein_spectrum(
            protein_avg_mass=params["protein_avg_mass"] + params["compound_avg_mass"],
            mz_range=params["mz_range"],
            mz_step_float=params["mz_step_float"],
            peak_sigma_mz_float=params["peak_sigma_mz_float"],
            intensity_scalar=dar1_intensity_scalar,
            isotopic_enabled=params["isotopic_enabled"],
            resolution=params["resolution"]
        )
        dar2_spectrum = generate_protein_spectrum(
            protein_avg_mass=params["protein_avg_mass"] + 2 * params["compound_avg_mass"],
            mz_range=params["mz_range"],
            mz_step_float=params["mz_step_float"],
            peak_sigma_mz_float=params["peak_sigma_mz_float"],
            intensity_scalar=dar2_intensity_scalar,
            isotopic_enabled=params["isotopic_enabled"],
            resolution=params["resolution"]
        )
        sequential_result = native_spectrum + dar1_spectrum + dar2_spectrum

        # 3. Compare the results
        np.testing.assert_allclose(
            parallel_result,
            sequential_result,
            rtol=1e-7,
            atol=1e-7,
            err_msg="Parallel binding spectrum generation does not match sequential execution."
        )


if __name__ == '__main__':
    unittest.main()
