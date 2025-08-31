import unittest
import unittest.mock
import numpy as np

from spec_generator.core.lc import _get_lc_peak_shape, apply_lc_profile_and_noise


class TestLCPeakShape(unittest.TestCase):

    def test_gaussian_shape_is_symmetric(self):
        """Test that a pure Gaussian peak is generated when tau is zero."""
        num_scans = 21
        apex_scan_index = (num_scans - 1) / 2.0
        std_dev = 3.0
        tau = 0.0

        shape = _get_lc_peak_shape(num_scans, apex_scan_index, std_dev, tau)

        # For a symmetric peak, the max should be exactly at the center
        self.assertEqual(np.argmax(shape), int(apex_scan_index))

        # It should be symmetric around the center
        self.assertAlmostEqual(shape[int(apex_scan_index) - 1], shape[int(apex_scan_index) + 1], places=5)
        self.assertAlmostEqual(shape[0], shape[-1], places=5)

    def test_emg_shape_is_asymmetric(self):
        """Test that an EMG peak is generated when tau is positive and is asymmetric."""
        num_scans = 51
        apex_scan_index = (num_scans - 1) / 2.0
        std_dev = 5.0
        tau = 2.0

        shape = _get_lc_peak_shape(num_scans, apex_scan_index, std_dev, tau)

        # For a tailing peak, the max should occur *after* the theoretical Gaussian center
        self.assertGreater(np.argmax(shape), apex_scan_index)

        # It should be asymmetric
        self.assertNotAlmostEqual(shape[np.argmax(shape) - 1], shape[np.argmax(shape) + 1])

    def test_normalization(self):
        """Test that the peak is always normalized to a maximum of 1.0."""
        shape1 = _get_lc_peak_shape(21, 10.0, 3.0, 0.0)
        self.assertAlmostEqual(np.max(shape1), 1.0, places=5)

        shape2 = _get_lc_peak_shape(51, 25.0, 5.0, 2.0)
        self.assertAlmostEqual(np.max(shape2), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()


class TestLCProfileApplication(unittest.TestCase):

    def setUp(self):
        self.mz_range = np.arange(1000, 2001, 1)
        self.num_scans = 100
        # Create two simple, clean spectra with one peak each
        self.spec1 = np.zeros_like(self.mz_range)
        self.spec1[200] = 1000  # Peak at 1200 m/z
        self.spec2 = np.zeros_like(self.mz_range)
        self.spec2[800] = 1000  # Peak at 1800 m/z

    @unittest.mock.patch('spec_generator.core.lc.default_rng')
    def test_retention_time_scaling(self, mock_rng):
        """
        Test that species with different apex_scans elute at different times.
        """
        # Configure the mock RNG to return a generator that produces 1.0 for the variation
        mock_generator = unittest.mock.Mock()
        mock_generator.normal.return_value = 1.0
        mock_rng.return_value = mock_generator

        all_clean_spectra = [self.spec1, self.spec2]
        apex_scans = [20, 80]  # Elute spec1 early, spec2 late

        chromatogram = apply_lc_profile_and_noise(
            mz_range=self.mz_range,
            all_clean_spectra=all_clean_spectra,
            num_scans=self.num_scans,
            gaussian_std_dev=5.0,
            lc_tailing_factor=0.0,  # Use Gaussian for predictable peak
            seed=0,
            noise_option="No Noise",
            pink_noise_enabled=False,
            apex_scans=apex_scans
        )

        # Find where the m/z=1200 signal maximizes
        intensity_at_1200 = [scan[200] for scan in chromatogram]
        # With a gaussian (no tailing), the max should be exactly on the apex
        self.assertEqual(np.argmax(intensity_at_1200), apex_scans[0])

        # Find where the m/z=1800 signal maximizes
        intensity_at_1800 = [scan[800] for scan in chromatogram]
        self.assertEqual(np.argmax(intensity_at_1800), apex_scans[1])

        # The mock RNG is called for retention time variation for each scan for each species.
        mock_rng.assert_called_once()
        expected_calls = self.num_scans * len(all_clean_spectra)
        self.assertEqual(mock_generator.normal.call_count, expected_calls)
