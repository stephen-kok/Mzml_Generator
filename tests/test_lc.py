import unittest
import numpy as np

# The function to test is private, so we need to import it with a leading underscore
from spec_generator.core.lc import _get_lc_peak_shape

class TestLCPeakShape(unittest.TestCase):

    def test_gaussian_shape_is_symmetric(self):
        """Test that a pure Gaussian peak is generated when tau is zero."""
        num_scans = 21
        std_dev = 3.0
        tau = 0.0

        shape = _get_lc_peak_shape(num_scans, std_dev, tau)

        # For a symmetric peak, the max should be exactly at the center
        center_index = (num_scans - 1) // 2
        self.assertEqual(np.argmax(shape), center_index)

        # It should be symmetric around the center
        self.assertAlmostEqual(shape[center_index - 1], shape[center_index + 1], places=5)
        self.assertAlmostEqual(shape[0], shape[-1], places=5)

    def test_emg_shape_is_asymmetric(self):
        """Test that an EMG peak is generated when tau is positive and is asymmetric."""
        num_scans = 51
        std_dev = 5.0
        tau = 2.0

        shape = _get_lc_peak_shape(num_scans, std_dev, tau)

        center_index = (num_scans - 1) // 2

        # For a tailing peak, the max should occur *after* the theoretical Gaussian center
        self.assertGreater(np.argmax(shape), center_index)

        # It should be asymmetric
        self.assertNotAlmostEqual(shape[np.argmax(shape) - 1], shape[np.argmax(shape) + 1])

    def test_normalization(self):
        """Test that the peak is always normalized to a maximum of 1.0."""
        shape1 = _get_lc_peak_shape(21, 3.0, 0.0)
        self.assertAlmostEqual(np.max(shape1), 1.0, places=5)

        shape2 = _get_lc_peak_shape(51, 5.0, 2.0)
        self.assertAlmostEqual(np.max(shape2), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
