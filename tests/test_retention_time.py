import unittest
import numpy as np
from spec_generator.logic.retention_time import (
    predict_retention_times,
    calculate_apex_scans_from_hydrophobicity,
)


class TestRetentionTimePrediction(unittest.TestCase):
    def test_empty_list(self):
        """Test that an empty list of peptides returns an empty dict."""
        self.assertEqual(predict_retention_times([]), {})

    def test_hydrophobicity_order(self):
        """
        Test that a more hydrophobic peptide gets a later retention time.
        Isoleucine (I) is very hydrophobic (4.5), Arginine (R) is very hydrophilic (-4.5).
        """
        peptides = ["IIII", "RRRR"]
        rts = predict_retention_times(peptides)
        self.assertLess(rts["RRRR"], rts["IIII"])

    def test_scaling_to_run_time(self):
        """Test that the retention times are scaled correctly to the total run time."""
        peptides = ["IIII", "RRRR"]  # Hydrophobic and hydrophilic
        total_run_time = 100.0
        rts = predict_retention_times(peptides, total_run_time_minutes=total_run_time)

        # The most hydrophilic should have RT 0.0
        self.assertAlmostEqual(rts["RRRR"], 0.0)
        # The most hydrophobic should have RT equal to the total run time
        self.assertAlmostEqual(rts["IIII"], total_run_time)

    def test_all_peptides_same_score(self):
        """Test that if all peptides have the same score, they elute in the middle."""
        peptides = ["AAA", "AAA", "AAA"]
        total_run_time = 60.0
        rts = predict_retention_times(peptides, total_run_time_minutes=total_run_time)

        expected_rt = total_run_time / 2.0
        for peptide in peptides:
            self.assertAlmostEqual(rts[peptide], expected_rt)

    def test_single_peptide(self):
        """Test that a single peptide elutes in the middle of the run."""
        peptides = ["PEPTIDE"]
        total_run_time = 60.0
        rts = predict_retention_times(peptides, total_run_time_minutes=total_run_time)
        self.assertAlmostEqual(rts["PEPTIDE"], total_run_time / 2.0)


class TestRetentionTimeScaling(unittest.TestCase):
    def test_linear_scaling(self):
        """Test linear scaling of retention times."""
        scores = [0, 50, 100]
        num_scans = 1000
        scans = calculate_apex_scans_from_hydrophobicity(
            scores, num_scans, retention_time_model="linear"
        )
        # With 10% padding, scans should be at 100, 500, 900
        self.assertEqual(scans[0], 100)
        self.assertEqual(scans[1], 500)
        self.assertEqual(scans[2], 900)

    def test_rpc_scaling(self):
        """Test RPC scaling for a wide range of hydrophobicity scores."""
        scores = [0, 25, 50, 75, 100]  # Light chain to full antibody
        num_scans = 1000
        scans = calculate_apex_scans_from_hydrophobicity(
            scores,
            num_scans,
            retention_time_model="rpc",
            rpc_hydrophobicity_coefficient=0.05,
        )
        # Elution should be non-linear
        self.assertTrue(np.all(np.diff(scans) > 0))  # Should be sorted
        # Check that the spacing increases, indicating exponential behavior
        diffs = np.diff(scans)
        self.assertTrue(diffs[1] > diffs[0])
        self.assertTrue(diffs[2] > diffs[1])
        self.assertTrue(diffs[3] > diffs[2])


if __name__ == "__main__":
    unittest.main()
