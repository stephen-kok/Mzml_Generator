import unittest
from spec_generator.logic.retention_time import predict_retention_times

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
        peptides = ["IIII", "RRRR"] # Hydrophobic and hydrophilic
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

if __name__ == '__main__':
    unittest.main()
