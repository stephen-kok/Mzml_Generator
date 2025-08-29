import unittest
from spec_generator.logic.peptide import digest_sequence

class TestPeptideDigestion(unittest.TestCase):

    def test_simple_digest_no_missed_cleavages(self):
        """Test a basic tryptic digest with no missed cleavages."""
        sequence = "AGRSEPTIDEKSEQUENCE"
        expected = ["AGR", "SEPTIDEK", "SEQUENCE"]
        result = digest_sequence(sequence, missed_cleavages=0)
        self.assertEqual(sorted(result), sorted(expected))

    def test_digest_with_one_missed_cleavage(self):
        """Test a tryptic digest allowing for one missed cleavage."""
        sequence = "AGRSEPTIDEKSEQUENCE"
        expected = [
            "AGR",
            "SEPTIDEK",
            "SEQUENCE",
            "AGRSEPTIDEK",
            "SEPTIDEKSEQUENCE"
        ]
        result = digest_sequence(sequence, missed_cleavages=1)
        self.assertEqual(sorted(result), sorted(expected))

    def test_digest_with_two_missed_cleavages(self):
        """Test a tryptic digest allowing for two missed cleavages."""
        sequence = "AGRSEPTIDEKSEQUENCE"
        expected = [
            "AGR",
            "SEPTIDEK",
            "SEQUENCE",
            "AGRSEPTIDEK",
            "SEPTIDEKSEQUENCE",
            "AGRSEPTIDEKSEQUENCE"
        ]
        result = digest_sequence(sequence, missed_cleavages=2)
        self.assertEqual(sorted(result), sorted(expected))

    def test_no_cleavage_sites(self):
        """Test a sequence that contains no tryptic cleavage sites."""
        sequence = "AGFDE"
        expected = ["AGFDE"]
        result = digest_sequence(sequence)
        self.assertEqual(sorted(result), sorted(expected))

    def test_cleavage_inhibited_by_proline(self):
        """Test that cleavage after K or R is inhibited if followed by P."""
        sequence = "TESTKPTEST" # K is followed by P, so no cleavage
        expected = ["TESTKPTEST"]
        result = digest_sequence(sequence)
        self.assertEqual(sorted(result), sorted(expected))

    def test_empty_sequence(self):
        """Test that an empty input sequence results in an empty list."""
        sequence = ""
        expected = []
        result = digest_sequence(sequence)
        self.assertEqual(result, expected)

    def test_sequence_ending_with_cleavage_site(self):
        """Test digestion of a sequence ending with K or R."""
        sequence = "PEPTIDEK"
        expected = ["PEPTIDEK"]
        result = digest_sequence(sequence)
        self.assertEqual(sorted(result), sorted(expected))

if __name__ == '__main__':
    unittest.main()
