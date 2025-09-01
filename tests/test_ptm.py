import pytest
from unittest.mock import patch
from spec_generator.logic.ptm import calculate_ptm_mass_shift, Ptm

# --- Test Constants ---
OXIDATION = Ptm(name="Oxidation", mass_shift=15.9949, residue="M", probability=0.5)
METHYLATION = Ptm(name="Methylation", mass_shift=14.0157, residue="K", probability=0.3)
DEAMIDATION = Ptm(name="Deamidation", mass_shift=0.9840, residue="N", probability=0.8)

# --- Test Cases ---

def test_no_ptms_configured():
    """
    Tests that the mass shift is zero when no PTMs are provided.
    """
    assert calculate_ptm_mass_shift("PEPTIDE", []) == 0.0

def test_no_target_residues_in_sequence():
    """
    Tests that the mass shift is zero when the sequence contains no target residues.
    """
    assert calculate_ptm_mass_shift("PEPTIDE", [OXIDATION, METHYLATION]) == 0.0

def test_guaranteed_ptm_application():
    """
    Tests that the mass shift is correctly calculated when a PTM has a probability of 1.0.
    """
    guaranteed_oxidation = Ptm(name="Oxidation", mass_shift=15.9949, residue="M", probability=1.0)
    sequence = "MPEPTIDEM"  # Contains two methionine residues
    expected_shift = 2 * guaranteed_oxidation.mass_shift
    assert calculate_ptm_mass_shift(sequence, [guaranteed_oxidation]) == pytest.approx(expected_shift)

def test_impossible_ptm_application():
    """
    Tests that the mass shift is zero when a PTM has a probability of 0.0.
    """
    impossible_oxidation = Ptm(name="Oxidation", mass_shift=15.9949, residue="M", probability=0.0)
    sequence = "MPEPTIDEM"
    assert calculate_ptm_mass_shift(sequence, [impossible_oxidation]) == 0.0

@patch('spec_generator.logic.ptm.random.random')
def test_stochastic_ptm_is_applied(mock_random):
    """
    Tests that a PTM is applied when random.random() returns a value below the probability.
    """
    mock_random.return_value = 0.4  # 0.4 is less than OXIDATION's probability of 0.5
    sequence = "M"
    assert calculate_ptm_mass_shift(sequence, [OXIDATION]) == pytest.approx(OXIDATION.mass_shift)

@patch('spec_generator.logic.ptm.random.random')
def test_stochastic_ptm_is_not_applied(mock_random):
    """
    Tests that a PTM is not applied when random.random() returns a value above the probability.
    """
    mock_random.return_value = 0.6  # 0.6 is greater than OXIDATION's probability of 0.5
    sequence = "M"
    assert calculate_ptm_mass_shift(sequence, [OXIDATION]) == 0.0

@patch('spec_generator.logic.ptm.random.random')
def test_multiple_sites_and_ptms_stochastic(mock_random):
    """
    Tests a complex scenario with multiple PTMs and multiple sites, mocking the
    random number generator to control the outcome.
    """
    # Sequence has two 'M's, one 'K', and one 'N'
    sequence = "MPEPTIDEMNK"

    # The function iterates through the sequence: M, M, N, K
    # We expect random.random() to be called for each potential site.
    # 1. First M:  random=0.2 -> 0.2 < 0.5 (Oxidation prob) -> APPLY
    # 2. Second M: random=0.7 -> 0.7 > 0.5 (Oxidation prob) -> SKIP
    # 3. N:        random=0.5 -> 0.5 < 0.8 (Deamidation prob) -> APPLY
    # 4. K:        random=0.4 -> 0.4 > 0.3 (Methylation prob) -> SKIP
    mock_random.side_effect = [0.2, 0.7, 0.5, 0.4]

    ptm_configs = [OXIDATION, METHYLATION, DEAMIDATION]

    expected_shift = OXIDATION.mass_shift + DEAMIDATION.mass_shift
    actual_shift = calculate_ptm_mass_shift(sequence, ptm_configs)

    assert actual_shift == pytest.approx(expected_shift)
    assert mock_random.call_count == 4

def test_multiple_ptms_on_same_residue_type():
    """
    Tests that multiple different PTMs can target the same amino acid type.
    """
    # Both PTMs target Lysine (K)
    ptm1 = Ptm(name="Mod1", mass_shift=10.0, residue="K", probability=1.0)
    ptm2 = Ptm(name="Mod2", mass_shift=5.0, residue="K", probability=1.0)

    sequence = "K"
    expected_shift = ptm1.mass_shift + ptm2.mass_shift
    assert calculate_ptm_mass_shift(sequence, [ptm1, ptm2]) == pytest.approx(expected_shift)
