import pytest
import numpy as np
from spec_generator.logic.deconvolution import run_unidec
from spec_generator.core.spectrum import generate_protein_spectrum

@pytest.fixture
def single_protein_spectrum():
    """
    Generates a test spectrum for a single protein.
    """
    protein_mass = 25000.0
    mz_range_tuple = (1000, 2500)
    resolution = 10000
    mz_grid_size = 20000

    mz_axis = np.linspace(mz_range_tuple[0], mz_range_tuple[1], mz_grid_size)
    mz_step = mz_axis[1] - mz_axis[0]

    # These parameters are based on the core.spectrum function signature
    intensity_array = generate_protein_spectrum(
        protein_avg_mass=protein_mass,
        mz_range=mz_axis,
        mz_step_float=mz_step,
        peak_sigma_mz_float=0.02, # A reasonable value for this resolution
        intensity_scalar=1e5,
        isotopic_enabled=False, # Generate spectrum based on avg mass directly
        resolution=resolution
    )

    # Create the (N, 2) raw spectrum array, filtering out zero intensities
    non_zero_indices = intensity_array > 1e-6 # Use a small threshold
    raw_spectrum = np.vstack((mz_axis[non_zero_indices], intensity_array[non_zero_indices])).T

    return raw_spectrum, protein_mass

def test_run_unidec_single_protein(single_protein_spectrum):
    """
    Tests the deconvolution on a simple, known spectrum of a single protein.
    """
    raw_spectrum, protein_mass = single_protein_spectrum

    config = {
        "mz_range": (1000, 2500),
        "charge_range": (10, 30),
        "mass_range": (24000, 26000),
        "num_iterations": 20,
        "peak_sigma_mz": 0.02,
        "mass_grid_size": 20000
    }

    peaks, score = run_unidec(raw_spectrum, config)

    # Assertions
    assert isinstance(peaks, list)
    assert isinstance(score, float)
    assert score > 80.0, f"UniScore ({score}) is lower than expected."

    assert len(peaks) > 0, "No peaks were detected."

    # Find the most intense detected peak
    detected_mass = sorted(peaks, key=lambda x: x[1], reverse=True)[0][0]

    mass_tolerance = 2.0 # in Daltons
    assert abs(detected_mass - protein_mass) < mass_tolerance, \
        f"Detected mass {detected_mass:.2f} is not within {mass_tolerance} Da of expected mass {protein_mass:.2f}."

def test_run_unidec_missing_config():
    """
    Tests that run_unidec raises a ValueError if essential config keys are missing.
    """
    with pytest.raises(ValueError, match="mz_range, charge_range, and mass_range must be provided."):
        run_unidec(np.array([[1000, 1]]), {})

def test_run_unidec_empty_spectrum():
    """
    Tests that run_unidec handles an empty spectrum gracefully.
    """
    empty_spectrum = np.empty((0, 2))

    config = {
        "mz_range": (1000, 2500),
        "charge_range": (10, 30),
        "mass_range": (24000, 26000),
    }

    peaks, score = run_unidec(empty_spectrum, config)

    assert isinstance(peaks, list)
    assert isinstance(score, float)
    assert len(peaks) == 0
    # Score should be low or zero for no data
    assert score < 1.0


@pytest.mark.parametrize("protein_mass, mz_range_tuple, charge_range_tuple, mass_range_tuple", [
    (75000.0, (1500, 3500), (20, 55), (74000, 76000)),
    (150000.0, (2000, 5000), (30, 80), (148000, 152000)),
])
def test_run_unidec_high_mass_protein(protein_mass, mz_range_tuple, charge_range_tuple, mass_range_tuple):
    """
    Tests the deconvolution for single high-mass proteins.
    """
    resolution = 5000 # Lower resolution is more typical for very high mass
    mz_grid_size = 30000
    mz_axis = np.linspace(mz_range_tuple[0], mz_range_tuple[1], mz_grid_size)
    mz_step = mz_axis[1] - mz_axis[0]

    intensity_array = generate_protein_spectrum(
        protein_avg_mass=protein_mass,
        mz_range=mz_axis,
        mz_step_float=mz_step,
        peak_sigma_mz_float=0.05,
        intensity_scalar=1e5,
        isotopic_enabled=False,
        resolution=resolution
    )
    non_zero_indices = intensity_array > 1e-6
    raw_spectrum = np.vstack((mz_axis[non_zero_indices], intensity_array[non_zero_indices])).T

    config = {
        "mz_range": mz_range_tuple,
        "charge_range": charge_range_tuple,
        "mass_range": mass_range_tuple,
        "num_iterations": 30,
        "peak_sigma_mz": 0.05,
        "mass_grid_size": 30000
    }

    peaks, score = run_unidec(raw_spectrum, config)

    assert score > 70.0, f"UniScore ({score}) is lower than expected for mass {protein_mass}."
    assert len(peaks) > 0, f"No peaks were detected for mass {protein_mass}."

    detected_mass = sorted(peaks, key=lambda x: x[1], reverse=True)[0][0]

    # Looser tolerance for higher mass
    mass_tolerance = protein_mass * 0.0001 # 0.01% tolerance
    assert abs(detected_mass - protein_mass) < mass_tolerance, \
        f"Detected mass {detected_mass:.2f} is not within {mass_tolerance:.2f} Da of expected mass {protein_mass:.2f}."
