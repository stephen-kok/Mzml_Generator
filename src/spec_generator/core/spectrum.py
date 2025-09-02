import math
import multiprocessing
import itertools
from dataclasses import dataclass, field
from typing import List
import numpy as np
import numba
from pyteomics import mass

from .constants import (BASE_INTENSITY_SCALAR, FWHM_TO_SIGMA, MZ_SCALE_FACTOR,
                        PROTON_MASS)
from .isotopes import isotope_calculator
from .peptide_isotopes import peptide_isotope_calculator
from ..logic.fragmentation import generate_fragment_ions, FragmentIon
from .types import FragmentationEvent


@numba.jit(nopython=True, fastmath=True)
def _build_spectrum_from_peaks_numba(
    mz_range: np.ndarray,
    peak_mzs: np.ndarray,
    peak_intensities: np.ndarray,
    peak_sigmas: np.ndarray
) -> np.ndarray:
    """
    Generates a spectrum by summing Gaussian peaks. This function is JIT-compiled
    by Numba for significant performance improvement.
    """
    final_spectrum = np.zeros_like(mz_range, dtype=np.float64)

    # Numba works best with explicit loops.
    for i in range(len(peak_mzs)):
        mz = peak_mzs[i]
        intensity = peak_intensities[i]
        sigma = peak_sigmas[i]

        m_dist = 5 * sigma
        start_idx = np.searchsorted(mz_range, mz - m_dist, side='left')
        end_idx = np.searchsorted(mz_range, mz + m_dist, side='right')

        if start_idx >= end_idx:
            continue

        relevant_mz = mz_range[start_idx:end_idx]
        two_sigma_sq = 2 * sigma**2
        gaussian = intensity * np.exp(-((relevant_mz - mz)**2) / two_sigma_sq)
        final_spectrum[start_idx:end_idx] += gaussian

    return final_spectrum


def generate_protein_spectrum(
    protein_avg_mass: float,
    mz_range: np.ndarray,
    mz_step_float: float,
    peak_sigma_mz_float: float,
    intensity_scalar: float,
    isotopic_enabled: bool,
    resolution: float
) -> np.ndarray:
    """
    Generates a single, clean protein spectrum including isotopic distribution
    and charge state envelope.
    """
    if isotopic_enabled:
        isotopic_distribution, most_abundant_offset = isotope_calculator.get_distribution(protein_avg_mass)
        protein_mono_mass = protein_avg_mass - most_abundant_offset
    else:
        isotopic_distribution, most_abundant_offset = [(0.0, 1.0)], 0.0
        protein_mono_mass = protein_avg_mass

    effective_mass = protein_mono_mass + most_abundant_offset
    min_charge = math.ceil(effective_mass / (mz_range[-1] - PROTON_MASS)) if mz_range[-1] > PROTON_MASS else 1
    max_charge = math.floor(effective_mass / (mz_range[0] - PROTON_MASS)) if mz_range[0] > PROTON_MASS else 150
    min_charge, max_charge = max(1, min_charge), min(150, max_charge)

    if min_charge > max_charge:
        return np.zeros_like(mz_range, dtype=float)

    charge_states = np.arange(min_charge, max_charge + 1)
    num_valid_charge_states = len(charge_states)
    peak_charge_index_relative = num_valid_charge_states // 2
    charge_indices = np.arange(num_valid_charge_states)
    sigma_charge_env = num_valid_charge_states / 4.0
    charge_env_intensities = (
        BASE_INTENSITY_SCALAR *
        np.exp(-((charge_indices - peak_charge_index_relative)**2) / (2 * max(1, sigma_charge_env)**2)) *
        intensity_scalar
    )

    all_peak_mzs, all_peak_intensities, all_peak_sigmas = [], [], []
    isotope_offsets, isotope_rel_intensities = np.array([p[0] for p in isotopic_distribution]), np.array([p[1] for p in isotopic_distribution])

    for i, charge in enumerate(charge_states):
        monoisotopic_mz = (protein_mono_mass + charge * PROTON_MASS) / charge
        base_intensity = charge_env_intensities[i]
        isotope_mzs = monoisotopic_mz + (isotope_offsets / charge)
        visible_mask = (isotope_mzs >= mz_range[0]) & (isotope_mzs <= mz_range[-1])

        if not np.any(visible_mask):
            continue

        visible_mzs = isotope_mzs[visible_mask]
        all_peak_mzs.extend(visible_mzs)
        all_peak_intensities.extend(base_intensity * isotope_rel_intensities[visible_mask])

        sigma_intrinsic = peak_sigma_mz_float * (visible_mzs / MZ_SCALE_FACTOR)
        if isotopic_enabled and resolution > 0:
            sigma_resolution = (visible_mzs / resolution) / FWHM_TO_SIGMA
            total_sigma = np.sqrt(sigma_intrinsic**2 + sigma_resolution**2)
        else:
            total_sigma = sigma_intrinsic
        all_peak_sigmas.extend(total_sigma)

    if not all_peak_mzs:
        return np.zeros_like(mz_range)

    all_peak_mzs = np.array(all_peak_mzs, dtype=np.float64)
    all_peak_intensities = np.array(all_peak_intensities, dtype=np.float64)
    all_peak_sigmas = np.array(all_peak_sigmas, dtype=np.float64)

    return _build_spectrum_from_peaks_numba(
        mz_range, all_peak_mzs, all_peak_intensities, all_peak_sigmas
    )


def generate_peptide_spectrum(
    peptide_sequence: str,
    mz_range: np.ndarray,
    peak_sigma_mz_float: float,
    intensity_scalar: float,
    resolution: float,
    charge: int,
) -> np.ndarray:
    """
    Generates a single, clean spectrum for a peptide at a specific charge state.
    """
    isotopic_distribution = peptide_isotope_calculator.get_distribution(
        peptide_sequence, charge=charge
    )

    if not isotopic_distribution:
        return np.zeros_like(mz_range)

    peak_mzs = np.array([p[0] for p in isotopic_distribution])
    peak_rel_intensities = np.array([p[1] for p in isotopic_distribution])

    visible_mask = (peak_mzs >= mz_range[0]) & (peak_mzs <= mz_range[-1])
    if not np.any(visible_mask):
        return np.zeros_like(mz_range)

    visible_mzs = peak_mzs[visible_mask]
    visible_intensities = intensity_scalar * peak_rel_intensities[visible_mask]

    sigma_intrinsic = peak_sigma_mz_float * (visible_mzs / MZ_SCALE_FACTOR)
    if resolution > 0:
        sigma_resolution = (visible_mzs / resolution) / FWHM_TO_SIGMA
        visible_sigmas = np.sqrt(sigma_intrinsic**2 + sigma_resolution**2)
    else:
        visible_sigmas = sigma_intrinsic

    return _build_spectrum_from_peaks_numba(
        mz_range, visible_mzs, visible_intensities, visible_sigmas
    )


def generate_binding_spectrum(
    protein_avg_mass: float,
    compound_avg_mass: float,
    mz_range: np.ndarray,
    mz_step_float: float,
    peak_sigma_mz_float: float,
    total_binding_percentage: float,
    dar2_percentage_of_bound: float,
    original_intensity_scalar: float,
    isotopic_enabled: bool,
    resolution: float,
) -> np.ndarray:
    """
    Generates a spectrum for a covalent binding scenario, including native,
    DAR-1, and DAR-2 species. The generation of each species is done in parallel.
    """
    native_intensity_scalar = (
        original_intensity_scalar * (100 - total_binding_percentage) / 100.0
    )
    total_bound_intensity = original_intensity_scalar * (
        total_binding_percentage / 100.0
    )

    dar2_intensity_scalar = 0.0
    if total_binding_percentage > 0 and dar2_percentage_of_bound > 0:
        dar2_intensity_scalar = total_bound_intensity * (
            dar2_percentage_of_bound / 100.0
        )

    dar1_intensity_scalar = total_bound_intensity - dar2_intensity_scalar

    tasks = [
        (protein_avg_mass, mz_range, mz_step_float, peak_sigma_mz_float, native_intensity_scalar, isotopic_enabled, resolution),
        (protein_avg_mass + compound_avg_mass, mz_range, mz_step_float, peak_sigma_mz_float, dar1_intensity_scalar, isotopic_enabled, resolution),
        (protein_avg_mass + 2 * compound_avg_mass, mz_range, mz_step_float, peak_sigma_mz_float, dar2_intensity_scalar, isotopic_enabled, resolution),
    ]

    with multiprocessing.Pool(processes=3) as pool:
        results = pool.starmap(generate_protein_spectrum, tasks)

    return np.sum(results, axis=0)


def generate_fragment_spectrum(
    peptide_sequence: str,
    mz_range: np.ndarray,
    peak_sigma_mz_float: float,
    intensity_scalar: float,
    resolution: float,
    ion_types: list[str],
    fragment_charges: list[int],
) -> np.ndarray:
    """
    Generates a tandem mass spectrum for a peptide, including specified
    fragment ions with full isotopic distributions.
    """
    fragment_definitions = generate_fragment_ions(
        sequence=peptide_sequence,
        ion_types=ion_types,
        charges=fragment_charges,
    )

    if not fragment_definitions:
        return np.zeros_like(mz_range)

    all_peak_mzs, all_peak_intensities, all_peak_sigmas = [], [], []

    # Define terminal modifications as pyteomics Composition objects
    ion_comp_mods = {
        'a': mass.Composition({'C': -1, 'O': -1}),
        'b': mass.Composition({}),
        'c': mass.Composition({'N': 1, 'H': 3}),
        'x': mass.Composition({'C': 1, 'O': 1}),
        'y': mass.Composition({'H': 2, 'O': 1}),
        'z': mass.Composition({'O': 1, 'H': -1, 'N': -1}),
    }

    for frag in fragment_definitions:
        try:
            base_comp = mass.Composition(frag.sequence)
            mod_comp = ion_comp_mods.get(frag.ion_type, mass.Composition({}))
            final_comp = base_comp + mod_comp

            if frag.neutral_loss:
                loss_comp = mass.Composition(frag.neutral_loss)
                final_comp -= loss_comp

            dist = list(mass.isotopologues(
                composition=final_comp,
                report_abundance=True,
                overall_threshold=1e-5
            ))

            if not dist:
                continue

            isotope_comps, isotope_abundances = zip(*dist)
            isotope_mzs = [mass.calculate_mass(composition=c, charge=frag.charge) for c in isotope_comps]
            isotope_abundances = np.array(isotope_abundances)
            max_abundance = np.max(isotope_abundances)
            if max_abundance < 1e-9: continue

            relative_intensities = isotope_abundances / max_abundance
            scaled_intensities = relative_intensities * frag.intensity * intensity_scalar

            visible_mask = (np.array(isotope_mzs) >= mz_range[0]) & (np.array(isotope_mzs) <= mz_range[-1])
            if not np.any(visible_mask):
                continue

            visible_mzs = np.array(isotope_mzs)[visible_mask]
            visible_intensities = scaled_intensities[visible_mask]

            all_peak_mzs.extend(visible_mzs)
            all_peak_intensities.extend(visible_intensities)

            sigma_intrinsic = peak_sigma_mz_float * (visible_mzs / MZ_SCALE_FACTOR)
            if resolution > 0:
                sigma_resolution = (visible_mzs / resolution) / FWHM_TO_SIGMA
                total_sigma = np.sqrt(sigma_intrinsic**2 + sigma_resolution**2)
            else:
                total_sigma = sigma_intrinsic
            all_peak_sigmas.extend(total_sigma)

        except Exception:
            continue # Skip fragments that cause errors

    if not all_peak_mzs:
        return np.zeros_like(mz_range)

    return _build_spectrum_from_peaks_numba(
        mz_range,
        np.array(all_peak_mzs),
        np.array(all_peak_intensities),
        np.array(all_peak_sigmas),
    )
