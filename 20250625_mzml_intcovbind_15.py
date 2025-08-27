# --- START OF FILE 20250625_mzml_intcovbind_draft.py ---

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk, filedialog
from ttkbootstrap import Style
from ttkbootstrap.constants import *
import numpy as np
import os
import random
import math
import base64
import hashlib
import zlib
import xml.etree.ElementTree as ET
from numpy.random import SeedSequence, default_rng
import time
from functools import partial
import re
import threading
import queue
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import multiprocessing
import copy
from scipy.stats import poisson
from numba import jit # <-- NEW IMPORT FOR PERFORMANCE

# --- Constants and Noise Presets ---
BASE_INTENSITY_SCALAR = 2000.0
MZ_SCALE_FACTOR = 1000.0
PROTON_MASS = 1.007276
NEUTRON_MASS_APPROX = 1.00866491588
FWHM_TO_SIGMA = 2 * math.sqrt(2 * math.log(2))

noise_presets = {
    "Default Noise": {
        "base_noise_level": 0.05, "decay_constant": 200.0, "white_noise_level": 0.002,
        "shot_noise_factor": 0.01, "perlin_scale": 50.0, "perlin_octaves": 4,
        "baseline_wobble_amplitude": 0.1, "baseline_wobble_scale": 100.0,
        "white_noise_decay_constant": 200.0, "baseline_wobble_decay_constant": 300.0,
    },
    "Noisy": {
        "base_noise_level": 5.0, "decay_constant": 250.0, "white_noise_level": 0.05,
        "shot_noise_factor": 0.08, "perlin_scale": 50.0, "perlin_octaves": 4,
        "baseline_wobble_amplitude": 3.0, "baseline_wobble_scale": 500.0,
        "white_noise_decay_constant": 600.0, "baseline_wobble_decay_constant": 1500.0,
    },
}

# --- Numba JIT-Compiled Helper for Noise ---
@jit(nopython=True)
def _generate_perlin_like_noise_numba(n_points, mz_values, scale, octaves):
    """
    A Numba-JIT compiled function to generate Perlin-like noise extremely fast.
    This function contains the performance-critical loops.
    """
    total_noise = np.zeros(n_points, dtype=np.float64)
    # Numba requires the random seed to be handled inside, so we pre-seed it
    # We can't use the default_rng object directly inside a nopython function.
    
    for i in range(octaves):
        amplitude = 1 / (2**i)
        frequency_scale = scale / (2**i)
        
        num_control_points = int(n_points / frequency_scale)
        if num_control_points < 2:
            num_control_points = 2
        
        # Create control points for this octave
        control_points_x = np.linspace(mz_values[0], mz_values[-1], num_control_points)
        control_points_y = np.random.uniform(-1.0, 1.0, num_control_points)
        
        # Linearly interpolate to generate the noise octave
        total_noise += np.interp(mz_values, control_points_x, control_points_y) * amplitude
        
    return total_noise

# --- Top-Level Worker Functions for Multiprocessing ---
def run_simulation_task(args):
    """A top-level function that runs in a separate process for the Spectrum Generator."""
    try:
        (mass, scalar, common_params, file_seed) = args
        _, most_abundant_offset = isotope_calculator.get_distribution(mass)
        effective_protein_mass = mass + most_abundant_offset
        placeholders = {
            "date": datetime.now().strftime('%Y-%m-%d'), "time": datetime.now().strftime('%H%M%S'), 
            "protein_mass": int(round(effective_protein_mass)), "scalar": scalar, "scans": common_params['num_scans'], 
            "noise": common_params['noise_option'].replace(" ", ""), "seed": file_seed, "num_proteins": 1
        }
        filename = format_filename(common_params['filename_template'], placeholders)
        filepath = os.path.join(common_params['output_directory'], filename)
        success = execute_simulation_and_write_mzml(
            None, str(mass), common_params['mz_step'], common_params['peak_sigma_mz'],
            common_params['mz_range_start'], common_params['mz_range_end'], [scalar],
            common_params['noise_option'], file_seed, common_params['lc_simulation_enabled'],
            common_params['num_scans'], common_params['scan_interval'], 
            common_params['gaussian_std_dev'], filepath, common_params['isotopic_enabled'],
            common_params['resolution']
        )
        if success:
            return (True, f"--- Successfully generated file for Protein ~{int(round(effective_protein_mass))} Da ---\n")
        else:
            return (False, f"--- FAILED to generate file for Protein {int(mass)} Da ---\n")
    except Exception as e:
        return (False, f"--- Critical error for Protein {int(args[0])} Da: {e} ---\n")

def run_binding_task(args):
    """A top-level function that runs in a separate process for the Covalent Binding tab."""
    try:
        (compound_name, compound_mass, protein_avg_mass, common_params, file_seed) = args
        _, most_abundant_offset = isotope_calculator.get_distribution(protein_avg_mass)
        effective_protein_mass = protein_avg_mass + most_abundant_offset
        rng_scenario, rng_intensity = default_rng(file_seed), default_rng(file_seed + 1)
        total_binding, dar2_of_bound, desc = 0.0, 0.0, "No Binding"
        if rng_scenario.random() < common_params['prob_binding']:
            total_binding = rng_intensity.uniform(*common_params['total_binding_range'])
            desc = f"Binding ({total_binding:.2f}%)"
            if rng_scenario.random() < common_params['prob_dar2']:
                dar2_of_bound = rng_intensity.uniform(*common_params['dar2_range'])
                desc += f", DAR-2 ({dar2_of_bound:.2f}% of bound)"
        message = f"--- Processing: {compound_name} | Scenario: {desc} ---\n"
        mz_range = np.arange(float(common_params['mz_range_start']), float(common_params['mz_range_end']) + float(common_params['mz_step']), float(common_params['mz_step']))
        clean_spec = generate_binding_spectrum(protein_avg_mass, compound_mass, mz_range, float(common_params['mz_step']), float(common_params['peak_sigma_mz']), total_binding, dar2_of_bound, BASE_INTENSITY_SCALAR, common_params['isotopic_enabled'], common_params['resolution'])
        scans_to_gen = common_params['num_scans']
        final_spectra = generate_gaussian_scaled_spectra(mz_range, [clean_spec], scans_to_gen, common_params['gaussian_std_dev'], None, file_seed, common_params['noise_option'])
        mzml_content = create_mzml_content_et(mz_range, final_spectra, common_params['scan_interval'], None)
        if not mzml_content: return (False, message + "  ERROR: mzML creation failed.\n")
        placeholders = {"date": datetime.now().strftime('%Y-%m-%d'), "time": datetime.now().strftime('%H%M%S'), "compound_name": compound_name, "protein_mass": int(round(effective_protein_mass)), "scans": scans_to_gen, "noise": common_params['noise_option'].replace(" ", ""), "seed": file_seed}
        filename = format_filename(common_params['filename_template'], placeholders); filepath = os.path.join(common_params['output_directory'], filename)
        unique_filepath = create_unique_filename(filepath); os.makedirs(os.path.dirname(unique_filepath), exist_ok=True)
        with open(unique_filepath, "wb") as f: f.write(mzml_content)
        message += f"  SUCCESS: Wrote to {os.path.basename(unique_filepath)}\n"
        return (True, message)
    except Exception as e:
        return (False, f"--- Critical error for compound {args[0]}: {e} ---\n")

# --- Tooltip Helper Class ---
class Tooltip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tooltip = widget, text, None
        self.widget.bind("<Enter>", self.show_tooltip); self.widget.bind("<Leave>", self.hide_tooltip)
    def show_tooltip(self, event=None):
        if self.tooltip or not self.text: return
        x, y, _, _ = self.widget.bbox("insert"); x += self.widget.winfo_rootx() + 25; y += self.widget.winfo_rooty() + 25
        self.tooltip = tk.Toplevel(self.widget); self.tooltip.wm_overrideredirect(True); self.tooltip.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(self.tooltip, text=self.text, background="#FFFFE0", foreground="black", relief="solid", borderwidth=1, padding=5, justify=tk.LEFT)
        label.pack()
    def hide_tooltip(self, event=None):
        if self.tooltip: self.tooltip.destroy(); self.tooltip = None

# --- Scientific Model Helpers ---
class IsotopeCalculator:
    AVERAGINE_MASS_PER_NEUTRON_PROB = 1110.0
    def get_distribution(self, mass: float, num_peaks: int = 40):
        if mass <= 0: return [(0.0, 1.0)], 0.0
        lambda_val = mass / self.AVERAGINE_MASS_PER_NEUTRON_PROB
        k = np.arange(num_peaks); probabilities = poisson.pmf(k, lambda_val)
        significant_indices = np.where(probabilities > 1e-5)[0]
        last_significant_index = significant_indices[-1] if len(significant_indices) > 0 else int(lambda_val) + 5
        num_peaks_to_calc = min(num_peaks, last_significant_index + 3)
        k = np.arange(num_peaks_to_calc); probabilities = poisson.pmf(k, lambda_val)
        mass_offsets = k * NEUTRON_MASS_APPROX
        max_prob = np.max(probabilities)
        if max_prob < 1e-12: return [(0.0, 1.0)], 0.0
        normalized_probs = probabilities / max_prob
        most_abundant_offset = mass_offsets[np.argmax(probabilities)]
        distribution = list(zip(mass_offsets, normalized_probs))
        return distribution, most_abundant_offset
isotope_calculator = IsotopeCalculator()

# --- Core Spectrum Generation Functions (Final Logic) ---
def generate_protein_spectrum(
    protein_avg_mass: float, mz_range: np.ndarray, mz_step_float: float,
    peak_sigma_mz_float: float, intensity_scalar: float,
    isotopic_enabled: bool, resolution: float
):
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
    if min_charge > max_charge: return np.zeros_like(mz_range, dtype=float)
    charge_states = np.arange(min_charge, max_charge + 1)
    num_valid_charge_states = len(charge_states); peak_charge_index_relative = num_valid_charge_states // 2
    charge_indices = np.arange(num_valid_charge_states); sigma_charge_env = num_valid_charge_states / 4.0
    charge_env_intensities = BASE_INTENSITY_SCALAR * np.exp(-((charge_indices - peak_charge_index_relative) ** 2) / (2 * max(1, sigma_charge_env) ** 2)) * intensity_scalar
    all_peak_mzs, all_peak_intensities, all_peak_sigmas = [], [], []
    isotope_offsets, isotope_rel_intensities = np.array([p[0] for p in isotopic_distribution]), np.array([p[1] for p in isotopic_distribution])
    for i, charge in enumerate(charge_states):
        monoisotopic_mz = (protein_mono_mass + charge * PROTON_MASS) / charge; base_intensity = charge_env_intensities[i]
        isotope_mzs = monoisotopic_mz + (isotope_offsets / charge)
        visible_mask = (isotope_mzs >= mz_range[0]) & (isotope_mzs <= mz_range[-1])
        if not np.any(visible_mask): continue
        visible_mzs = isotope_mzs[visible_mask]
        all_peak_mzs.extend(visible_mzs); all_peak_intensities.extend(base_intensity * isotope_rel_intensities[visible_mask])
        sigma_intrinsic = peak_sigma_mz_float * (visible_mzs / MZ_SCALE_FACTOR)
        total_sigma = np.sqrt(sigma_intrinsic**2 + ((visible_mzs / resolution) / FWHM_TO_SIGMA)**2) if isotopic_enabled and resolution > 0 else sigma_intrinsic
        all_peak_sigmas.extend(total_sigma)
    if not all_peak_mzs: return np.zeros_like(mz_range)
    all_peak_mzs, all_peak_intensities, all_peak_sigmas = np.array(all_peak_mzs), np.array(all_peak_intensities), np.array(all_peak_sigmas)
    final_spectrum = np.zeros_like(mz_range, dtype=float); chunk_size = 100
    for i in range(0, len(all_peak_mzs), chunk_size):
        chunk_mzs, chunk_intensities, chunk_sigmas = all_peak_mzs[i:i+chunk_size], all_peak_intensities[i:i+chunk_size], all_peak_sigmas[i:i+chunk_size]
        mz_grid = mz_range[:, np.newaxis]; two_sigma_sq = 2 * chunk_sigmas**2
        gaussians = chunk_intensities * np.exp(-((mz_grid - chunk_mzs)**2) / two_sigma_sq)
        final_spectrum += np.sum(gaussians, axis=1)
    return final_spectrum

def add_noise(
    mz_values: np.ndarray, intensities: np.ndarray, base_noise_level: float, min_noise_level: float,
    max_intensity: float, decay_constant: float, white_noise_level: float, shot_noise_factor: float,
    perlin_scale: float, perlin_octaves: int, baseline_wobble_amplitude: float, baseline_wobble_scale: float,
    white_noise_decay_constant: float, baseline_wobble_decay_constant: float, seed: int
) -> np.ndarray:
    rng = default_rng(seed)
    # Seed numpy's legacy random generator for Numba compatibility
    np.random.seed(seed)
    
    # --- Generate noise using the super-fast Numba-compiled function ---
    total_perlin_chem = _generate_perlin_like_noise_numba(len(mz_values), mz_values, perlin_scale, perlin_octaves)
    # Re-seed for the second call to get different noise
    np.random.seed(seed + 1) 
    perlin_wobble = _generate_perlin_like_noise_numba(len(mz_values), mz_values, baseline_wobble_scale, 1)

    # --- Combine all noise sources using vectorized numpy operations ---
    chemical_noise_level = np.maximum(min_noise_level, base_noise_level * np.exp(-mz_values / decay_constant) * (1 + 0.5 * total_perlin_chem))
    white_noise = white_noise_level * max_intensity
    shot_noise = shot_noise_factor * np.sqrt(np.maximum(0, intensities))
    total_noise_stddev = np.sqrt(chemical_noise_level**2 + white_noise**2 + shot_noise**2)
    random_noise = rng.normal(0, total_noise_stddev, size=intensities.shape)
    baseline_wobble = baseline_wobble_amplitude * perlin_wobble

    return intensities + random_noise + baseline_wobble

def generate_scaled_spectra(
    protein_masses_list: list[float], mz_range: np.ndarray, mz_step_float: float,
    peak_sigma_mz_float: float, intensity_scalars: list[float], update_queue: queue.Queue | None,
    isotopic_enabled: bool, resolution: float
):
    all_clean_spectra = []
    num_proteins, progress_per_protein = len(protein_masses_list), (50 / len(protein_masses_list)) if len(protein_masses_list) > 0 else 0
    try:
        for i, protein_mass in enumerate(protein_masses_list):
            if update_queue: update_queue.put(('log', f"Generating base spectrum for Protein (Mass: {protein_mass})...\n"))
            all_clean_spectra.append(generate_protein_spectrum(protein_mass, mz_range, mz_step_float, peak_sigma_mz_float, intensity_scalars[i], isotopic_enabled, resolution))
            if update_queue: update_queue.put(('progress_add', progress_per_protein))
        return mz_range, all_clean_spectra
    except Exception as e:
        if update_queue: update_queue.put(('error', f"Error during base spectrum generation: {e}"))
        return None, None

def generate_gaussian_scaled_spectra(
    mz_range: np.ndarray, all_clean_spectra: list[np.ndarray], num_scans: int,
    gaussian_std_dev: float, update_queue: queue.Queue | None, seed: int, noise_option: str
):
    if num_scans <= 0:
        if update_queue: update_queue.put(('log', "Error: Number of scans must be positive.\n"))
        return []
    min_noise_level, baseline_offset = 0.01, 10.0; apex_scan_index = (num_scans - 1) / 2.0
    scan_index_values = np.arange(num_scans); lc_scaling_factors = np.exp(-((scan_index_values - apex_scan_index) ** 2) / (2 * max(1e-6, gaussian_std_dev) ** 2))
    gaussian_scaled_spectra_all_proteins, rng = [], default_rng(seed)
    num_proteins = len(all_clean_spectra); progress_per_scan = (40 / (num_proteins * num_scans)) if (num_proteins * num_scans) > 0 else 0
    if update_queue: update_queue.put(('log', "Applying LC profile and scan-level noise...\n"))
    for protein_idx, base_spectrum in enumerate(all_clean_spectra):
        spectra_for_protein, max_intensity_for_noise = [], np.max(base_spectrum) if base_spectrum.size > 0 else 0
        for scan_idx in range(num_scans):
            scaled_intensity_array = base_spectrum * lc_scaling_factors[scan_idx] * rng.normal(loc=1.0, scale=0.05, size=len(mz_range))
            if noise_option != "No Noise":
                noise_params = noise_presets.get(noise_option)
                if noise_params: scaled_intensity_array = add_noise(mz_range, scaled_intensity_array, min_noise_level=min_noise_level, max_intensity=max_intensity_for_noise, seed=seed + protein_idx * num_scans + scan_idx, **noise_params)
                elif update_queue: update_queue.put(('log', f"Warning: Noise preset '{noise_option}' not found. Skipping noise.\n"))
            spectra_for_protein.append(np.maximum(0, scaled_intensity_array + baseline_offset))
            if update_queue: update_queue.put(('progress_add', progress_per_scan))
        gaussian_scaled_spectra_all_proteins.append(spectra_for_protein)
    return gaussian_scaled_spectra_all_proteins

def generate_binding_spectrum(
    protein_avg_mass: float, compound_avg_mass: float, mz_range: np.ndarray, mz_step_float: float,
    peak_sigma_mz_float: float, total_binding_percentage: float, dar2_percentage_of_bound: float,
    original_intensity_scalar: float, isotopic_enabled: bool, resolution: float
):
    native_intensity_scalar = original_intensity_scalar * (100 - total_binding_percentage) / 100.0
    native_spectrum = generate_protein_spectrum(protein_avg_mass, mz_range, mz_step_float, peak_sigma_mz_float, native_intensity_scalar, isotopic_enabled, resolution)
    total_bound_intensity = original_intensity_scalar * (total_binding_percentage / 100.0)
    dar2_spectrum, dar2_intensity_scalar = np.zeros_like(mz_range), 0.0
    if total_binding_percentage > 0 and dar2_percentage_of_bound > 0:
        dar2_intensity_scalar = total_bound_intensity * (dar2_percentage_of_bound / 100.0)
        if dar2_intensity_scalar > 0: dar2_spectrum = generate_protein_spectrum(protein_avg_mass + 2 * compound_avg_mass, mz_range, mz_step_float, peak_sigma_mz_float, dar2_intensity_scalar, isotopic_enabled, resolution)
    dar1_spectrum, dar1_intensity_scalar = np.zeros_like(mz_range), total_bound_intensity - dar2_intensity_scalar
    if dar1_intensity_scalar > 0: dar1_spectrum = generate_protein_spectrum(protein_avg_mass + compound_avg_mass, mz_range, mz_step_float, peak_sigma_mz_float, dar1_intensity_scalar, isotopic_enabled, resolution)
    return native_spectrum + dar1_spectrum + dar2_spectrum

# --- mzML Utilities (Optimized) ---
def encode_floats(float_array: np.ndarray) -> str:
    compressed_bytes = zlib.compress(float_array.astype(np.float64).tobytes(), level=1)
    return base64.b64encode(compressed_bytes).decode('ascii')
def create_unique_filename(filepath: str) -> str:
    base, ext = os.path.splitext(filepath); counter = 1
    while os.path.exists(filepath): filepath = f"{base}_{counter}{ext}"; counter += 1
    return filepath

def create_mzml_content_et(mz_range: np.ndarray, run_data: list[list[np.ndarray]], scan_interval: float, update_queue: queue.Queue | None) -> bytes | None:
    if update_queue: update_queue.put(('log', "Generating mzML structure...\n"))
    try:
        if not run_data or not run_data[0]:
            if update_queue: update_queue.put(('error', "No spectrum data to write to mzML."))
            return None
        num_proteins, num_scans_per_protein = len(run_data), len(run_data[0]); total_spectra_count = num_proteins * num_scans_per_protein
        scan_times = [i * scan_interval for i in range(num_scans_per_protein)]
        mzml = ET.Element("mzML", {"xmlns": "http://psi.hupo.org/ms/mzml", "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance", "xsi:schemaLocation": "http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd", "version": "1.1.0", "id": "mzML_Generated_Example"})
        cv_list = ET.SubElement(mzml, "cvList", count="2"); ET.SubElement(cv_list, "cv", id="MS", fullName="Proteomics Standards Initiative Mass Spectrometry Ontology", URI="https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"); ET.SubElement(cv_list, "cv", id="UO", fullName="Unit Ontology", URI="https://raw.githubusercontent.com/bio-ontology-research-group/unit-ontology/master/unit.obo")
        file_desc = ET.SubElement(mzml, "fileDescription"); file_content = ET.SubElement(file_desc, "fileContent"); ET.SubElement(file_content, "cvParam", cvRef="MS", accession="MS:1000580", name="MSn spectrum"); ET.SubElement(file_content, "cvParam", cvRef="MS", accession="MS:1000579", name="MS1 spectrum"); ET.SubElement(file_content, "cvParam", cvRef="MS", accession="MS:1000128", name="profile spectrum")
        sw_list = ET.SubElement(mzml, "softwareList", count="1"); sw = ET.SubElement(sw_list, "software", id="sw_default", version="3.5-final"); ET.SubElement(sw, "cvParam", cvRef="MS", accession="MS:1000799", name="custom unreleased software tool", value="Spectrum Generator Script")
        dp_list = ET.SubElement(mzml, "dataProcessingList", count="1"); dp = ET.SubElement(dp_list, "dataProcessing", id="dp_default"); pm = ET.SubElement(dp, "processingMethod", order="1", softwareRef="sw_default"); ET.SubElement(pm, "cvParam", cvRef="MS", accession="MS:1000544", name="data processing action")
        run = ET.SubElement(mzml, "run", id="simulated_lcms_run", defaultInstrumentConfigurationRef="ic_default")
        ic_list = ET.SubElement(run, "instrumentConfigurationList", count="1"); ic_elem = ET.SubElement(ic_list, "instrumentConfiguration", id="ic_default"); ET.SubElement(ic_elem, "cvParam", cvRef="MS", accession="MS:1000031", name="instrument model")
        spectrum_list = ET.SubElement(run, "spectrumList", count=str(total_spectra_count), defaultDataProcessingRef="dp_default")
        mz_binary, mz_array_len = encode_floats(mz_range), str(len(mz_range))
        spec_template = ET.Element("spectrum", defaultArrayLength=mz_array_len); ET.SubElement(spec_template, "cvParam", cvRef="MS", accession="MS:1000579", name="MS1 spectrum"); ET.SubElement(spec_template, "cvParam", cvRef="MS", accession="MS:1000511", name="ms level", value="1"); ET.SubElement(spec_template, "cvParam", cvRef="MS", accession="MS:1000130", name="positive scan"); ET.SubElement(spec_template, "cvParam", cvRef="MS", accession="MS:1000128", name="profile spectrum"); ET.SubElement(spec_template, "cvParam", cvRef="MS", accession="MS:1000504", name="base peak m/z", value="", unitCvRef="MS", unitAccession="MS:1000040", unitName="m/z"); ET.SubElement(spec_template, "cvParam", cvRef="MS", accession="MS:1000505", name="base peak intensity", value="", unitCvRef="MS", unitAccession="MS:1000131", unitName="number of detector counts"); ET.SubElement(spec_template, "cvParam", cvRef="MS", accession="MS:1000285", name="total ion current", value=""); scan_list_template = ET.SubElement(spec_template, "scanList", count="1"); ET.SubElement(scan_list_template, "cvParam", cvRef="MS", accession="MS:1000795", name="no combination"); scan_template = ET.SubElement(scan_list_template, "scan"); ET.SubElement(scan_template, "cvParam", cvRef="MS", accession="MS:1000016", name="scan start time", value="", unitCvRef="UO", unitAccession="UO:0000031", unitName="minute"); bdal_template = ET.SubElement(spec_template, "binaryDataArrayList", count="2"); bda_mz_template = ET.SubElement(bdal_template, "binaryDataArray", encodedLength=str(len(mz_binary))); ET.SubElement(bda_mz_template, "cvParam", cvRef="MS", accession="MS:1000514", name="m/z array", unitCvRef="MS", unitAccession="MS:1000040", unitName="m/z"); ET.SubElement(bda_mz_template, "cvParam", cvRef="MS", accession="MS:1000523", name="64-bit float"); ET.SubElement(bda_mz_template, "cvParam", cvRef="MS", accession="MS:1000574", name="zlib compression"); ET.SubElement(bda_mz_template, "binary").text = mz_binary; bda_int_template = ET.SubElement(bdal_template, "binaryDataArray"); ET.SubElement(bda_int_template, "cvParam", cvRef="MS", accession="MS:1000515", name="intensity array", unitCvRef="MS", unitAccession="MS:1000131", unitName="number of detector counts"); ET.SubElement(bda_int_template, "cvParam", cvRef="MS", accession="MS:1000523", name="64-bit float"); ET.SubElement(bda_int_template, "cvParam", cvRef="MS", accession="MS:1000574", name="zlib compression"); ET.SubElement(bda_int_template, "binary").text = ""
        spectrum_index, native_id_counter = 0, 1
        for protein_spectra in run_data:
            for scan_idx, intensity_array in enumerate(protein_spectra):
                spec = copy.deepcopy(spec_template); spec.set("index", str(spectrum_index)); spec.set("id", f"scan={native_id_counter}")
                base_peak_intensity = np.max(intensity_array) if intensity_array.size > 0 else 0.0; base_peak_mz = mz_range[np.argmax(intensity_array)] if base_peak_intensity > 0 else 0.0; total_ion_current = np.sum(intensity_array); intensity_binary = encode_floats(intensity_array)
                spec.find(".//*[@accession='MS:1000504']").set('value', str(base_peak_mz)); spec.find(".//*[@accession='MS:1000505']").set('value', str(base_peak_intensity)); spec.find(".//*[@accession='MS:1000285']").set('value', str(total_ion_current))
                scan = spec.find("scanList/scan"); scan.set("id", f"scan={native_id_counter}"); scan.find(".//*[@accession='MS:1000016']").set('value', str(scan_times[scan_idx]))
                bda_int = spec.find("binaryDataArrayList/binaryDataArray[2]"); bda_int.set("encodedLength", str(len(intensity_binary))); bda_int.find("binary").text = intensity_binary
                spectrum_list.append(spec); spectrum_index += 1; native_id_counter += 1
        mzml_bytes_no_checksum = ET.tostring(mzml, encoding='utf-8', method='xml'); sha1_checksum = hashlib.sha1(mzml_bytes_no_checksum).hexdigest()
        file_checksum_tag = ET.SubElement(mzml, "fileChecksum"); ET.SubElement(file_checksum_tag, "cvParam", cvRef="MS", accession="MS:1000569", name="SHA-1", value=sha1_checksum)
        final_xml_bytes = b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(mzml, encoding='utf-8', method='xml')
        if update_queue: update_queue.put(('log', "mzML structure generated.\n"))
        return final_xml_bytes
    except Exception as e:
        if update_queue: update_queue.put(('error', f"Error during mzML content creation: {e}"))
        return None

# --- Main Application Logic & Helpers ---
def format_filename(template: str, values: dict) -> str:
    for key in re.findall(r'\{(.*?)\}', template):
        if key not in values: values[key] = ""
    formatted_name = template.format(**values)
    return re.sub(r'[\\/*?:"<>|]', "_", formatted_name)

def execute_simulation_and_write_mzml(
    update_queue: queue.Queue | None, protein_masses_str: str, mz_step_str: str, peak_sigma_mz_str: str,
    mz_range_start_str: str, mz_range_end_str: str, intensity_scalars: list[float], noise_option: str,
    seed: int, lc_simulation_enabled: bool, num_scans: int, scan_interval: float, gaussian_std_dev: float,
    final_filepath: str, isotopic_enabled: bool, resolution: float
) -> bool:
    try:
        protein_masses_list = [float(m.strip()) for m in protein_masses_str.split(",")]; mz_range_start_f, mz_range_end_f = float(mz_range_start_str), float(mz_range_end_str)
        mz_step_float, peak_sigma_mz_float = float(mz_step_str), float(peak_sigma_mz_str)
        if not protein_masses_list or not all(m > 0 for m in protein_masses_list) or mz_step_float <= 0 or peak_sigma_mz_float < 0 or mz_range_start_f >= mz_range_end_f or resolution < 0:
            raise ValueError("Invalid parameters provided.")
    except ValueError as e:
        if update_queue: update_queue.put(('error', f"Invalid parameters: {e}. Check inputs."))
        return False
    try:
        if not os.path.basename(final_filepath):
            if update_queue: update_queue.put(('error', "Filename template resulted in an empty name."))
            return False
        scans_to_gen = num_scans if lc_simulation_enabled else 1
        if update_queue: update_queue.put(('log', f"  Proteins: {len(protein_masses_list)} ({protein_masses_str})\n  m/z Range: {mz_range_start_f}-{mz_range_end_f}, Step: {mz_step_float}\n  Isotopes: {'Enabled' if isotopic_enabled else 'Disabled'}, Resolution: {resolution/1000}k\n  LC Simulation: {'Enabled' if lc_simulation_enabled else 'Disabled'} ({scans_to_gen} scans)\n  Noise: {noise_option}, Seed: {seed}\n"))
        if update_queue: update_queue.put(('progress_set', 5))
        mz_range = np.arange(mz_range_start_f, mz_range_end_f + mz_step_float, mz_step_float)
        mz_range, all_clean_spectra = generate_scaled_spectra(protein_masses_list, mz_range, mz_step_float, peak_sigma_mz_float, intensity_scalars, update_queue, isotopic_enabled, resolution)
        if mz_range is None: return False
        if update_queue: update_queue.put(('progress_set', 55))
        spectra_for_mzml = generate_gaussian_scaled_spectra(mz_range, all_clean_spectra, scans_to_gen, gaussian_std_dev, update_queue, seed, noise_option)
        mzml_content = create_mzml_content_et(mz_range, spectra_for_mzml, scan_interval if lc_simulation_enabled else 0.0, update_queue)
        if mzml_content is None: return False
        if update_queue: update_queue.put(('progress_set', 95))
        unique_filepath = create_unique_filename(final_filepath); os.makedirs(os.path.dirname(unique_filepath), exist_ok=True)
        if update_queue: update_queue.put(('log', f"Writing mzML file to: {os.path.basename(unique_filepath)}\n"))
        with open(unique_filepath, "wb") as outfile: outfile.write(mzml_content)
        if update_queue: update_queue.put(('log', "File successfully created.\n\n")); update_queue.put(('progress_set', 100))
        return True
    except Exception as e:
        if update_queue: update_queue.put(('error', f"An unexpected error occurred: {e}"))
        return False

def _parse_float_entry(entry_value: str, name: str) -> float:
    try: return float(entry_value)
    except ValueError: raise ValueError(f"Invalid value for {name}.")
def _parse_range_entry(entry_value: str, name: str) -> tuple[float, float]:
    try:
        parts = entry_value.split('-')
        if len(parts) != 2: raise ValueError
        start, end = float(parts[0]), float(parts[1])
        if start > end: raise ValueError("Start of range cannot be > end.")
        return start, end
    except (ValueError, IndexError): raise ValueError(f"Invalid range for {name}. Use 'start-end' format.")

def _read_protein_list_file(filepath: str) -> list[tuple[float, float]]:
    if not os.path.exists(filepath): raise FileNotFoundError(f"Protein list not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter='\t')
        header = [h.strip().lower() for h in next(reader)]
        try: mass_idx = header.index("protein")
        except ValueError: raise ValueError("File must contain a 'Protein' column header for masses.")
        try: intensity_idx = header.index("intensity")
        except ValueError: raise ValueError("File must contain an 'Intensity' column header for scalars.")
        proteins = []
        for i, row in enumerate(reader, 2):
            if not row: continue
            try:
                mass, scalar = float(row[mass_idx]), float(row[intensity_idx])
                if mass <= 0 or scalar < 0: raise ValueError("Mass must be > 0, and Intensity scalar must be >= 0.")
                proteins.append((mass, scalar))
            except (ValueError, IndexError) as e: raise ValueError(f"Invalid data on line {i}: {e}")
    if not proteins: raise ValueError("Protein list file is empty or contains no valid data rows.")
    return proteins

def _read_compound_list_file(filepath: str) -> list[tuple[str, float]]:
    if not os.path.exists(filepath): raise FileNotFoundError(f"Compound list not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter='\t')
        header = [h.strip().lower() for h in next(reader)]
        try: name_idx = header.index("name")
        except ValueError: raise ValueError("File must contain a 'Name' column header for compound names.")
        try: delta_idx = header.index("delta")
        except ValueError: raise ValueError("File must contain a 'Delta' column header for mass differences.")
        compounds = []
        for i, row in enumerate(reader, 2):
            if not row: continue
            try:
                name, mass = row[name_idx].strip(), float(row[delta_idx])
                if not name: raise ValueError("Compound name cannot be empty.")
                compounds.append((name, mass))
            except (ValueError, IndexError) as e: raise ValueError(f"Invalid data on line {i}: {e}")
    if not compounds: raise ValueError("Compound list is empty or contains no valid data rows.")
    return compounds

# --- GUI Class ---
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self); scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview); self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw"); canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", self._on_mousewheel); canvas.bind_all("<Button-4>", self._on_mousewheel); canvas.bind_all("<Button-5>", self._on_mousewheel)
    def _on_mousewheel(self, event):
        delta = 0
        if hasattr(event, 'delta') and event.delta != 0: delta = event.delta
        elif event.num == 5: delta = -1
        elif event.num == 4: delta = 1
        else: return
        self.winfo_children()[0].yview_scroll(-1 if delta > 0 else 1, "units")

class CombinedSpectrumSequenceApp:
    def __init__(self, master: tk.Tk):
        self.master = master; master.title("Simulated Spectrum Generator (Genedata Expressionist - 27Jun2025 v3.6 )")
        try: self.style = Style(theme="solar")
        except: self.style = Style(theme="litera")
        self.queue = queue.Queue(); self.process_queue()
        self.notebook = ttk.Notebook(master); self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        docs_tab = ttk.Frame(self.notebook)
        self.notebook.add(docs_tab, text="Overview & Docs")
        self.create_docs_tab_content(docs_tab)
        spectrum_tab = ttk.Frame(self.notebook); self.notebook.add(spectrum_tab, text="Spectrum Generator"); self.create_spectrum_generator_tab_content(spectrum_tab)
        binding_tab = ttk.Frame(self.notebook); self.notebook.add(binding_tab, text="Covalent Binding"); self.create_binding_spectra_tab_content(binding_tab)
        master.minsize(650, 600)

        # In CombinedSpectrumSequenceApp class...

    def create_docs_tab_content(self, tab):
        """Creates the content for the documentation and overview tab."""
        frame = ScrollableFrame(tab)
        frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        text_widget = tk.Text(frame.scrollable_frame, wrap=tk.WORD, relief=tk.FLAT, background=self.style.colors.bg)
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Define text styles ---
        text_widget.tag_configure("h1", font=("Helvetica", 16, "bold"), spacing3=10)
        text_widget.tag_configure("h2", font=("Helvetica", 12, "bold"), spacing3=8)
        text_widget.tag_configure("bold", font=("Helvetica", 10, "bold"))
        text_widget.tag_configure("body", font=("Helvetica", 10), lmargin1=10, lmargin2=10, spacing1=5)
        
        # --- Add content ---
        docs_content = [
            ("Spectrum Simulator (v3.6)\n\n", "h1"),
            ("This tool is designed to generate realistic, simulated mass spectrometry data (.mzML files) for intact proteins and covalent binding screens.\n\n", "body"),

            ("How it Works: The Core Model\n", "h2"),
            ("1. Isotopic Distribution:", "bold"),
            (" The simulation starts by calculating the theoretical isotopic distribution of a given mass using a Poisson distribution based on the 'averagine' model. It correctly uses the user-provided ", "body"),
            ("Average Mass", "bold"),
            (" to derive the corresponding monoisotopic mass.\n", "body"),
            ("2. Charge State Envelope:", "bold"),
            (" It then calculates a realistic charge state envelope. The charge states are centered around the most abundant isotope, ensuring the final deconvoluted mass matches the input average mass.\n", "body"),
            ("3. Peak Generation:", "bold"),
            (" For each isotope in each charge state, a Gaussian peak is generated. The final width of the peak is a combination of the ", "body"),
            ("Instrument Resolution", "bold"),
            (" (which determines the m/z-dependent broadening) and the ", "body"),
            ("Intrinsic Peak Sigma", "bold"),
            (" (which models constant physical effects like Doppler broadening).\n", "body"),
            ("4. Noise Simulation:", "bold"),
            (" Several layers of noise are added for realism, including chemical noise (low m/z 'haystack'), white electronic noise, signal-dependent shot noise, and a low-frequency baseline wobble. The noise generation is highly optimized using Numba.\n\n", "body"),
            
            ("Tabs Overview\n", "h2"),
            ("Spectrum Generator Tab\n", "bold"),
            ("This tab is for generating spectra for one or more proteins. You can enter masses manually as a comma-separated list or provide a tab-delimited file with 'Protein' and 'Intensity' columns. The 'Save as Template...' button provides an easy way to create a valid file from the manual inputs.\n\n", "body"),
            ("Covalent Binding Tab\n", "bold"),
            ("This tab simulates a covalent binding screen. It takes a single protein average mass and a list of compounds (from a file with 'Name' and 'Delta' columns). For each compound, it probabilistically determines if binding occurs (and to what extent) and generates a corresponding spectrum containing native, singly-adducted (DAR-1), and doubly-adducted (DAR-2) species.\n\n", "body"),
            
            ("Performance & Dependencies\n", "h2"),
            ("This application is highly optimized for performance and uses multiple CPU cores for batch processing. To run, it requires several external libraries. You can install them all with pip:\n", "body"),
            ("pip install ttkbootstrap numpy scipy numba lxml pandas matplotlib", "bold"),

        ]

        for text, tag in docs_content:
            text_widget.insert(tk.END, text, tag)
            
        text_widget.config(state=tk.DISABLED) # Make text read-only

    def process_queue(self):
        try:
            while True:
                msg_type, msg_data = self.queue.get_nowait()
                active_tab = self.notebook.index(self.notebook.select())
                out_text, prog_bar = (self.spectrum_output_text, self.progress_bar) if active_tab == 0 else (self.binding_output_text, self.binding_progress_bar)
                if msg_type == 'log': out_text.insert(tk.END, msg_data); out_text.see(tk.END)
                elif msg_type == 'clear_log': out_text.delete('1.0', tk.END)
                elif msg_type == 'progress_set': prog_bar["value"] = msg_data
                elif msg_type == 'progress_add': prog_bar["value"] += msg_data
                elif msg_type == 'error': messagebox.showerror("Error", msg_data); prog_bar["value"] = 0
                elif msg_type == 'warning': messagebox.showwarning("Warning", msg_data)
                elif msg_type == 'done':
                    if active_tab == 0: self.spectrum_generate_button.config(state=tk.NORMAL)
                    else: self.binding_generate_button.config(state=tk.NORMAL)
                    if msg_data: messagebox.showinfo("Complete", msg_data)
        except queue.Empty: pass
        finally: self.master.after(100, self.process_queue)

    def _show_plot(self, mz_range, intensity_data, title, xlabel="m/z", ylabel="Intensity"):
        try:
            plt.style.use('seaborn-v0_8-darkgrid'); fig, ax = plt.subplots(figsize=(10, 6))
            for label, data in intensity_data.items(): ax.plot(mz_range, data, label=label, lw=1.5)
            ax.set_title(title, fontsize=14); ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
            if len(intensity_data) > 1: ax.legend()
            ax.grid(True); fig.tight_layout(); plt.show()
        except Exception as e: messagebox.showerror("Plotting Error", f"Failed to show plot. Ensure matplotlib is installed correctly.\nError: {e}")

    def _browse_directory_for_var(self, string_var: tk.StringVar):
        directory = filedialog.askdirectory(initialdir=string_var.get())
        if directory: string_var.set(directory)

    def _create_common_parameters_frame(self, parent, mz_start, mz_end, noise):
        params = {}; mz_frame = ttk.LabelFrame(parent, text="m/z Parameters", padding=(15, 10)); mz_frame.grid(row=0, column=0, sticky=tk.EW, padx=10, pady=5)
        params['isotopic_enabled_var'] = tk.BooleanVar(value=False); params['isotopic_enabled_check'] = ttk.Checkbutton(mz_frame, text="Enable Isotopic Distribution", variable=params['isotopic_enabled_var'], bootstyle="primary-round-toggle"); params['isotopic_enabled_check'].grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5); Tooltip(params['isotopic_enabled_check'], "If enabled, simulates the isotopic distribution for each charge state.\nIf disabled, generates a single peak for each charge state.")
        ttk.Label(mz_frame, text="Instrument Resolution (in thousands):").grid(row=1, column=0, sticky=tk.W, pady=2); params['resolution_entry'] = ttk.Entry(mz_frame, width=10); params['resolution_entry'].insert(0, "120"); params['resolution_entry'].grid(row=1, column=1, sticky=tk.W, pady=2, padx=5); Tooltip(params['resolution_entry'], "The resolving power of the mass analyzer (e.g., 120 for 120,000).\nAffects the width of the generated peaks.")
        ttk.Label(mz_frame, text="Intrinsic Peak Sigma (at 1000 m/z):").grid(row=2, column=0, sticky=tk.W, pady=2); params['peak_sigma_mz_entry'] = ttk.Entry(mz_frame, width=10); params['peak_sigma_mz_entry'].insert(0, "0.01"); params['peak_sigma_mz_entry'].grid(row=2, column=1, sticky=tk.W, pady=2, padx=5); Tooltip(params['peak_sigma_mz_entry'], "The 'natural' peak width (standard deviation) from effects like Doppler broadening,\nindependent of instrument resolution. A small value is recommended.")
        ttk.Label(mz_frame, text="m/z Step:").grid(row=3, column=0, sticky=tk.W, pady=2); params['mz_step_entry'] = ttk.Entry(mz_frame, width=10); params['mz_step_entry'].insert(0, "0.02"); params['mz_step_entry'].grid(row=3, column=1, sticky=tk.W, pady=2, padx=5); Tooltip(params['mz_step_entry'], "The distance between data points in the m/z axis.")
        ttk.Label(mz_frame, text="m/z Range Start:").grid(row=4, column=0, sticky=tk.W, pady=2); params['mz_range_start_entry'] = ttk.Entry(mz_frame, width=10); params['mz_range_start_entry'].insert(0, mz_start); params['mz_range_start_entry'].grid(row=4, column=1, sticky=tk.W, pady=2, padx=5); Tooltip(params['mz_range_start_entry'], "The minimum m/z value for the spectrum.")
        ttk.Label(mz_frame, text="m/z Range End:").grid(row=5, column=0, sticky=tk.W, pady=2); params['mz_range_end_entry'] = ttk.Entry(mz_frame, width=10); params['mz_range_end_entry'].insert(0, mz_end); params['mz_range_end_entry'].grid(row=5, column=1, sticky=tk.W, pady=2, padx=5); Tooltip(params['mz_range_end_entry'], "The maximum m/z value for the spectrum.")
        out_frame = ttk.LabelFrame(parent, text="Noise & Output", padding=(15, 10)); out_frame.grid(row=1, column=0, sticky=tk.EW, padx=10, pady=5); out_frame.columnconfigure(1, weight=1)
        ttk.Label(out_frame, text="Noise Level:").grid(row=0, column=0, sticky=tk.W, pady=2); params['noise_option_var'] = tk.StringVar(value=noise); params['noise_option_combobox'] = ttk.Combobox(out_frame, textvariable=params['noise_option_var'], values=["No Noise"] + list(noise_presets.keys()), state="readonly", width=15); params['noise_option_combobox'].grid(row=0, column=1, sticky=tk.W, pady=2, padx=5); Tooltip(params['noise_option_combobox'], "Select a preset for the type and amount of noise to add to the spectrum.")
        ttk.Label(out_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=2); params['output_directory_var'] = tk.StringVar(value=os.getcwd()); params['output_directory_entry'] = ttk.Entry(out_frame, textvariable=params['output_directory_var']); params['output_directory_entry'].grid(row=1, column=1, sticky=tk.EW, pady=2, padx=5); Tooltip(params['output_directory_entry'], "The folder where the generated .mzML files will be saved.")
        params['browse_button'] = ttk.Button(out_frame, text="Browse...", command=partial(self._browse_directory_for_var, params['output_directory_var']), style='Outline.TButton'); params['browse_button'].grid(row=1, column=2, sticky=tk.W, pady=2, padx=5); Tooltip(params['browse_button'], "Browse for an output directory.")
        ttk.Label(out_frame, text="Random Seed:").grid(row=2, column=0, sticky=tk.W, pady=2); params['seed_var'] = tk.StringVar(value=""); params['seed_entry'] = ttk.Entry(out_frame, textvariable=params['seed_var'], width=15); params['seed_entry'].grid(row=2, column=1, sticky=tk.W, pady=2, padx=5); Tooltip(params['seed_entry'], "Seed for the random number generator to ensure reproducible noise.\nLeave blank for a different random seed each time.")
        ttk.Label(out_frame, text="(Leave blank for random)").grid(row=2, column=2, sticky=tk.W, pady=2, padx=5)
        ttk.Label(out_frame, text="Filename Template:").grid(row=3, column=0, sticky=tk.W, pady=2); params['filename_template_var'] = tk.StringVar(); params['filename_template_entry'] = ttk.Entry(out_frame, textvariable=params['filename_template_var']); params['filename_template_entry'].grid(row=3, column=1, columnspan=2, sticky=tk.EW, pady=2, padx=5); Tooltip(params['filename_template_entry'], "Define the pattern for output filenames using available tags.")
        placeholder_text = "Tags: {date} {time} {protein_mass} {compound_name} {num_proteins} {scalar} {scans} {noise} {seed}"; ttk.Label(out_frame, text=placeholder_text, wraplength=350, justify=tk.LEFT, bootstyle="secondary").grid(row=4, column=1, columnspan=2, sticky=tk.W, padx=5)
        return params

    def _create_lc_simulation_frame(self, parent, enabled_by_default=False):
        lc_params = {}; container = ttk.Frame(parent)
        lc_params['enabled_var'] = tk.BooleanVar(value=enabled_by_default); lc_check = ttk.Checkbutton(container, text="Enable LC Simulation", variable=lc_params['enabled_var'], bootstyle="primary-round-toggle"); lc_check.grid(row=0, column=0, sticky=tk.W, pady=(0, 5)); Tooltip(lc_check, "If enabled, simulates a chromatographic peak by generating multiple scans with varying intensity.\nIf disabled, generates a single scan (spectrum).")
        lc_frame = ttk.LabelFrame(container, text="LC Simulation Parameters", padding=(15, 10)); lc_frame.grid(row=1, column=0, sticky=tk.EW)
        ttk.Label(lc_frame, text="Number of Scans:").grid(row=0, column=0, sticky=tk.W, pady=2); lc_params['num_scans_entry'] = ttk.Entry(lc_frame, width=10); lc_params['num_scans_entry'].insert(0, "10"); lc_params['num_scans_entry'].grid(row=0, column=1, sticky=tk.W, pady=2, padx=5); Tooltip(lc_params['num_scans_entry'], "The total number of scans to generate across the LC peak.")
        ttk.Label(lc_frame, text="Scan Interval (min):").grid(row=1, column=0, sticky=tk.W, pady=2); lc_params['scan_interval_entry'] = ttk.Entry(lc_frame, width=10); lc_params['scan_interval_entry'].insert(0, "0.05"); lc_params['scan_interval_entry'].grid(row=1, column=1, sticky=tk.W, pady=2, padx=5); Tooltip(lc_params['scan_interval_entry'], "The simulated time between consecutive scans.")
        ttk.Label(lc_frame, text="LC Peak Std Dev (scans):").grid(row=2, column=0, sticky=tk.W, pady=2); lc_params['gaussian_std_dev_entry'] = ttk.Entry(lc_frame, width=10); lc_params['gaussian_std_dev_entry'].insert(0, "1"); lc_params['gaussian_std_dev_entry'].grid(row=2, column=1, sticky=tk.W, pady=2, padx=5); Tooltip(lc_params['gaussian_std_dev_entry'], "The width (standard deviation) of the LC peak, in units of scans.\nHigher values create a wider chromatographic peak.")
        def _toggle(): state = tk.NORMAL if lc_params['enabled_var'].get() else tk.DISABLED; [w.configure(state=state) for w in lc_frame.winfo_children()]
        lc_check.config(command=_toggle); _toggle(); return container, lc_params

    def create_spectrum_generator_tab_content(self, tab):
        frame = ScrollableFrame(tab); frame.pack(fill="both", expand=True); main = frame.scrollable_frame
        in_frame = ttk.LabelFrame(main, text="Protein Parameters", padding=(15, 10)); in_frame.grid(row=0, column=0, sticky=tk.EW, padx=10, pady=10); in_frame.columnconfigure(1, weight=1)
        ttk.Label(in_frame, text="Protein List File (.txt):").grid(row=0, column=0, sticky=tk.W, pady=5); self.protein_list_file_var = tk.StringVar(); self.protein_list_file_entry = ttk.Entry(in_frame, textvariable=self.protein_list_file_var); self.protein_list_file_entry.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=5); Tooltip(self.protein_list_file_entry, "Path to a tab-delimited file with protein data.\nMust contain 'Protein' (Average Mass) and 'Intensity' headers.")
        in_frame.columnconfigure(2, weight=0); in_frame.columnconfigure(3, weight=0); self.protein_list_browse_button = ttk.Button(in_frame, text="Browse...", command=self.browse_protein_list, style='Outline.TButton'); self.protein_list_browse_button.grid(row=0, column=2, sticky=tk.W, pady=5, padx=(5,0)); Tooltip(self.protein_list_browse_button, "Browse for a protein list file."); self.save_template_button = ttk.Button(in_frame, text="Save as Template...", command=self._save_protein_template, style='Outline.TButton'); self.save_template_button.grid(row=0, column=3, sticky=tk.W, pady=5, padx=(2,5)); Tooltip(self.save_template_button, "Save the manually entered masses and scalars below as a valid\ntab-delimited template file for future use.")
        ttk.Separator(in_frame, orient=tk.HORIZONTAL).grid(row=1, column=0, columnspan=4, sticky=tk.EW, pady=10); ttk.Label(in_frame, text="OR Enter Manually Below", bootstyle="secondary").grid(row=2, column=0, columnspan=4)
        ttk.Label(in_frame, text="Protein Avg. Masses (Da, comma-sep):").grid(row=3, column=0, sticky=tk.W, pady=5); self.spectrum_protein_masses_entry = ttk.Entry(in_frame); self.spectrum_protein_masses_entry.insert(0, "25000"); self.spectrum_protein_masses_entry.grid(row=3, column=1, columnspan=3, sticky=tk.EW, pady=5, padx=5); Tooltip(self.spectrum_protein_masses_entry, "A comma-separated list of protein AVERAGE masses to simulate.")
        ttk.Label(in_frame, text="Intensity Scalars (comma-sep):").grid(row=4, column=0, sticky=tk.W, pady=5); self.intensity_scalars_entry = ttk.Entry(in_frame); self.intensity_scalars_entry.insert(0, "1.0"); self.intensity_scalars_entry.grid(row=4, column=1, columnspan=3, sticky=tk.EW, pady=5, padx=5); Tooltip(self.intensity_scalars_entry, "A comma-separated list of relative intensity multipliers.\nMust match the number of protein masses.")
        common_frame = ttk.Frame(main); common_frame.grid(row=1, column=0, sticky=tk.EW, padx=10, pady=0); self.spec_gen_params = self._create_common_parameters_frame(common_frame, "400.0", "2500.0", "Default Noise"); self.spec_gen_params['output_directory_var'].set(os.path.join(os.getcwd(), "Mzml Mock Spectra")); self.spec_gen_params['filename_template_var'].set("{date}_protein_{protein_mass}_{scans}scans_{noise}.mzML")
        lc_container, self.spec_gen_lc_params = self._create_lc_simulation_frame(main, False); lc_container.grid(row=2, column=0, sticky=tk.EW, padx=10, pady=10)
        button_frame = ttk.Frame(main); button_frame.grid(row=3, column=0, pady=15); self.spectrum_preview_button = ttk.Button(button_frame, text="Preview Spectrum", command=self._preview_spectrum_tab_command, style='Outline.TButton'); self.spectrum_preview_button.pack(side=tk.LEFT, padx=5); Tooltip(self.spectrum_preview_button, "Generate and display a plot of a single spectrum using the current settings.\nUses the first protein mass if multiple are entered."); self.spectrum_generate_button = ttk.Button(button_frame, text="Generate mzML File(s)", command=self.generate_spectrum_tab_command, bootstyle=PRIMARY); self.spectrum_generate_button.pack(side=tk.LEFT, padx=5); Tooltip(self.spectrum_generate_button, "Generate and save .mzML file(s) with the specified parameters.")
        self.progress_bar = ttk.Progressbar(main, orient=tk.HORIZONTAL, mode="determinate"); self.progress_bar.grid(row=4, column=0, pady=5, sticky=tk.EW, padx=10)
        out_frame = ttk.Frame(main); out_frame.grid(row=5, column=0, sticky=tk.NSEW, padx=10, pady=(5, 10)); out_frame.columnconfigure(0, weight=1); out_frame.rowconfigure(0, weight=1); main.rowconfigure(5, weight=1); self.spectrum_output_text = tk.Text(out_frame, height=10, wrap=tk.WORD, relief=tk.SUNKEN, borderwidth=1); self.spectrum_output_text.grid(row=0, column=0, sticky=tk.NSEW); scrollbar = ttk.Scrollbar(out_frame, command=self.spectrum_output_text.yview); scrollbar.grid(row=0, column=1, sticky=tk.NS); self.spectrum_output_text['yscrollcommand'] = scrollbar.set
        self.protein_list_file_var.trace_add("write", self._toggle_protein_inputs); self._toggle_protein_inputs()

    def _save_protein_template(self):
        try:
            mass_str, scalar_str = self.spectrum_protein_masses_entry.get(), self.intensity_scalars_entry.get()
            masses = [m.strip() for m in mass_str.split(',') if m.strip()]; scalars = [s.strip() for s in scalar_str.split(',') if s.strip()]
            if not masses: messagebox.showerror("Error", "No protein masses entered to save."); return
            if len(masses) != len(scalars):
                if messagebox.askokcancel("Warning", "The number of masses and intensity scalars do not match. Continue with 1.0 for missing scalars?"): scalars = (scalars + ['1.0'] * len(masses))[:len(masses)]
                else: return
            filepath = filedialog.asksaveasfilename(title="Save Protein List Template", initialfile="protein_list_template.txt", defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
            if not filepath: return
            with open(filepath, 'w', newline='', encoding='utf-8') as f: writer = csv.writer(f, delimiter='\t'); writer.writerow(['Protein', 'Intensity']); writer.writerows(zip(masses, scalars))
            self.queue.put(('log', f"Saved template to {os.path.basename(filepath)}\n")); self.protein_list_file_var.set(filepath)
        except Exception as e: messagebox.showerror("Save Error", f"Could not save template file.\nError: {e}")

    def _toggle_protein_inputs(self, *args): state = tk.DISABLED if self.protein_list_file_var.get() else tk.NORMAL; self.spectrum_protein_masses_entry.config(state=state); self.intensity_scalars_entry.config(state=state)
    def browse_protein_list(self): filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt;*.tsv")], initialdir=os.getcwd()); (self.protein_list_file_var.set(filepath) if filepath else None)
    def generate_spectrum_tab_command(self): self.spectrum_generate_button.config(state=tk.DISABLED); self.progress_bar["value"] = 0; self.queue.put(('clear_log', None)); worker = self._worker_generate_from_protein_file if self.protein_list_file_var.get() else self._worker_generate_from_manual_input; threading.Thread(target=worker, daemon=True).start()

    def _get_common_gen_params(self, params_dict, lc_params_dict):
        params = {}; [params.update({key.replace('_var', ''): widget.get()}) for key, widget in params_dict.items() if 'var' in key]; [params.update({key.replace('_entry', ''): widget.get()}) for key, widget in params_dict.items() if 'entry' in key]
        params['lc_simulation_enabled'] = lc_params_dict['enabled_var'].get()
        if params['lc_simulation_enabled']: params['num_scans'], params['scan_interval'], params['gaussian_std_dev'] = int(lc_params_dict['num_scans_entry'].get()), float(lc_params_dict['scan_interval_entry'].get()), float(lc_params_dict['gaussian_std_dev_entry'].get())
        else: params['num_scans'], params['scan_interval'], params['gaussian_std_dev'] = 1, 0.0, 0.0
        seed_str = params_dict['seed_var'].get().strip(); params['seed'] = int(seed_str) if seed_str else random.randint(0, 2**32 - 1); (params_dict['seed_var'].set(str(params['seed'])) if not seed_str else None)
        params['resolution'] = float(params_dict['resolution_entry'].get()) * 1000
        return params

    def _worker_generate_from_protein_file(self):
        try: protein_list = _read_protein_list_file(self.protein_list_file_var.get()); common_params = self._get_common_gen_params(self.spec_gen_params, self.spec_gen_lc_params)
        except (ValueError, FileNotFoundError) as e: self.queue.put(('error', str(e))); self.queue.put(('done', None)); return
        jobs = [(mass, scalar, common_params, common_params['seed'] + i) for i, (mass, scalar) in enumerate(protein_list)]
        self.queue.put(('log', f"Starting batch generation for {len(jobs)} proteins using {os.cpu_count()} processes...\n\n")); self.progress_bar["maximum"] = len(jobs); self.progress_bar["value"] = 0
        try:
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                success_count = 0
                for i, (success, message) in enumerate(pool.imap_unordered(run_simulation_task, jobs)):
                    self.queue.put(('log', message)); success_count += success
                    self.queue.put(('progress_set', i + 1))
            self.queue.put(('done', f"Batch complete. Generated {success_count} of {len(jobs)} mzML files."))
        except Exception as e: self.queue.put(('error', f"A multiprocessing error occurred: {e}")); self.queue.put(('done', None))

    def _worker_generate_from_manual_input(self):
        try:
            common_params = self._get_common_gen_params(self.spec_gen_params, self.spec_gen_lc_params); mass_str = self.spectrum_protein_masses_entry.get(); mass_list = [m.strip() for m in mass_str.split(',') if m.strip()]; scalar_str = self.intensity_scalars_entry.get(); scalar_list = [float(s.strip()) for s in scalar_str.split(',') if s.strip()]
            if not mass_list: raise ValueError("No protein masses entered.")
            if not scalar_list: scalar_list = [1.0] * len(mass_list)
            if len(scalar_list) != len(mass_list): self.queue.put(('warning', "Mismatched scalars and masses. Adjusting...")); scalar_list = (scalar_list + [1.0] * len(mass_list))[:len(mass_list)]
        except ValueError as e: self.queue.put(('error', f"Invalid input: {e}")); self.queue.put(('done', None)); return
        avg_mass = float(mass_list[0])
        placeholders = {"date": datetime.now().strftime('%Y-%m-%d'), "time": datetime.now().strftime('%H%M%S'), "num_proteins": len(mass_list), "scans": common_params['num_scans'], "noise": common_params['noise_option'].replace(" ", ""), "seed": common_params['seed'], "protein_mass": int(round(avg_mass))}; filename = format_filename(common_params['filename_template'], placeholders); filepath = os.path.join(common_params['output_directory'], filename)
        if execute_simulation_and_write_mzml(self.queue, mass_str, common_params['mz_step'], common_params['peak_sigma_mz'], common_params['mz_range_start'], common_params['mz_range_end'], scalar_list, common_params['noise_option'], common_params['seed'], common_params['lc_simulation_enabled'], common_params['num_scans'], common_params['scan_interval'], common_params['gaussian_std_dev'], filepath, common_params['isotopic_enabled'], common_params['resolution']): self.queue.put(('done', "mzML file successfully created."))
        else: self.queue.put(('done', None))

    def create_binding_spectra_tab_content(self, tab):
        frame = ScrollableFrame(tab); frame.pack(fill="both", expand=True); main = frame.scrollable_frame
        in_frame = ttk.LabelFrame(main, text="Target & Compound", padding=(15, 10)); in_frame.grid(row=0, column=0, sticky=tk.EW, padx=10, pady=10); in_frame.columnconfigure(1, weight=1)
        ttk.Label(in_frame, text="Protein Avg. Mass (Da):").grid(row=0, column=0, sticky=tk.W, pady=5); self.binding_protein_mass_entry = ttk.Entry(in_frame); self.binding_protein_mass_entry.insert(0, "25000"); self.binding_protein_mass_entry.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=5); Tooltip(self.binding_protein_mass_entry, "The AVERAGE mass of the target protein.")
        ttk.Label(in_frame, text="Compound List File (.txt):").grid(row=1, column=0, sticky=tk.W, pady=5); self.compound_list_file_var = tk.StringVar(); self.compound_list_file_entry = ttk.Entry(in_frame, textvariable=self.compound_list_file_var); self.compound_list_file_entry.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=5); Tooltip(self.compound_list_file_entry, "Path to a tab-delimited file of compounds to test.\nMust contain 'Name' and 'Delta' (Average Mass) headers."); self.compound_list_browse_button = ttk.Button(in_frame, text="Browse...", command=self.browse_compound_list, style='Outline.TButton'); self.compound_list_browse_button.grid(row=1, column=2, sticky=tk.W, pady=5, padx=5); Tooltip(self.compound_list_browse_button, "Browse for a compound list file.")
        common_frame = ttk.Frame(main); common_frame.grid(row=1, column=0, sticky=tk.EW, padx=10, pady=0); self.binding_params = self._create_common_parameters_frame(common_frame, "400.0", "2000.0", "Default Noise"); self.binding_params['output_directory_var'].set(os.path.join(os.getcwd(), "Intact Covalent Binding Mock Spectra")); self.binding_params['filename_template_var'].set("{date}_{compound_name}_on_{protein_mass}_{scans}scans_{noise}.mzML")
        lc_container, self.binding_lc_params = self._create_lc_simulation_frame(main, False); lc_container.grid(row=2, column=0, sticky=tk.EW, padx=10, pady=10)
        prob_frame = ttk.LabelFrame(main, text="Binding Probabilities", padding=(15, 10)); prob_frame.grid(row=3, column=0, sticky=tk.EW, padx=10, pady=10)
        ttk.Label(prob_frame, text="Probability of Binding:").grid(row=0, column=0, sticky=tk.W, pady=2); self.prob_binding_entry = ttk.Entry(prob_frame, width=10); self.prob_binding_entry.insert(0, "0.5"); self.prob_binding_entry.grid(row=0, column=1, sticky=tk.W, pady=2, padx=5); Tooltip(self.prob_binding_entry, "The chance (0.0 to 1.0) that any given compound will bind to the protein.")
        ttk.Label(prob_frame, text="Probability of DAR-2 (if Binding):").grid(row=1, column=0, sticky=tk.W, pady=2); self.prob_dar2_if_binding_entry = ttk.Entry(prob_frame, width=10); self.prob_dar2_if_binding_entry.insert(0, "0.1"); self.prob_dar2_if_binding_entry.grid(row=1, column=1, sticky=tk.W, pady=2, padx=5); Tooltip(self.prob_dar2_if_binding_entry, "If a compound binds, this is the chance (0.0 to 1.0)\nthat it will form a doubly-adducted species (DAR-2).")
        ttk.Label(prob_frame, text="Total Binding % Range (if Binding):").grid(row=2, column=0, sticky=tk.W, pady=2); self.total_binding_percentage_range_entry = ttk.Entry(prob_frame); self.total_binding_percentage_range_entry.insert(0, "10-50"); self.total_binding_percentage_range_entry.grid(row=2, column=1, sticky=tk.EW, pady=2, padx=5); Tooltip(self.total_binding_percentage_range_entry, "If binding occurs, the total percentage of protein that is modified\nwill be randomly chosen from this range (e.g., '10-50').")
        ttk.Label(prob_frame, text="DAR-2 % Range (of Total Bound):").grid(row=3, column=0, sticky=tk.W, pady=2); self.dar2_percentage_of_bound_range_entry = ttk.Entry(prob_frame); self.dar2_percentage_of_bound_range_entry.insert(0, "5-20"); self.dar2_percentage_of_bound_range_entry.grid(row=3, column=1, sticky=tk.EW, pady=2, padx=5); Tooltip(self.dar2_percentage_of_bound_range_entry, "If DAR-2 occurs, the percentage of the bound protein that is DAR-2\nwill be randomly chosen from this range (e.g., '5-20').")
        button_frame = ttk.Frame(main); button_frame.grid(row=4, column=0, pady=15); self.binding_preview_button = ttk.Button(button_frame, text="Preview Binding", command=self._preview_binding_tab_command, style='Outline.TButton'); self.binding_preview_button.pack(side=tk.LEFT, padx=5); Tooltip(self.binding_preview_button, "Generate and display a plot of a binding spectrum using average probability values."); self.binding_generate_button = ttk.Button(button_frame, text="Generate Binding Spectra", command=self.generate_binding_spectra_command, bootstyle=PRIMARY); self.binding_generate_button.pack(side=tk.LEFT, padx=5); Tooltip(self.binding_generate_button, "Generate an .mzML file for each compound in the list,\nwith binding determined by the specified probabilities.")
        self.binding_progress_bar = ttk.Progressbar(main, orient=tk.HORIZONTAL, mode="determinate"); self.binding_progress_bar.grid(row=5, column=0, pady=5, sticky=tk.EW, padx=10)
        out_frame = ttk.Frame(main); out_frame.grid(row=6, column=0, sticky=tk.NSEW, padx=10, pady=(5, 10)); out_frame.columnconfigure(0, weight=1); out_frame.rowconfigure(0, weight=1); main.rowconfigure(6, weight=1); self.binding_output_text = tk.Text(out_frame, height=10, wrap=tk.WORD, relief=tk.SUNKEN, borderwidth=1); self.binding_output_text.grid(row=0, column=0, sticky=tk.NSEW); scrollbar = ttk.Scrollbar(out_frame, command=self.binding_output_text.yview); scrollbar.grid(row=0, column=1, sticky=tk.NS); self.binding_output_text['yscrollcommand'] = scrollbar.set

    def browse_compound_list(self): filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt;*.tsv")], initialdir=os.getcwd()); (self.compound_list_file_var.set(filepath) if filepath else None)
    def generate_binding_spectra_command(self): self.binding_generate_button.config(state=tk.DISABLED); self.binding_progress_bar["value"] = 0; self.queue.put(('clear_log', None)); threading.Thread(target=self._worker_generate_binding_spectra, daemon=True).start()

    def _worker_generate_binding_spectra(self):
        try:
            common_params = self._get_common_gen_params(self.binding_params, self.binding_lc_params); protein_mass = _parse_float_entry(self.binding_protein_mass_entry.get(), "Protein Avg. Mass"); compounds = _read_compound_list_file(self.compound_list_file_var.get())
            common_params['prob_binding'] = _parse_float_entry(self.prob_binding_entry.get(), "Prob Binding"); common_params['prob_dar2'] = _parse_float_entry(self.prob_dar2_if_binding_entry.get(), "Prob DAR-2"); common_params['total_binding_range'] = _parse_range_entry(self.total_binding_percentage_range_entry.get(), "Total Binding %"); common_params['dar2_range'] = _parse_range_entry(self.dar2_percentage_of_bound_range_entry.get(), "DAR-2 %")
        except (ValueError, FileNotFoundError) as e: self.queue.put(('error', str(e))); self.queue.put(('done', None)); return
        jobs = [(name, mass, protein_mass, common_params, common_params['seed'] + i) for i, (name, mass) in enumerate(compounds)]
        self.queue.put(('log', f"Starting batch generation for {len(jobs)} compounds using {os.cpu_count()} processes...\n\n")); self.binding_progress_bar["maximum"] = len(jobs); self.binding_progress_bar["value"] = 0
        try:
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                success_count = 0
                for i, (success, message) in enumerate(pool.imap_unordered(run_binding_task, jobs)):
                    self.queue.put(('log', message)); success_count += success
                    self.queue.put(('progress_set', i + 1))
            self.queue.put(('done', f"Batch complete. Generated {success_count} of {len(jobs)} binding mzML files."))
        except Exception as e: self.queue.put(('error', f"A multiprocessing error occurred: {e}")); self.queue.put(('done', None))

    def _preview_spectrum_tab_command(self):
        try:
            params = self._get_common_gen_params(self.spec_gen_params, self.spec_gen_lc_params); mass_str = self.spectrum_protein_masses_entry.get(); mass_list = [float(m.strip()) for m in mass_str.split(',') if m.strip()]
            if not mass_list: raise ValueError("Please enter at least one protein mass.")
            protein_avg_mass = mass_list[0]
            mz_range = np.arange(float(params['mz_range_start']), float(params['mz_range_end']) + float(params['mz_step']), float(params['mz_step']))
            clean_spec = generate_protein_spectrum(protein_avg_mass, mz_range, float(params['mz_step']), float(params['peak_sigma_mz']), BASE_INTENSITY_SCALAR, params['isotopic_enabled'], params['resolution'])
            apex_scan_spectrum = generate_gaussian_scaled_spectra(mz_range, [clean_spec], params['num_scans'], params['gaussian_std_dev'], None, params['seed'], params['noise_option'])[0][(params['num_scans']-1)//2]
            title=f"Preview (Avg Mass: {protein_avg_mass:.0f} Da, Res: {params['resolution']/1000}k)"
            self._show_plot(mz_range, {"Apex Scan Preview": apex_scan_spectrum}, title)
        except (ValueError, IndexError) as e: messagebox.showerror("Preview Error", f"Invalid parameters for preview: {e}")
        except Exception as e: messagebox.showerror("Preview Error", f"An unexpected error occurred during preview: {e}")

    def _preview_binding_tab_command(self):
        try:
            params = self._get_common_gen_params(self.binding_params, self.binding_lc_params); protein_avg_mass = _parse_float_entry(self.binding_protein_mass_entry.get(), "Protein Avg. Mass"); compound_avg_mass = 500.0; total_binding_range = _parse_range_entry(self.total_binding_percentage_range_entry.get(), "Total Binding %"); dar2_range = _parse_range_entry(self.dar2_percentage_of_bound_range_entry.get(), "DAR-2 %"); total_binding_pct = (total_binding_range[0] + total_binding_range[1]) / 2; dar2_pct_of_bound = (dar2_range[0] + dar2_range[1]) / 2
            mz_range = np.arange(float(params['mz_range_start']), float(params['mz_range_end']) + float(params['mz_step']), float(params['mz_step']))
            isotopic_enabled, resolution = params['isotopic_enabled'], params['resolution']
            base_native_spec = generate_protein_spectrum(protein_avg_mass, mz_range, float(params['mz_step']), float(params['peak_sigma_mz']), BASE_INTENSITY_SCALAR, isotopic_enabled, resolution)
            base_dar1_spec = generate_protein_spectrum(protein_avg_mass + compound_avg_mass, mz_range, float(params['mz_step']), float(params['peak_sigma_mz']), BASE_INTENSITY_SCALAR, isotopic_enabled, resolution)
            base_dar2_spec = generate_protein_spectrum(protein_avg_mass + 2 * compound_avg_mass, mz_range, float(params['mz_step']), float(params['peak_sigma_mz']), BASE_INTENSITY_SCALAR, isotopic_enabled, resolution)
            native_scalar = (100 - total_binding_pct) / 100.0; total_bound_scalar = total_binding_pct / 100.0
            dar2_scalar = total_bound_scalar * (dar2_pct_of_bound / 100.0); dar1_scalar = total_bound_scalar - dar2_scalar
            final_spec_clean = (base_native_spec * native_scalar) + (base_dar1_spec * dar1_scalar) + (base_dar2_spec * dar2_scalar)
            final_spec_noisy = generate_gaussian_scaled_spectra(mz_range, [final_spec_clean], params['num_scans'], params['gaussian_std_dev'], None, params['seed'], params['noise_option'])[0][(params['num_scans']-1)//2]
            plot_data = {"Combined Spectrum (with noise)": final_spec_noisy}
            if np.any(base_native_spec): plot_data["Native Protein (ref)"] = base_native_spec * 0.5
            if np.any(base_dar1_spec): plot_data["DAR-1 Adduct (ref)"] = base_dar1_spec * 0.5
            if np.any(base_dar2_spec): plot_data["DAR-2 Adduct (ref)"] = base_dar2_spec * 0.5
            title=f"Binding Preview (Target Avg Mass: ~{protein_avg_mass:.0f} Da, Res: {resolution/1000}k)"
            self._show_plot(mz_range, plot_data, title=title)
        except (ValueError, IndexError) as e: messagebox.showerror("Preview Error", f"Invalid parameters for preview: {e}")
        except Exception as e: messagebox.showerror("Preview Error", f"An unexpected error occurred during preview: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = CombinedSpectrumSequenceApp(root)
    root.mainloop()