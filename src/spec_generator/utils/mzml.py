import base64
import copy
import hashlib
import threading
import xml.etree.ElementTree as ET
import zlib
from typing import List, Optional

import numpy as np

from ..core.types import MSMSSpectrum
from ..core.types import FragmentationEvent


def encode_floats(float_array: np.ndarray) -> str:
    """Encodes a numpy array of floats into a zlib-compressed, base64-encoded string."""
    # Use zlib compression level 1 for a good balance of speed and size
    compressed_bytes = zlib.compress(float_array.astype(np.float64).tobytes(), level=1)
    return base64.b64encode(compressed_bytes).decode('ascii')


def create_mzml_content_et(
    mz_range: np.ndarray,
    run_data: list[list[np.ndarray]],
    scan_interval: float,
    progress_callback=None,
    msms_spectra: Optional[List[MSMSSpectrum]] = None,
    stop_event: Optional[threading.Event] = None,
) -> bytes | None:
    """
    Creates the full mzML file content as a byte string using xml.etree.ElementTree.
    This version is highly optimized by creating a template for the spectrum element
    and deep-copying it for each scan, which is much faster than creating each
    element from scratch in a loop.
    """
    progress_callback = progress_callback or (lambda *args: None)
    progress_callback('log', "Generating mzML structure...\n")

    try:
        if not run_data or not run_data[0]:
            progress_callback('error', "No spectrum data to write to mzML.")
            return None

        num_proteins = len(run_data)
        num_scans_per_protein = len(run_data[0])
        total_spectra_count = num_proteins * num_scans_per_protein
        if msms_spectra:
            total_spectra_count += len(msms_spectra)
        scan_times = [i * scan_interval for i in range(num_scans_per_protein)]

        # --- Boilerplate XML Setup ---
        mzml = ET.Element("mzML", {
            "xmlns": "http://psi.hupo.org/ms/mzml",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": "http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd",
            "version": "1.1.0", "id": "mzML_Generated_Example"
        })
        # CV List
        cv_list = ET.SubElement(mzml, "cvList", count="2")
        ET.SubElement(cv_list, "cv", id="MS", fullName="Proteomics Standards Initiative Mass Spectrometry Ontology", URI="https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo")
        ET.SubElement(cv_list, "cv", id="UO", fullName="Unit Ontology", URI="https://raw.githubusercontent.com/bio-ontology-research-group/unit-ontology/master/unit.obo")
        # File Description
        file_desc = ET.SubElement(mzml, "fileDescription")
        file_content = ET.SubElement(file_desc, "fileContent")
        ET.SubElement(file_content, "cvParam", cvRef="MS", accession="MS:1000580", name="MSn spectrum")
        ET.SubElement(file_content, "cvParam", cvRef="MS", accession="MS:1000579", name="MS1 spectrum")
        ET.SubElement(file_content, "cvParam", cvRef="MS", accession="MS:1000128", name="profile spectrum")
        # Software List
        sw_list = ET.SubElement(mzml, "softwareList", count="1")
        sw = ET.SubElement(sw_list, "software", id="sw_default", version="3.6-refactored")
        ET.SubElement(sw, "cvParam", cvRef="MS", accession="MS:1000799", name="custom unreleased software tool", value="Spectrum Generator Script")
        # Data Processing List
        dp_list = ET.SubElement(mzml, "dataProcessingList", count="1")
        dp = ET.SubElement(dp_list, "dataProcessing", id="dp_default")
        pm = ET.SubElement(dp, "processingMethod", order="1", softwareRef="sw_default")
        ET.SubElement(pm, "cvParam", cvRef="MS", accession="MS:1000544", name="data processing action")
        # Run and Spectrum List
        run = ET.SubElement(mzml, "run", id="simulated_lcms_run", defaultInstrumentConfigurationRef="ic_default")
        ic_list = ET.SubElement(run, "instrumentConfigurationList", count="1")
        ic_elem = ET.SubElement(ic_list, "instrumentConfiguration", id="ic_default")
        ET.SubElement(ic_elem, "cvParam", cvRef="MS", accession="MS:1000031", name="instrument model")
        spectrum_list = ET.SubElement(run, "spectrumList", count=str(total_spectra_count), defaultDataProcessingRef="dp_default")

        # --- Spectrum Template for efficiency ---
        mz_binary, mz_array_len = encode_floats(mz_range), str(len(mz_range))
        spec_template = ET.Element("spectrum", defaultArrayLength=mz_array_len)
        # CV Params for spectrum
        ET.SubElement(spec_template, "cvParam", cvRef="MS", accession="MS:1000579", name="MS1 spectrum")
        ET.SubElement(spec_template, "cvParam", cvRef="MS", accession="MS:1000511", name="ms level", value="1")
        ET.SubElement(spec_template, "cvParam", cvRef="MS", accession="MS:1000130", name="positive scan")
        ET.SubElement(spec_template, "cvParam", cvRef="MS", accession="MS:1000128", name="profile spectrum")
        ET.SubElement(spec_template, "cvParam", cvRef="MS", accession="MS:1000504", name="base peak m/z", value="", unitCvRef="MS", unitAccession="MS:1000040", unitName="m/z")
        ET.SubElement(spec_template, "cvParam", cvRef="MS", accession="MS:1000505", name="base peak intensity", value="", unitCvRef="MS", unitAccession="MS:1000131", unitName="number of detector counts")
        ET.SubElement(spec_template, "cvParam", cvRef="MS", accession="MS:1000285", name="total ion current", value="")
        # Scan List
        scan_list_template = ET.SubElement(spec_template, "scanList", count="1")
        ET.SubElement(scan_list_template, "cvParam", cvRef="MS", accession="MS:1000795", name="no combination")
        scan_template = ET.SubElement(scan_list_template, "scan")
        ET.SubElement(scan_template, "cvParam", cvRef="MS", accession="MS:1000016", name="scan start time", value="", unitCvRef="UO", unitAccession="UO:0000031", unitName="minute")
        # Binary Data Array List (m/z and intensity)
        bdal_template = ET.SubElement(spec_template, "binaryDataArrayList", count="2")
        # m/z array (constant for all scans)
        bda_mz_template = ET.SubElement(bdal_template, "binaryDataArray", encodedLength=str(len(mz_binary)))
        ET.SubElement(bda_mz_template, "cvParam", cvRef="MS", accession="MS:1000514", name="m/z array", unitCvRef="MS", unitAccession="MS:1000040", unitName="m/z")
        ET.SubElement(bda_mz_template, "cvParam", cvRef="MS", accession="MS:1000523", name="64-bit float")
        ET.SubElement(bda_mz_template, "cvParam", cvRef="MS", accession="MS:1000574", name="zlib compression")
        ET.SubElement(bda_mz_template, "binary").text = mz_binary
        # Intensity array (template part)
        bda_int_template = ET.SubElement(bdal_template, "binaryDataArray")
        ET.SubElement(bda_int_template, "cvParam", cvRef="MS", accession="MS:1000515", name="intensity array", unitCvRef="MS", unitAccession="MS:1000131", unitName="number of detector counts")
        ET.SubElement(bda_int_template, "cvParam", cvRef="MS", accession="MS:1000523", name="64-bit float")
        ET.SubElement(bda_int_template, "cvParam", cvRef="MS", accession="MS:1000574", name="zlib compression")
        ET.SubElement(bda_int_template, "binary").text = ""

        # --- Main Loop to Populate Spectra ---
        spectrum_index, native_id_counter = 0, 1
        for protein_spectra in run_data:
            for scan_idx, intensity_array in enumerate(protein_spectra):
                if stop_event and stop_event.is_set():
                    progress_callback('log', "mzML generation cancelled.\n")
                    return None
                spec = copy.deepcopy(spec_template)
                spec.set("index", str(spectrum_index))
                spec.set("id", f"scan={native_id_counter}")

                # Calculate and set scan-specific values
                base_peak_intensity = np.max(intensity_array) if intensity_array.size > 0 else 0.0
                base_peak_mz = mz_range[np.argmax(intensity_array)] if base_peak_intensity > 0 else 0.0
                total_ion_current = np.sum(intensity_array)
                intensity_binary = encode_floats(intensity_array)

                spec.find(".//*[@accession='MS:1000504']").set('value', str(base_peak_mz))
                spec.find(".//*[@accession='MS:1000505']").set('value', str(base_peak_intensity))
                spec.find(".//*[@accession='MS:1000285']").set('value', str(total_ion_current))

                scan = spec.find("scanList/scan")
                scan.set("id", f"scan={native_id_counter}")
                scan.find(".//*[@accession='MS:1000016']").set('value', str(scan_times[scan_idx]))

                bda_int = spec.find("binaryDataArrayList/binaryDataArray[2]")
                bda_int.set("encodedLength", str(len(intensity_binary)))
                bda_int.find("binary").text = intensity_binary

                spectrum_list.append(spec)
                spectrum_index += 1
                native_id_counter += 1

        # --- MS2 Spectra Loop ---
        if msms_spectra:
            # --- MS2 Spectrum Template ---
            ms2_spec_template = ET.Element("spectrum", defaultArrayLength="0")
            ET.SubElement(ms2_spec_template, "cvParam", cvRef="MS", accession="MS:1000580", name="MSn spectrum")
            ET.SubElement(ms2_spec_template, "cvParam", cvRef="MS", accession="MS:1000511", name="ms level", value="2")
            ET.SubElement(ms2_spec_template, "cvParam", cvRef="MS", accession="MS:1000130", name="positive scan")
            ET.SubElement(ms2_spec_template, "cvParam", cvRef="MS", accession="MS:1000127", name="centroid spectrum") # MS2 are centroided
            ET.SubElement(ms2_spec_template, "cvParam", cvRef="MS", accession="MS:1000504", name="base peak m/z", value="", unitCvRef="MS", unitAccession="MS:1000040", unitName="m/z")
            ET.SubElement(ms2_spec_template, "cvParam", cvRef="MS", accession="MS:1000505", name="base peak intensity", value="", unitCvRef="MS", unitAccession="MS:1000131", unitName="number of detector counts")
            ET.SubElement(ms2_spec_template, "cvParam", cvRef="MS", accession="MS:1000285", name="total ion current", value="")

            scan_list_template_ms2 = ET.SubElement(ms2_spec_template, "scanList", count="1")
            ET.SubElement(scan_list_template_ms2, "cvParam", cvRef="MS", accession="MS:1000795", name="no combination")
            scan_template_ms2 = ET.SubElement(scan_list_template_ms2, "scan")
            ET.SubElement(scan_template_ms2, "cvParam", cvRef="MS", accession="MS:1000016", name="scan start time", value="", unitCvRef="UO", unitAccession="UO:0000031", unitName="minute")

            precursor_list_template = ET.SubElement(ms2_spec_template, "precursorList", count="1")
            precursor_template = ET.SubElement(precursor_list_template, "precursor")

            isolation_window_template = ET.SubElement(precursor_template, "isolationWindow")
            ET.SubElement(isolation_window_template, "cvParam", cvRef="MS", accession="MS:1000827", name="isolation window target m/z", value="", unitCvRef="MS", unitAccession="MS:1000040", unitName="m/z")

            selected_ion_list_template = ET.SubElement(precursor_template, "selectedIonList", count="1")
            selected_ion_template = ET.SubElement(selected_ion_list_template, "selectedIon")
            ET.SubElement(selected_ion_template, "cvParam", cvRef="MS", accession="MS:1000744", name="selected ion m/z", value="", unitCvRef="MS", unitAccession="MS:1000040", unitName="m/z")
            ET.SubElement(selected_ion_template, "cvParam", cvRef="MS", accession="MS:1000041", name="charge state", value="")

            activation_template = ET.SubElement(precursor_template, "activation")
            ET.SubElement(activation_template, "cvParam", cvRef="MS", accession="MS:1000133", name="collision-induced dissociation", value="") # Default activation

            bdal_template_ms2 = ET.SubElement(ms2_spec_template, "binaryDataArrayList", count="2")
            bda_mz_template_ms2 = ET.SubElement(bdal_template_ms2, "binaryDataArray", encodedLength="0")
            ET.SubElement(bda_mz_template_ms2, "cvParam", cvRef="MS", accession="MS:1000514", name="m/z array", unitCvRef="MS", unitAccession="MS:1000040", unitName="m/z")
            ET.SubElement(bda_mz_template_ms2, "cvParam", cvRef="MS", accession="MS:1000523", name="64-bit float")
            ET.SubElement(bda_mz_template_ms2, "cvParam", cvRef="MS", accession="MS:1000574", name="zlib compression")
            ET.SubElement(bda_mz_template_ms2, "binary").text = ""

            bda_int_template_ms2 = ET.SubElement(bdal_template_ms2, "binaryDataArray", encodedLength="0")
            ET.SubElement(bda_int_template_ms2, "cvParam", cvRef="MS", accession="MS:1000515", name="intensity array", unitCvRef="MS", unitAccession="MS:1000131", unitName="number of detector counts")
            ET.SubElement(bda_int_template_ms2, "cvParam", cvRef="MS", accession="MS:1000523", name="64-bit float")
            ET.SubElement(bda_int_template_ms2, "cvParam", cvRef="MS", accession="MS:1000574", name="zlib compression")
            ET.SubElement(bda_int_template_ms2, "binary").text = ""

            for msms_scan in msms_spectra:
                for event in msms_scan.fragmentation_events:
                    if stop_event and stop_event.is_set():
                        progress_callback('log', "mzML generation cancelled.\n")
                        return None
                    spec = copy.deepcopy(ms2_spec_template)
                    spec.set("index", str(spectrum_index))
                    spec.set("id", f"scan={native_id_counter}")
                    spec.set("defaultArrayLength", str(len(event.fragments.mz)))

                    # Precursor info
                    precursor = spec.find("precursorList/precursor")
                    precursor.find("isolationWindow/cvParam[@accession='MS:1000827']").set("value", str(event.precursor_mz))
                    selected_ion = precursor.find("selectedIonList/selectedIon")
                    selected_ion.find("cvParam[@accession='MS:1000744']").set("value", str(event.precursor_mz))
                    selected_ion.find("cvParam[@accession='MS:1000041']").set("value", str(event.precursor_charge))

                    # Scan info
                    scan = spec.find("scanList/scan")
                    scan.set("id", f"scan={native_id_counter}")
                    scan.find(".//*[@accession='MS:1000016']").set('value', str(event.rt / 60.0)) # convert to minutes

                    # Binary data
                    mz_binary = encode_floats(event.fragments.mz)
                    intensity_binary = encode_floats(event.fragments.intensity)

                    bda_mz = spec.find("binaryDataArrayList/binaryDataArray[1]")
                    bda_mz.set("encodedLength", str(len(mz_binary)))
                    bda_mz.find("binary").text = mz_binary

                    bda_int = spec.find("binaryDataArrayList/binaryDataArray[2]")
                    bda_int.set("encodedLength", str(len(intensity_binary)))
                    bda_int.find("binary").text = intensity_binary

                    # Spectrum stats
                    base_peak_intensity = np.max(event.fragments.intensity) if event.fragments.intensity.size > 0 else 0.0
                    base_peak_mz = event.fragments.mz[np.argmax(event.fragments.intensity)] if base_peak_intensity > 0 else 0.0
                    total_ion_current = np.sum(event.fragments.intensity)

                    spec.find(".//*[@accession='MS:1000504']").set('value', str(base_peak_mz))
                    spec.find(".//*[@accession='MS:1000505']").set('value', str(base_peak_intensity))
                    spec.find(".//*[@accession='MS:1000285']").set('value', str(total_ion_current))

                    spectrum_list.append(spec)
                    spectrum_index += 1
                    native_id_counter += 1

        # --- Finalization ---
        # The SHA-1 checksum must be calculated on the XML *before* the checksum tag is added.
        mzml_bytes_no_checksum = ET.tostring(mzml, encoding='utf-8', method='xml')
        sha1_checksum = hashlib.sha1(mzml_bytes_no_checksum).hexdigest()

        # Now add the checksum tag to the main element
        file_checksum_tag = ET.SubElement(mzml, "fileChecksum")
        ET.SubElement(file_checksum_tag, "cvParam", cvRef="MS", accession="MS:1000569", name="SHA-1", value=sha1_checksum)

        # Final XML bytes with declaration
        final_xml_bytes = b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(mzml, encoding='utf-8', method='xml')

        progress_callback('log', "mzML structure generated.\n")
        return final_xml_bytes

    except Exception as e:
        progress_callback('error', f"Error during mzML content creation: {e}")
        return None
