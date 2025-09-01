# It is a good practice to centralize constants, especially strings,
# to improve maintainability and simplify future updates or translations.

# --- Tooltips ---
PROTEIN_LIST_FILE_TOOLTIP = "Path to a tab-delimited file with protein data.\nMust contain 'Protein' (Average Mass) and 'Intensity' headers."
BROWSE_PROTEIN_LIST_TOOLTIP = "Browse for a protein list file."
SAVE_TEMPLATE_TOOLTIP = "Save the manually entered masses and scalars below as a valid\ntab-delimited template file for future use."
PROTEIN_MASSES_TOOLTIP = "A comma-separated list of protein AVERAGE masses to simulate."
INTENSITY_SCALARS_TOOLTIP = "A comma-separated list of relative intensity multipliers.\nMust match the number of protein masses."
MASS_INHOMOGENEITY_TOOLTIP = "Standard deviation of protein mass distribution to simulate conformational broadening.\nSet to 0 to disable. A small value (e.g., 1-5 Da) is recommended."
PREVIEW_SPECTRUM_TOOLTIP = "Generate and display a plot of a single spectrum using the current settings.\nUses the first protein mass if multiple are entered."
GENERATE_PLOT_TOOLTIP = "Generate the spectrum and view it in the 'Plot Viewer' tab without saving a file."
GENERATE_MZML_TOOLTIP = "Generate and save .mzML file(s) with the specified parameters."

# --- Labels ---
PROTEIN_LIST_FILE_LABEL = "Protein List File (.txt):"
PROTEIN_MASSES_LABEL = "Protein Avg. Masses (Da, comma-sep):"
INTENSITY_SCALARS_LABEL = "Intensity Scalars (comma-sep):"
MASS_INHOMOGENEITY_LABEL = "Mass Inhomogeneity (Std. Dev., Da):"
MANUAL_INPUT_LABEL = "OR Enter Manually Below"

# --- Button Text ---
BROWSE_BUTTON_TEXT = "Browse..."
SAVE_TEMPLATE_BUTTON_TEXT = "Save as Template..."
PREVIEW_SPECTRUM_BUTTON_TEXT = "Preview Spectrum"
GENERATE_PLOT_BUTTON_TEXT = "Generate & Plot"
GENERATE_MZML_BUTTON_TEXT = "Generate mzML File(s)"

# --- File Dialogs ---
SAVE_TEMPLATE_TITLE = "Save Protein List Template"
PROTEIN_LIST_TEMPLATE_FILENAME = "protein_list_template.txt"
FILE_TYPES = [("Text Files", "*.txt"), ("All Files", "*.*")]

# --- Messages ---
PLOT_WARNING_TITLE = "Warning"
PLOT_WARNING_MESSAGE = "Plotting is only available for manually entered proteins, not for file-based batch processing."
SAVE_ERROR_TITLE = "Save Error"
SAVE_ERROR_MESSAGE = "Could not save template file.\nError: {}"
INVALID_INPUT_ERROR_TITLE = "Error"
UNEXPECTED_ERROR_MESSAGE = "An unexpected error occurred: {}"
INVALID_PREVIEW_INPUT_ERROR = "Invalid input for preview: {}"
INVALID_INPUT_ERROR = "Invalid input: {}"
