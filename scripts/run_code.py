"""
Script to automatically trace DNA and nucleosome structure parameters in AFM images
"""

from tqdm import tqdm

import import_custom
import export_custom
import molecule_categorization as cat
import analysis_bare_DNA
import analysis_nucleosome
import analysis_nucleosome_eb
import tip_shape_estimation
import plot_functions as plotting

# Analysis decision making
# Note: Endbound nucleosomes will only be analyzed if both manual filtering and analyze_nucleosomes_eb are True
manual_filtering = True                     # Manually inspect all uncategorized molecules
kwargs = {'analyze_bare_DNA': True,         # Analyze bare DNA - True or False
          'analyze_nucleosomes': True,      # Analyze nucleosomes - True or False
          'analyze_nucleosomes_eb': False}  # Analyze endbound nucleosomes - True or False
deconvolve = {'decon': True,                # Applies a deconv. for tracing. Only works when bare DNA analysis is True
              'tip_shape': None}
output_file = 'cenpa.xlsx'  # Name of the Excel Workbook to store the data in

# Data export decision making
save_close_ups = True                       # Save close-up shots of each molecule in a png - True or False
export_data = True                          # Store the analyzed data in an Excel sheet - True or False
export_select_manually = True               # Select the data to be exported as valid manually
add_file_name_pars = True                   # Adds image specs to the excel sheet - proper naming necessary (GUIDE)

anal_pars = {'back_thresh': 0.08,           # Height value for the first background threshold
             'mol_min_area': 300,           # Minimum amount of pixels for a molecule to be considered for analysis
             'nuc_min_height': 1.00,        # Height threshold for nucleosome pixels
             'nuc_min_area': 12,            # Minimum amount of pixels above nuc_min_height to be a nucleosome
             'nuc_max_area': 300,
             'dna_length_bp': 486}

# Import the desired .ascii file (manual selection)
img_original, img_meta_data = import_custom.import_ascii()

# Find the molecules in the image, each molecule is stored as an entry in the molecules list
img_filtered, molecules = cat.find_molecules(img_original, img_meta_data, **anal_pars)

# Create an AFM molecule instance for each individual molecule
afm_molecules = [cat.AFMMolecule(mol, img_meta_data, anal_pars) for mol in molecules]

# Split the AFM molecules into lists depending on their type
mol_bare_DNA = [mol for mol in afm_molecules if mol.mol_pars['type'] == 'Bare DNA']
mol_nucleosome = [mol for mol in afm_molecules if mol.mol_pars['type'] == 'Nucleosome']
mol_potential = [mol for mol in afm_molecules if mol.mol_pars['type'] == 'Potential']
mol_trash = [mol for mol in afm_molecules if mol.mol_pars['type'] == 'Trash']
print('\nMolecules found:')
print('{} bare DNA strands'.format(len(mol_bare_DNA)))
print('{} nucleosomes'.format(len(mol_nucleosome)))
print('{} Potential molecules'.format(len(mol_potential)))
print('{} Trash molecules'.format(len(mol_trash)))

if manual_filtering is True:
    # Give possibility to manually separate trashed molecules
    afm_molecules = cat.manual_trash_analysis(afm_molecules)

# Again split the AFM molecules into lists depending on their type after manually helping categorization
mol_bare_DNA = [mol for mol in afm_molecules if mol.mol_pars['type'] == 'Bare DNA']
mol_nucleosome = [mol for mol in afm_molecules if mol.mol_pars['type'] == 'Nucleosome']
mol_nucleosome_eb = [mol for mol in afm_molecules if mol.mol_pars['type'] == 'Nucleosome endbound']
mol_potential = [mol for mol in afm_molecules if mol.mol_pars['type'] == 'Potential']
mol_trash = [mol for mol in afm_molecules if mol.mol_pars['type'] == 'Trash']
if manual_filtering is True:
    print('\nMolecules found after manual separation:')
    print('{} bare DNA strands'.format(len(mol_bare_DNA)))
    print('{} nucleosomes'.format(len(mol_nucleosome)))
    print('{} nucleosomes endbound'.format(len(mol_nucleosome_eb)))
    print('{} undefined molecules'.format(len(mol_trash) + len(mol_potential)))

results_final = {}
# Analysis of Bare DNA
if kwargs['analyze_bare_DNA'] is True:
    print('\nAnalyzing bare DNA:')
    analyzed_bare_DNA = [analysis_bare_DNA.BareDNA(mol, deconvolve) for mol in tqdm(mol_bare_DNA)]
    deconvolve['tip_shape'] = tip_shape_estimation.estimate_tip_from_DNA(analyzed_bare_DNA)
    if deconvolve['decon'] is True:
        print('\nAnalyzing deconvolved bare DNA:')
        analyzed_bare_DNA = [analysis_bare_DNA.BareDNA(mol, deconvolve) for mol in tqdm(mol_bare_DNA)]
    results_final.update({'analyzed_bare_DNA': analyzed_bare_DNA})

# Analysis of normal mono-nucleosomes
if kwargs['analyze_nucleosomes'] is True:
    print('\nAnalyzing mono-nucleosomes:')
    analyzed_nucleosomes = [analysis_nucleosome.Nucleosome(mol, deconvolve) for mol in tqdm(mol_nucleosome)]
    results_final.update({'analyzed_nucleosomes': analyzed_nucleosomes})

# Analysis of endbound mono-nucleosomes
if kwargs['analyze_nucleosomes_eb'] is True:
    print('\nAnalyzing endbound mono-nucleosomes:')
    analyzed_nucleosomes_eb = [analysis_nucleosome_eb.NucleosomeEB(mol, deconvolve) for mol in tqdm(mol_nucleosome_eb)]
    results_final.update({'analyzed_nucleosomes_eb': analyzed_nucleosomes_eb})

# Export the data
if export_data is True:
    if export_select_manually is True:
        results_final = export_custom.manual_export_selection(results_final)

    export_custom.export_to_excel(output_file,
                                  results_final,
                                  img_meta_data['file_path'],
                                  add_file_name_pars=add_file_name_pars,
                                  **kwargs)

if save_close_ups is True:
    results_final.update({'mol_trash': mol_trash})
    plotting.plot_save_close_ups(results_final,
                                 img_meta_data['file_name'],
                                 **kwargs,
                                 plot_trash=True)

# Plot the results
plotting.plot_overview_image(img_filtered,
                             img_meta_data['file_name'],
                             results_final,
                             **kwargs)
