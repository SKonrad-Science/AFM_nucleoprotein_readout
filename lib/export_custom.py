import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plot_functions as plotting


def manual_export_selection(results_final):
    # Improve this code - quick and dirty
    my_colormap = plotting.create_custom_colormap()

    if 'analyzed_bare_DNA' in results_final:
        for mol in results_final['analyzed_bare_DNA']:
            if mol.results['failed'] is False:
                plt.figure()
                mol_filtered_rgb = plotting.convert_to_rgb(mol.mol_filtered, my_colormap)

                if mol.results['length_fwd'] is not False:
                    wiggins_pixels = copy.deepcopy(mol.results['wiggins_pixels_fwd'])
                elif mol.results['length_bwd'] is not False:
                    wiggins_pixels = copy.deepcopy(mol.results['wiggins_pixels_bwd'])

                plt.imshow(mol_filtered_rgb, interpolation='None')
                plotting.plot_wiggins_pixels_close_up(wiggins_pixels)
                plt.text(2, 6, 'Length: ' + repr(np.round(mol.results['length_avg'], decimals=1))
                         + ' nm', color='white', fontsize=12)
                mol.results['failed'] = plt.waitforbuttonpress()
                if mol.results['failed'] is True:
                    mol.results.update({'failed_reason': 'Discarded manually'})
                plt.close()

    if 'analyzed_nucleosomes' in results_final:
        for mol in results_final['analyzed_nucleosomes']:
            if mol.results['failed'] is False:
                plt.figure()
                mol_filtered_rgb = plotting.convert_to_rgb(mol.mol_filtered, my_colormap)

                plt.imshow(mol_filtered_rgb, interpolation='None')
                plotting.plot_wiggins_pixels_close_up(mol.results['pixels_arm1'])
                plotting.plot_wiggins_pixels_close_up(mol.results['pixels_arm2'])
                plotting.plot_ellipse_close_up(mol.results['ellipsoid_coeff'])
                plt.scatter(mol.results['ellipsoid_coeff'][1], mol.results['ellipsoid_coeff'][0], color='#2A3132')
                plt.text(2, 3, 'Arm sum: ' + repr(np.round(mol.results['length_sum'], decimals=1))
                         + ' nm', color='white', fontsize=10)
                plt.text(2, 6, 'Angle: ' + repr(np.round(mol.results['angle_arms'], decimals=1))
                         + ' Degree', color='white', fontsize=10)
                plt.text(2, 9, 'Volume: ' + repr(np.round(mol.results['nucleosome_volume'], decimals=1))
                         + ' nm^3', color='white', fontsize=10)
                mol.results['failed'] = plt.waitforbuttonpress()
                if mol.results['failed'] is True:
                    mol.results.update({'failed_reason': 'Discarded manually'})
                plt.close()

    if 'analyzed_nucleosomes_eb' in results_final:
        for mol in results_final['analyzed_nucleosomes_eb']:
            mol.results.update({'failed_reason': 'Unknown'})
            if mol.results['failed'] is False:
                plt.figure()
                mol_filtered_rgb = plotting.convert_to_rgb(mol.mol_filtered, my_colormap)

                plt.imshow(mol_filtered_rgb, interpolation='None')
                plotting.plot_wiggins_pixels_close_up(mol.results['pixels_arm1'])
                plotting.plot_ellipse_close_up(mol.results['ellipsoid_coeff'])
                plt.scatter(mol.results['ellipsoid_coeff'][1], mol.results['ellipsoid_coeff'][0], color='#2A3132')

                plt.text(2, 3, 'Arm: ' + repr(np.round(mol.results['length_arm1'], decimals=1))
                         + ' nm', color='white', fontsize=10)
                plt.text(2, 6, 'Volume: ' + repr(np.round(mol.results['nucleosome_volume'], decimals=1))
                         + ' nm^3', color='white', fontsize=10)
                mol.results['failed'] = plt.waitforbuttonpress()
                if mol.results['failed'] is True:
                    mol.results.update({'failed_reason': 'Discarded manually'})
                plt.close()

    if 'analyzed_ints' in results_final:
        for mol in results_final['analyzed_ints']:
            mol.results.update({'failed_reason': 'Unknown'})
            if mol.results['failed'] is False:
                plt.figure()
                mol_filtered_rgb = plotting.convert_to_rgb(mol.mol_filtered, my_colormap)

                plt.imshow(mol_filtered_rgb, interpolation='None')

                if mol.results['ellipsoid_height'] != 0:
                    plotting.plot_ellipse_close_up(mol.results['ellipsoid_coeff'], color='black')
                    plt.scatter(mol.results['ellipsoid_coeff'][1], mol.results['ellipsoid_coeff'][0], color='#2A3132')

                mol.results['failed'] = plt.waitforbuttonpress()
                if mol.results['failed'] is True:
                    mol.results.update({'failed_reason': 'Discarded manually'})
                plt.close()

    return results_final


def results_to_df(analyzed_molecules, pars, file_path, add_file_name_pars=False):

    results = []
    for mol in analyzed_molecules:
        if mol.results['failed'] is False:
            results.append({key: np.round(mol.results[key], decimals=3)
                            if isinstance(mol.results[key], str) is False
                            else mol.results[key] for key in pars
                           })
    results = pd.DataFrame(results)

    if add_file_name_pars is True:
        file_name = file_path.split('/')[-1]
        results.insert(loc=0, column='Date', value=[file_name.split('_')[0]] * len(results))
        results.insert(loc=1, column='File', value=[file_name.split('_')[-1].split('.')[1]] * len(results))
        results.insert(loc=2, column='Type', value=[file_name.split('_')[1]] * len(results))
        results.insert(loc=3, column='Salt', value=[file_name.split('_')[2]] * len(results))
        results.insert(loc=4, column='Tip', value=[file_name.split('_')[-1].split('.')[0]] * len(results))
        results.insert(loc=5, column='Surface Dep.', value=[file_path.split('/')[-2].split('_')[0]] * len(results))
        results.insert(loc=6, column='Meas. Day', value=[file_path.split('/')[-2].split('_')[1]] * len(results))

    else:
        file_name = file_path.split('/')[-1]
        results.insert(loc=0, column='File', value=[file_name.split('_')[-1].split('.')[1]] * len(results))

    return results


def export_to_excel(output_file,
                    results_final,
                    file_path,
                    analyze_bare_DNA=False,
                    analyze_nucleosomes=False,
                    analyze_nucleosomes_eb=False,
                    analyze_ints=False,
                    add_file_name_pars=False):
    """ Store the desired data in an Excel sheet """

    # Define the parameters that are being stored for each molecule type
    dna_pars = ['length_avg', 'length_etoe', 'rog', 'height_avg', 'height_std', 'slope_avg', 'slope_std',
                'position_row', 'position_col', 'rightness', 'extension_right', 'extension_bot',
                'z_angles_1', 'z_angles_2', 'z_angles_3', 'z_angles_4']
    nuc_pars = ['length_arm1_40', 'length_arm2_40', 'length_arm1_50', 'length_arm2_50',
                'length_arm1_60', 'length_arm2_60', 'length_arm1_70', 'length_arm2_70',
                'length_etoe', 'angle_arms', 'nucleosome_volume',
                'nucleosome_volume_core', 'radius_of_gyration', 'height_avg', 'height_std',
                'slope_avg', 'slope_std', 'ellipsoid_width_a', 'ellipsoid_width_b',
                'ellipsoid_height', 'nuc_max_height', 'nuc_max_height_avg', 'ellipsoid_angle', 'position_row',
                'position_col', 'rightness_arm1', 'rightness_arm2', 'extension_right', 'extension_bot',
                'arm1_end_r', 'arm1_end_c', 'arm2_end_r', 'arm2_end_c', 'ell_center_r', 'ell_center_c',
                'z_nuc_cutout_str', 'bending_avg_arm1', 'bending_avg_arm2']
    nuc_eb_pars = ['length_arm1', 'nucleosome_volume', 'nucleosome_volume_core', 'height_avg', 'height_std',
                   'slope_avg', 'slope_std', 'ellipsoid_width_a', 'ellipsoid_width_b', 'ellipsoid_height',
                   'ellipsoid_angle', 'position_row', 'position_col',
                   'rightness_arm1', 'extension_right', 'extension_bot']
    int_pars = ['radius_of_gyration', 'radius_of_gyration_core', 'com_r', 'com_c', 'com_core_r', 'com_core_c',
                'ellipsoid_width_a', 'ellipsoid_width_b', 'ellipsoid_height', 'ellipsoid_angle',
                'total_volume', 'int_volume', 'int_max_height']

    # Test whether the Excel workbook already exists
    try:
        existing_data = pd.ExcelFile(output_file)
        existing = True
    except:
        existing = False

    if analyze_bare_DNA is True:
        df_bare_dna = results_to_df(results_final['analyzed_bare_DNA'], dna_pars, file_path,
                                    add_file_name_pars=add_file_name_pars)
        if existing is True:
            if 'Bare DNA' in existing_data.sheet_names:
                df_bare_dna_before = pd.read_excel(existing_data, 'Bare DNA', index_col=0)
                df_bare_dna = pd.concat([df_bare_dna_before, df_bare_dna], axis=0)
            else:
                print('No previous Bare DNA data was found in the output_file. New sheet was created.')
    elif existing is True:
        if analyze_bare_DNA is False and 'Bare DNA' in existing_data.sheet_names:
            df_bare_dna = pd.read_excel(existing_data, 'Bare DNA', index_col=0)
            analyze_bare_DNA = True

    if analyze_nucleosomes is True:
        df_nucleosomes = results_to_df(results_final['analyzed_nucleosomes'], nuc_pars, file_path,
                                       add_file_name_pars=add_file_name_pars)

        if existing is True:
            if 'Nucleosomes' in existing_data.sheet_names:
                df_nucleosomes_before = pd.read_excel(existing_data, 'Nucleosomes', index_col=0)
                df_nucleosomes = pd.concat([df_nucleosomes_before, df_nucleosomes], axis=0)
            else:
                print('No previous nucleosome data was found in the output_file. New sheet was created.')

    elif existing is True:
        if analyze_nucleosomes is False and 'Nucleosomes' in existing_data.sheet_names:
            df_nucleosomes = pd.read_excel(existing_data, 'Nucleosomes', index_col=0)
            analyze_nucleosomes = True

    if analyze_nucleosomes_eb is True:
        df_nucleosomes_eb = results_to_df(results_final['analyzed_nucleosomes_eb'], nuc_eb_pars, file_path,
                                          add_file_name_pars=add_file_name_pars)
        if existing is True:
            if 'Nucleosomes Endbound' in existing_data.sheet_names:
                df_nucleosomes_eb_before = pd.read_excel(existing_data, 'Nucleosomes Endbound', index_col=0)
                df_nucleosomes_eb = pd.concat([df_nucleosomes_eb_before, df_nucleosomes_eb], axis=0)
            else:
                print('No previous endbound nucleosomes data was found in the output_file. New sheet was created.')
    elif existing is True:
        if analyze_nucleosomes_eb is False and 'Nucleosomes Endbound' in existing_data.sheet_names:
            df_nucleosomes_eb = pd.read_excel(existing_data, 'Nucleosomes Endbound', index_col=0)
            analyze_nucleosomes_eb = True

    if analyze_ints is True:
        df_ints = results_to_df(results_final['analyzed_ints'], int_pars, file_path,
                                add_file_name_pars=add_file_name_pars)

        if existing is True:
            if 'Integrase' in existing_data.sheet_names:
                df_ints_before = pd.read_excel(existing_data, 'Integrase', index_col=0)
                df_ints = pd.concat([df_ints_before, df_ints], axis=0, sort=True)
            else:
                print('No previous nucleosome data was found in the output_file. New sheet was created.')

    elif existing is True:
        if analyze_ints is False and 'Integrase' in existing_data.sheet_names:
            df_ints = pd.read_excel(existing_data, 'Integrase', index_col=0)
            analyze_ints = True

    # Place the writer below the DataFrame creation to make it saver
    # In case you declare the writer and something fails during DataFrame creation it would leave an empty Excel sheet
    writer = pd.ExcelWriter(output_file)
    if analyze_bare_DNA is True:
        df_bare_dna.to_excel(writer, 'Bare DNA')

    if analyze_nucleosomes is True:
        df_nucleosomes.to_excel(writer, 'Nucleosomes')

    if analyze_nucleosomes_eb is True:
        df_nucleosomes_eb.to_excel(writer, 'Nucleosomes Endbound')

    if analyze_ints is True:
        df_ints.to_excel(writer, 'Integrase')
    writer.save()

    return


def export_to_excel_DNA_angles(output_file,
                    results_final,
                    angles,
                    file_path,
                    analyze_bare_DNA=False,
                    add_file_name_pars=False):

    """ Store the desired data in an Excel sheet """

    # Define the parameters that are being stored for each molecule type
    dna_pars = ['length_avg', 'length_etoe', 'rog', 'height_avg', 'height_std', 'slope_avg', 'slope_std',
                'position_row', 'position_col', 'rightness', 'extension_right', 'extension_bot']

    # Test whether the Excel workbook already exists
    try:
        existing_data = pd.ExcelFile(output_file)
        existing = True
    except:
        existing = False

    df_wiggins_pixels = angles['pixels']
    df_angles_1 = angles['angles_1']
    df_angles_2 = angles['angles_2']
    df_angles_3 = angles['angles_3']
    df_angles_4 = angles['angles_4']

    file_name = file_path.split('/')[-1]
    df_wiggins_pixels.insert(loc=0, column='File',
                             value=[file_name.split('_')[-1].split('.')[1]] * len(df_wiggins_pixels))
    df_angles_1.insert(loc=0, column='File', value=[file_name.split('_')[-1].split('.')[1]] * len(df_angles_1))
    df_angles_2.insert(loc=0, column='File', value=[file_name.split('_')[-1].split('.')[1]] * len(df_angles_2))
    df_angles_3.insert(loc=0, column='File', value=[file_name.split('_')[-1].split('.')[1]] * len(df_angles_3))
    df_angles_4.insert(loc=0, column='File', value=[file_name.split('_')[-1].split('.')[1]] * len(df_angles_4))


    if analyze_bare_DNA is True:
        df_bare_dna = results_to_df(results_final['analyzed_bare_DNA'], dna_pars, file_path,
                                    add_file_name_pars=add_file_name_pars)
        if existing is True:
            if 'Bare DNA' in existing_data.sheet_names:
                df_bare_dna_before = pd.read_excel(existing_data, 'Bare DNA')
                df_bare_dna = pd.concat([df_bare_dna_before, df_bare_dna], axis=0)

                df_wiggins_pixels_before = pd.read_excel(existing_data, 'Pixels')
                df_wiggins_pixels = pd.concat([df_wiggins_pixels_before, df_wiggins_pixels], axis=0)

                df_angles_1_before = pd.read_excel(existing_data, 'Angles 1')
                df_angles_1 = pd.concat([df_angles_1_before, df_angles_1], axis=0)

                df_angles_2_before = pd.read_excel(existing_data, 'Angles 2')
                df_angles_2 = pd.concat([df_angles_2_before, df_angles_2], axis=0)

                df_angles_3_before = pd.read_excel(existing_data, 'Angles 3')
                df_angles_3 = pd.concat([df_angles_3_before, df_angles_3], axis=0)

                df_angles_4_before = pd.read_excel(existing_data, 'Angles 4')
                df_angles_4 = pd.concat([df_angles_4_before, df_angles_4], axis=0)
            else:
                print('No previous Bare DNA data was found in the output_file. New sheet was created.')

    elif existing is True:
        if analyze_bare_DNA is False and 'Bare DNA' in existing_data.sheet_names:
            df_bare_dna = pd.read_excel(existing_data, 'Bare DNA')
            analyze_bare_DNA = True

    # Place the writer below the DataFrame creation to make it saver
    # In case you declare the writer and something fails during DataFrame creation it would leave an empty Excel sheet
    writer = pd.ExcelWriter(output_file)
    if analyze_bare_DNA is True:
        df_bare_dna.to_excel(writer, 'Bare DNA')

    df_wiggins_pixels.to_excel(writer, 'Pixels')
    df_angles_1.to_excel(writer, 'Angles 1')
    df_angles_2.to_excel(writer, 'Angles 2')
    df_angles_3.to_excel(writer, 'Angles 3')
    df_angles_4.to_excel(writer, 'Angles 4')

    writer.save()

    return
