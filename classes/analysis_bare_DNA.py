"""
Class to analyze categorized bare DNA
"""

import copy

import numpy as np
import pandas as pd

from molecule_categorization import AFMMolecule
import analysis_functions as analysis


class BareDNA(AFMMolecule):

    def __init__(self, mol, dna_bp, pixel_size, background_1, background_2, min_area, nuc_min_height, mol_pars,
                 apply_length_threshold=False):
        super().__init__(mol, dna_bp, pixel_size, background_1, background_2, min_area, nuc_min_height)
        # Copy the variable, otherwise they are also changed in the AFMMolecule instances
        self.mol_pars = copy.deepcopy(mol_pars)
        self.improve_skel()

        # Apply Wiggin's algorithm and store the calculated values in the results dict
        self.results = {}
        self.results.update({'position_row': self.mol_bbox[0],
                             'position_col': self.mol_bbox[1]})
        self.calculate_length()
        if apply_length_threshold is True:
            self.length_filter()
        radius, center_of_mass = analysis.radius_of_gyration(self.mol_filtered)
        self.results.update({'rog': radius*pixel_size,
                             'com': center_of_mass})
        self.further_analysis()
        self.angle_analysis()

    def improve_skel(self):
        mol_filtered = copy.deepcopy(self.mol_filtered)
        mol_pars = copy.deepcopy(self.mol_pars)

        # Sort the skeleton by using the sort_skel function
        mol_pars['skel_sorted'] = analysis.sort_skel(mol_pars, start=mol_pars['skel_eps_pixels'][0, :])

        # Skeletonize the first end
        mol_pars['mol_skel'] = analysis.skeletonize_end(mol_filtered, mol_pars, mol_pars['skel_sorted'][0:4])

        # Skeletonize the second end
        mol_pars['mol_skel'] = analysis.skeletonize_end(mol_filtered, mol_pars, mol_pars['skel_sorted'][:-5:-1])

        # Recalculate skeleton parameters
        eps_pixels, eps_number, bps_pixels, bps_number, pixels_number = analysis.skel_pars(mol_pars['mol_skel'])
        self.mol_pars.update({'skel_eps_pixels': eps_pixels,
                              'skel_eps_number': eps_number,
                              'skel_bps_pixels': bps_pixels,
                              'skel_bps_number': bps_number,
                              'skel_pixels_number': pixels_number})

        # Sort the skeleton with the new endpoints
        mol_pars['skel_sorted'] = analysis.sort_skel(mol_pars, start=eps_pixels[0, :])

        # Update the remaining parameters
        self.mol_pars.update({'skel_sorted': mol_pars['skel_sorted'],
                              'mol_skel': mol_pars['mol_skel']})
        return self

    def calculate_length(self):
        """ Use Wiggin's algorithm to calculate the DNA lengths """
        mol_filtered = copy.deepcopy(self.mol_filtered)
        mol_pars = copy.deepcopy(self.mol_pars)
        seg_length = 5/self.pixel_size

        # Calculate the length starting at the top-leftmost endpoint
        wiggins_pixels_fwd, failed_fwd = analysis.wiggins(mol_filtered,
                                                          seg_length=seg_length,
                                                          start=mol_pars['skel_sorted'][:4],
                                                          end=mol_pars['skel_sorted'][-1],
                                                          mol_type='Bare DNA')
        if failed_fwd is False:
            length_fwd = np.sum([np.linalg.norm(wiggins_pixels_fwd[i] - wiggins_pixels_fwd[i+1])
                                 for i in range(0, len(wiggins_pixels_fwd)-1)]) * self.pixel_size
        else:
            length_fwd = False

        # Calculate the length starting at the bottom-rightmost endpoint
        wiggins_pixels_bwd, failed_bwd = analysis.wiggins(mol_filtered,
                                                          seg_length=seg_length,
                                                          start=mol_pars['skel_sorted'][:-5:-1],
                                                          end=mol_pars['skel_sorted'][0],
                                                          mol_type='Bare DNA')
        if failed_bwd is False:
            length_bwd = np.sum([np.linalg.norm(wiggins_pixels_bwd[i] - wiggins_pixels_bwd[i+1])
                                 for i in range(0, len(wiggins_pixels_bwd)-1)]) * self.pixel_size
        else:
            length_bwd = False

        if failed_fwd is False and failed_bwd is False:
            length_avg = (length_fwd + length_bwd) / 2
            length_etoe =  np.linalg.norm(wiggins_pixels_fwd[0] - wiggins_pixels_fwd[-1]) * self.pixel_size
        elif failed_fwd is False and failed_bwd is True:
            length_avg = length_fwd
            length_etoe = np.linalg.norm(wiggins_pixels_fwd[0] - wiggins_pixels_fwd[-1]) * self.pixel_size
        elif failed_fwd is True and failed_bwd is False:
            length_avg = length_bwd
            length_etoe = np.linalg.norm(wiggins_pixels_bwd[0] - wiggins_pixels_bwd[-1]) * self.pixel_size
        else:
            length_avg = False
            length_etoe = False
            self.results.update({'failed_reason': 'Wiggins failed'})

        self.results.update({'wiggins_pixels_fwd': wiggins_pixels_fwd,
                             'wiggins_pixels_bwd': wiggins_pixels_bwd,
                             'length_fwd': length_fwd,
                             'length_bwd': length_bwd,
                             'length_avg': length_avg,
                             'length_etoe': length_etoe,
                             'failed': True if failed_bwd is True and failed_fwd is True else False})

        if failed_fwd is False and failed_bwd is False:
            if abs(length_fwd - length_bwd) >= 0.05 * self.results['length_avg']:
                self.results.update({'length_avg': False,
                                     'failed': True,
                                     'failed_reason': 'Back-Forth difference'})

        return self

    def length_filter(self):

        if self.results['length_fwd'] <= 0.80*self.DNA_bp*0.33 or self.results['length_bwd'] <= 0.80*self.DNA_bp*0.33:
            self.results.update({'length_avg': False,
                                 'failed_reason': 'Too short',
                                 'failed': True})

        elif self.results['length_fwd'] >= 1.25*self.DNA_bp*0.33 or self.results['length_bwd'] >= 1.25*self.DNA_bp*0.33:
            self.results.update({'length_avg': False,
                                 'failed_reason': 'Too long',
                                 'failed': True})

        return self

    def further_analysis(self):
        """ Check the height values along the Wiggins pixels and the slope between individual pixels """

        if self.results['failed'] is False:
            if self.results['length_fwd'] is not False:
                height_pars = analysis.wiggins_pixel_height_analysis(self.results['wiggins_pixels_fwd'],
                                                                     self.mol_filtered, self.pixel_size)
                orientation_pars = analysis.dna_orientation(self.results['wiggins_pixels_fwd'],
                                                            self.mol_bbox)

            elif self.results['length_bwd'] is not False:
                height_pars = analysis.wiggins_pixel_height_analysis(self.results['wiggins_pixels_bwd'],
                                                                     self.mol_filtered, self.pixel_size)
                orientation_pars = analysis.dna_orientation(self.results['wiggins_pixels_fwd'],
                                                            self.mol_bbox)

            self.results.update(height_pars)
            self.results.update(orientation_pars)

        return self

    def angle_analysis(self):
        #### Improve this and move it to the analysis_functions in the lib

        wiggins_pixels_all = []
        try:
            if self.results['failed'] is False:
                if self.results['wiggins_pixels_fwd'] is not False:
                    wiggins_pixels = np.asarray(self.results['wiggins_pixels_fwd'])
                else:
                    wiggins_pixels = np.asarray(self.results['wiggins_pixels_bwd'])
                wiggins_pixels_all.append(np.hstack((np.array([[0] * len(wiggins_pixels)]).T, wiggins_pixels)))

                wp_arr = np.vstack((wiggins_pixels_all[:]))
                df_wiggins_pixels = pd.DataFrame({'Molecule': wp_arr[:, 0].astype(int),
                                                  'r_coord': wp_arr[:, 1],
                                                  'c_coord': wp_arr[:, 2]})

                angles_1_segment = []
                for mol_pixels in wiggins_pixels_all:
                    angles = []
                    for i in range(0, len(mol_pixels) - 4):
                        angles.append(np.array([int(mol_pixels[i, 0]),
                                                analysis.angle_between(mol_pixels[i + 2, 1:3] - mol_pixels[i + 1, 1:3],
                                                                       mol_pixels[i + 3, 1:3] - mol_pixels[i + 2, 1:3])]))
                    angles_1_segment.append(np.asarray(angles))
                angles_1_arr = np.vstack((angles_1_segment[:]))
                df_angles_1 = pd.DataFrame({'Molecule': angles_1_arr[:, 0].astype(int),
                                            'Seg_Angle': angles_1_arr[:, 1]})

                angles_2_segment = []
                for mol_pixels in wiggins_pixels_all:
                    angles = []
                    for i in range(0, len(mol_pixels) - 6):
                        angles.append(np.array([int(mol_pixels[i, 0]),
                                                analysis.angle_between(mol_pixels[i + 3, 1:3] - mol_pixels[i + 1, 1:3],
                                                                       mol_pixels[i + 5, 1:3] - mol_pixels[i + 3, 1:3])]))
                    angles_2_segment.append(np.asarray(angles))
                angles_2_arr = np.vstack((angles_2_segment[:]))
                df_angles_2 = pd.DataFrame({'Molecule': angles_2_arr[:, 0].astype(int),
                                            'Seg_Angle': angles_2_arr[:, 1]})

                angles_3_segment = []
                for mol_pixels in wiggins_pixels_all:
                    angles = []
                    for i in range(0, len(mol_pixels) - 8):
                        angles.append(np.array([int(mol_pixels[i, 0]),
                                                analysis.angle_between(mol_pixels[i + 4, 1:3] - mol_pixels[i + 1, 1:3],
                                                                       mol_pixels[i + 7, 1:3] - mol_pixels[i + 4, 1:3])]))
                    angles_3_segment.append(np.asarray(angles))
                angles_3_arr = np.vstack((angles_3_segment[:]))
                df_angles_3 = pd.DataFrame({'Molecule': angles_3_arr[:, 0].astype(int),
                                            'Seg_Angle': angles_3_arr[:, 1]})

                angles_4_segment = []
                for mol_pixels in wiggins_pixels_all:
                    angles = []
                    for i in range(0, len(mol_pixels) - 10):
                        angles.append(np.array([int(mol_pixels[i, 0]),
                                                analysis.angle_between(mol_pixels[i + 5, 1:3] - mol_pixels[i + 1, 1:3],
                                                                       mol_pixels[i + 9, 1:3] - mol_pixels[i + 5, 1:3])]))
                    angles_4_segment.append(np.asarray(angles))
                angles_4_arr = np.vstack((angles_4_segment[:]))
                df_angles_4 = pd.DataFrame({'Molecule': angles_4_arr[:, 0].astype(int),
                                            'Seg_Angle': angles_4_arr[:, 1]})

                dna_angles = {'z_angles_1': np.array2string(np.asarray(df_angles_1['Seg_Angle']), formatter={'float_kind': '{0:.3f}'.format}),
                              'z_angles_2': np.array2string(np.asarray(df_angles_2['Seg_Angle']), formatter={'float_kind': '{0:.3f}'.format}),
                              'z_angles_3': np.array2string(np.asarray(df_angles_3['Seg_Angle']), formatter={'float_kind': '{0:.3f}'.format}),
                              'z_angles_4': np.array2string(np.asarray(df_angles_4['Seg_Angle']), formatter={'float_kind': '{0:.3f}'.format})}

                self.results.update(dna_angles)

        except:
            self.results['failed'] = True
            self.results['failed_reason'] = 'Angle measurement'



        return self
