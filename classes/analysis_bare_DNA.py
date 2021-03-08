"""
Class to analyze categorized bare DNA
"""

import copy
import numpy as np

import analysis_functions as analysis
import tip_shape_estimation


class BareDNA:

    def __init__(self, afm_molecule, decon):

        # Copy the variables, otherwise they are also changed in the AFMMolecule instances
        self.mol_original = copy.deepcopy(afm_molecule.mol_original)
        self.mol_filtered = copy.deepcopy(afm_molecule.mol_filtered)
        self.anal_pars = copy.deepcopy(afm_molecule.anal_pars)
        self.img_meta_data = copy.deepcopy(afm_molecule.img_meta_data)
        self.mol_pars = copy.deepcopy(afm_molecule.mol_pars)

        # Improve the skeleton by skeletonizing the ends and sorting it
        self.improve_skel()

        # Apply Wiggin's algorithm and store the calculated values in the results dict
        self.results = {}
        self.results.update({'position_row': self.mol_pars['mol_bbox'][0],
                             'position_col': self.mol_pars['mol_bbox'][1]})

        if decon['tip_shape'] is None:
            self.calculate_length(self.mol_filtered)
        elif decon['tip_shape'] is not None:
            self.results.update({'tip_shape': np.array2string(decon['tip_shape']['tip_shape_arr'],
                                                              formatter={'float_kind': '{0:.3f}'.format}),
                                 'tip_excentricity': decon['tip_shape']['tip_excentricity']})
            self.mol_filtered_decon = tip_shape_estimation.decon_mol(self.mol_filtered, decon['tip_shape'])
            self.calculate_length(self.mol_filtered_decon)

        self.results.update(analysis.radius_of_gyration(self.mol_filtered, self.img_meta_data['pixel_size']))
        self.further_analysis()
        self.angles()

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
        self.mol_pars.update(analysis.skel_pars(mol_pars['mol_skel']))

        # Sort the skeleton with the new endpoints
        mol_pars['skel_sorted'] = analysis.sort_skel(mol_pars, start=self.mol_pars['skel_eps_pixels'][0, :])

        # Update the remaining parameters
        self.mol_pars.update({'skel_sorted': mol_pars['skel_sorted'],
                              'mol_skel': mol_pars['mol_skel']})
        return self

    def calculate_length(self, mol_filtered):
        """ Use Wiggin's algorithm to calculate the DNA lengths """
        mol_filtered = copy.deepcopy(mol_filtered)
        mol_pars = copy.deepcopy(self.mol_pars)
        pixel_size = self.img_meta_data['pixel_size']
        seg_length = 5/pixel_size

        # Calculate the length starting at the top-leftmost endpoint
        wigg_fwd, failed_fwd = analysis.wiggins(mol_filtered, seg_length=seg_length,
                                                start=mol_pars['skel_sorted'][:4], end=mol_pars['skel_sorted'][-1],
                                                mol_type='Bare DNA')
        if failed_fwd is False:
            wigg_fwd = np.asarray(wigg_fwd)
            length_fwd = np.sum(np.linalg.norm(wigg_fwd[:-1] - wigg_fwd[1:], axis=1)) * pixel_size
        else:
            length_fwd = False

        # Calculate the length starting at the bottom-rightmost endpoint
        wigg_bwd, failed_bwd = analysis.wiggins(mol_filtered, seg_length=seg_length,
                                                start=mol_pars['skel_sorted'][:-5:-1], end=mol_pars['skel_sorted'][0],
                                                mol_type='Bare DNA')
        if failed_bwd is False:
            wigg_bwd = np.asarray(wigg_bwd)
            length_bwd = np.sum(np.linalg.norm(wigg_bwd[:-1] - wigg_bwd[1:], axis=1)) * pixel_size
        else:
            length_bwd = False

        if failed_fwd is False and failed_bwd is False:
            length_avg = (length_fwd + length_bwd) / 2
            length_etoe = np.linalg.norm(wigg_fwd[0] - wigg_fwd[-1]) * pixel_size
        elif failed_fwd is False and failed_bwd is True:
            length_avg = length_fwd
            length_etoe = np.linalg.norm(wigg_fwd[0] - wigg_fwd[-1]) * pixel_size
        elif failed_fwd is True and failed_bwd is False:
            length_avg = length_bwd
            length_etoe = np.linalg.norm(wigg_bwd[0] - wigg_bwd[-1]) * pixel_size
        else:
            length_avg = False
            length_etoe = False
            self.results.update({'failed_reason': 'Wiggins failed'})

        self.results.update({'wigg_fwd': wigg_fwd,
                             'wigg_bwd': wigg_bwd,
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
        dna_bp = self.anal_pars['dna_length_bp']

        if self.results['length_fwd'] <= 0.80*dna_bp*0.33 or self.results['length_bwd'] <= 0.80*dna_bp*0.33:
            self.results.update({'length_avg': False,
                                 'failed_reason': 'Too short',
                                 'failed': True})

        elif self.results['length_fwd'] >= 1.25*dna_bp*0.33 or self.results['length_bwd'] >= 1.25*dna_bp*0.33:
            self.results.update({'length_avg': False,
                                 'failed_reason': 'Too long',
                                 'failed': True})

        return self

    def further_analysis(self):
        """ Check the height values along the Wiggins pixels and the slope between individual pixels """
        pixel_size = self.img_meta_data['pixel_size']
        mol_bbox = self.mol_pars['mol_bbox']

        if self.results['failed'] is False:
            if self.results['length_fwd'] is not False:
                height_pars = analysis.wiggins_pixel_height_analysis(self.results['wigg_fwd'],
                                                                     self.mol_filtered, pixel_size)
                orientation_pars = analysis.dna_orientation(self.results['wigg_fwd'], mol_bbox)

            elif self.results['length_bwd'] is not False:
                height_pars = analysis.wiggins_pixel_height_analysis(self.results['wigg_bwd'],
                                                                     self.mol_filtered, pixel_size)
                orientation_pars = analysis.dna_orientation(self.results['wigg_fwd'], mol_bbox)

            self.results.update(height_pars)
            self.results.update(orientation_pars)

        return self

    def angles(self):

        try:
            if self.results['failed'] is False:
                if self.results['wigg_fwd'] is not False:
                    wigg_pixels = np.asarray(self.results['wigg_fwd'])
                else:
                    wigg_pixels = np.asarray(self.results['wigg_bwd'])

            # Angles between individual segments
            vecs_1 = wigg_pixels[2:-1] - wigg_pixels[1:-2]
            angles_1 = np.asarray([analysis.angle_between(v1, v2) for v1, v2 in zip(vecs_1[1:], vecs_1[:-1])])

            # Angles between two consecutive segments
            vecs_2 = wigg_pixels[3:-1] - wigg_pixels[1:-3]
            angles_2 = np.asarray([analysis.angle_between(v1, v2) for v1, v2 in zip(vecs_2[2:], vecs_2[:-2])])

            # Angles between three consecutive segments
            vecs_3 = wigg_pixels[4:-1] - wigg_pixels[1:-4]
            angles_3 = np.asarray([analysis.angle_between(v1, v2) for v1, v2 in zip(vecs_3[3:], vecs_3[:-3])])

            # Angles between four consecutive segments
            vecs_4 = wigg_pixels[5:-1] - wigg_pixels[1:-5]
            angles_4 = np.asarray([analysis.angle_between(v1, v2) for v1, v2 in zip(vecs_4[4:], vecs_4[:-4])])

            dna_angles_dict = {'z_angles_1': np.array2string(angles_1, formatter={'float_kind': '{0:.3f}'.format}),
                               'z_angles_2': np.array2string(angles_2, formatter={'float_kind': '{0:.3f}'.format}),
                               'z_angles_3': np.array2string(angles_3, formatter={'float_kind': '{0:.3f}'.format}),
                               'z_angles_4': np.array2string(angles_4, formatter={'float_kind': '{0:.3f}'.format})}
            self.results.update(dna_angles_dict)

        except:
            self.results['failed'] = True
            self.results['failed_reason'] = 'Angle measurement'

        return self
