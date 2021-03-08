"""
Class to analyze categorized nucleosomes
"""

import copy

import numpy as np
from skimage import morphology

from molecule_categorization import AFMMolecule
import analysis_functions as analysis
import tip_shape_estimation


class NucleosomeEB:

    def __init__(self, afm_molecule, decon):

        # Copy the variables, otherwise they are also changed in the AFMMolecule instances
        self.mol_original = copy.deepcopy(afm_molecule.mol_original)
        self.mol_filtered = copy.deepcopy(afm_molecule.mol_filtered)
        self.anal_pars = copy.deepcopy(afm_molecule.anal_pars)
        self.img_meta_data = copy.deepcopy(afm_molecule.img_meta_data)
        self.mol_pars = copy.deepcopy(afm_molecule.mol_pars)

        self.results = {}
        self.results.update({'position_row': self.mol_pars['mol_bbox'][0],
                             'position_col': self.mol_pars['mol_bbox'][1],
                             'failed': False})
        self.improve_skel()

        if self.results['failed'] is False:
            # Calculate all desired parameters
            self.calculate_rog()
            self.ellipsoid_fit()

        if self.results['failed'] is False:

            if decon['decon'] is False or decon['tip_shape'] is None:
                self.calculate_arm_lengths(self.results['ell_data']['mol_ellipsoid_cut'])
            elif decon['decon'] is True and decon['tip_shape'] is not None:
                self.mol_filtered_decon = tip_shape_estimation.decon_mol(self.mol_filtered, decon['tip_shape'])
                self.calculate_arm_lengths(self.mol_filtered_decon)

        if self.results['failed'] is False:
            self.further_analysis()
            self.nucleosome_volume()
            self.angles()

    def improve_skel(self):
        mol_filtered = copy.deepcopy(self.mol_filtered)
        mol_pars = copy.deepcopy(self.mol_pars)

        # Sort the skeleton of the arm by using the sort_skeleton function
        mol_pars['skel_arm1_sorted'] = analysis.sort_skel(mol_pars, start=mol_pars['skel_eps_pixels'][0, :])

        # Skeletonize the end of the arm
        mol_pars['mol_skel'] = analysis.skeletonize_end(mol_filtered, mol_pars, mol_pars['skel_arm1_sorted'][0:4])

        # Recalculate skeleton parameters
        self.mol_pars.update(analysis.skel_pars(mol_pars['mol_skel']))
        self.mol_pars.update({'mol_skel': mol_pars['mol_skel']})
        mol_pars = copy.deepcopy(self.mol_pars)

        # Sort the skeleton of the arm by using the sort_skeleton function
        mol_pars['skel_arm1_sorted'] = analysis.sort_skel(mol_pars, start=mol_pars['skel_eps_pixels'][0, :])

        # Update the remaining parameters
        self.mol_pars.update({'skel_arm1_sorted': mol_pars['skel_arm1_sorted'],
                              'mol_skel': mol_pars['mol_skel']})

        return self

    def calculate_rog(self):

        # Calculate radius of gyration and center of mass of the whole molecule
        rog_dict = analysis.radius_of_gyration(self.mol_filtered, self.img_meta_data['pixel_size'])
        rog, com = rog_dict['rog'], rog_dict['com']

        # Calculate the same for the nucleosome core particle
        mol_labelled = copy.deepcopy(self.mol_filtered)
        mol_labelled[mol_labelled < self.anal_pars['nuc_min_height']] = 0
        mol_labelled[mol_labelled >= self.anal_pars['nuc_min_height']] = 1
        mol_labelled = morphology.label(mol_labelled, connectivity=2)
        if np.amax(mol_labelled) != 1:
            mol_labelled = morphology.remove_small_objects(mol_labelled, self.mol_pars['max_area_over_height'])
        mol_core_part = copy.deepcopy(self.mol_filtered)
        mol_core_part[mol_labelled == 0] = 0

        rog_core_dict = analysis.radius_of_gyration(mol_core_part, self.img_meta_data['pixel_size'])
        rog_core, com_core = rog_core_dict['rog'], rog_core_dict['com']

        self.results.update({'radius_of_gyration': rog,
                             'radius_of_gyration_core': rog_core,
                             'center_of_mass': com,
                             'center_of_mass_core': com_core})

        return self

    def ellipsoid_fit(self):

        ell_data = analysis.ellipsoid_fit(self.mol_filtered, self.results['center_of_mass_core'])
        self.results.update({'ell_data': ell_data})
        if 'failed' in ell_data:
            self.results.update({'failed': True,
                                 'failed_reason': 'Ellipsoid fit'})
        if 'failed' not in ell_data:
            self.results.update({'ell_a': ell_data['abc'][0],
                                 'ell_b': ell_data['abc'][1],
                                 'ell_height': ell_data['abc'][2],
                                 'ell_rot': ell_data['rot_angle'] * 180 / np.pi})

        return self

    def calculate_arm_lengths(self, mol_filtered):

        mol_filtered = copy.deepcopy(mol_filtered)
        mol_pars = copy.deepcopy(self.mol_pars)
        ell_data = self.results['ell_data']
        pixel_size = self.img_meta_data['pixel_size']
        seg_length = 5/pixel_size
        failed = self.results['failed']

        # Find the Wiggins pixels of the arm starting at the top-leftmost arm-endpoint
        pixels_arm1, failed_arm1 = analysis.wiggins(mol_filtered,
                                                    seg_length=seg_length,
                                                    start=mol_pars['skel_arm1_sorted'][:4],
                                                    end=mol_pars['skel_arm1_sorted'][-1],
                                                    mol_type='Nucleosome',
                                                    ell_data=self.results['ell_data'],
                                                    )

        if failed_arm1 is False:

            # Find point on ellipse and remove all points within the ellipse
            ell_pixel = analysis.ellipse_arm_pixel(pixels_arm1, ell_data, ell_cutoff=0.6)
            while np.linalg.norm((ell_data['center'] - ell_pixel)) >= np.linalg.norm(
                    (ell_data['center'] - pixels_arm1[-1])):
                del pixels_arm1[-1]
            ell_pixel = analysis.ellipse_arm_pixel(pixels_arm1, ell_data, ell_cutoff=0.6)
            pixels_arm1.append(ell_pixel)
            pixels_arm1 = np.asarray(pixels_arm1)

            length_arm1_60 = np.sum(np.linalg.norm(pixels_arm1[:-1] - pixels_arm1[1:], axis=1)) * pixel_size

            ell_pixel_50 = analysis.ellipse_arm_pixel(pixels_arm1[:-1], ell_data, ell_cutoff=0.5)
            length_arm1_50 = length_arm1_60 - np.linalg.norm(ell_pixel_50 - pixels_arm1[-1]) * pixel_size

            ell_pixel_70 = analysis.ellipse_arm_pixel(pixels_arm1[:-1], ell_data, ell_cutoff=0.7)
            length_arm1_70 = length_arm1_60 + np.linalg.norm(ell_pixel_70 - pixels_arm1[-1]) * pixel_size

        else:
            failed = True
            length_arm1_50 = False
            length_arm1_60 = False
            length_arm1_70 = False

        self.results.update({'pixels_arm1': pixels_arm1,
                             'length_arm1_70': length_arm1_70,
                             'length_arm1_60': length_arm1_60,
                             'length_arm1_50': length_arm1_50,
                             'failed': failed})

        self.results.update({'arm1_end_r': self.results['pixels_arm1'][-1][0],
                             'arm1_end_c': self.results['pixels_arm1'][-1][1],
                             'ell_center_r': ell_data['center'][0],
                             'ell_center_c': ell_data['center'][1]})

        return self

    def further_analysis(self):
        """ Check the height values along the Wiggins pixels and the slope between individual pixels """
        pixel_size = self.img_meta_data['pixel_size']

        if self.results['failed'] is False:
            height_pars_arm1 = analysis.wiggins_pixel_height_analysis(self.results['pixels_arm1'],
                                                                      self.mol_filtered, pixel_size)
            orientation_pars_arm1 = analysis.dna_orientation(self.results['pixels_arm1'],
                                                             self.mol_pars['mol_bbox'])

            self.results.update({'height_avg': height_pars_arm1['height_avg'],
                                 'slope_avg': height_pars_arm1['slope_avg'],
                                 'height_std': height_pars_arm1['height_std'],
                                 'slope_std': height_pars_arm1['slope_std'],
                                 'extension_right': orientation_pars_arm1['extension_right'],
                                 'extension_bot': orientation_pars_arm1['extension_bot'],
                                 'rightness_arm1': orientation_pars_arm1['rightness']
                                 })

        return self

    def nucleosome_volume(self):
        """ Calculate the nucleosome volume based on the pixels within the fitted ground ellipse """
        ell_data = self.results['ell_data']
        mol_pixel_locs = np.where(self.mol_filtered != 0)

        # Pre-filter potential nucleosome pixels by selecting the ones within the range of the largest ellipse axis
        mol_pixels = [np.array([r, c]) for r, c in zip(mol_pixel_locs[0], mol_pixel_locs[1])
                      if np.linalg.norm((np.array([r, c]) - ell_data['center'])) <= np.amax(ell_data['abc'])]

        # Find all pixels whose centers lie within the ground ellipse of the fitted ellipsoid
        inner_pixels = [pixel for pixel in mol_pixels
                        if np.linalg.norm((analysis.ellipse_arm_pixel([pixel], ell_data, ell_cutoff=0) - ell_data['center']))
                        >= np.linalg.norm((pixel - ell_data['center']))]

        pixel_heights = [self.mol_filtered[r, c] for r, c in inner_pixels]
        self.results.update({'nucleosome_volume': sum(pixel_heights) * self.img_meta_data['pixel_size']**2})

        # Find all pixels whose centers lie within the z_h=0.6 ellipse of the fitted ellipsoid
        inner_pixels = [pixel for pixel in mol_pixels
                        if np.linalg.norm((analysis.ellipse_arm_pixel([pixel], ell_data, ell_cutoff=0.6) - ell_data['center']))
                        >= np.linalg.norm((pixel - ell_data['center']))]

        pixel_heights = [self.mol_filtered[r, c] for r, c in inner_pixels]

        self.results.update({'nucleosome_volume_core': sum(pixel_heights) * self.img_meta_data['pixel_size']**2,
                             'nuc_max_height': max(pixel_heights),
                             'nuc_max_height_avg': np.mean(sorted(pixel_heights)[-5:])})

        return self

    def angles(self):

        try:
            # Only calculate angles for 5nm segments
            arm1 = np.asarray(self.results['pixels_arm1'])
            arm1 = arm1[0:(len(np.where(np.linalg.norm(arm1[:-1] - arm1[1:], axis=1) > 1.51)[0]) + 1)]

            # Angles between individual segments
            vecs_1 = arm1[2:] - arm1[1:-1]
            angles_1 = np.asarray([analysis.angle_between(v1, v2) for v1, v2 in zip(vecs_1[1:], vecs_1[:-1])])

            # Angles between two consecutive segments
            vecs_2 = arm1[3:] - arm1[1:-2]
            angles_2 = np.asarray([analysis.angle_between(v1, v2) for v1, v2 in zip(vecs_2[2:], vecs_2[:-2])])

            dna_angles_dict = {'z_angles_arm1_1': np.array2string(angles_1, formatter={'float_kind': '{0:.3f}'.format}),
                               'z_angles_arm1_2': np.array2string(angles_2, formatter={'float_kind': '{0:.3f}'.format})}
            self.results.update(dna_angles_dict)

        except:
            dna_angles_dict = {'z_angles_arm1_1': None,
                               'z_angles_arm1_2': None}
            self.results.update(dna_angles_dict)

        return self
