"""
Class to analyze categorized nucleosomes
"""

import copy

import numpy as np
from skimage import morphology

from molecule_categorization import AFMMolecule
import analysis_functions as analysis


class Nucleosome(AFMMolecule):

    def __init__(self, mol, dna_bp, pixel_size, background_1, background_2, min_area, nuc_min_height, nuc_min_area, mol_pars):
        super().__init__(mol, dna_bp, pixel_size, background_1, background_2, min_area, nuc_min_height, nuc_min_area)
        # Copy the variable, otherwise they are also changed in the AFMMolecule instances
        self.mol_pars = copy.deepcopy(mol_pars)
        self.results = {}
        self.results.update({'position_row': self.mol_bbox[0],
                             'position_col': self.mol_bbox[1],
                             'failed': False})
        self.improve_skel()

        if self.results['failed'] is False:
            # Calculate all desired parameters
            self.calculate_rog()
            self.ellipsoid_fit()
        if self.results['failed'] is False:
            self.calculate_arm_lengths()
            self.further_analysis()
            self.wrapping_angle()
            self.nucleosome_volume()

    def improve_skel(self):
        mol_filtered = copy.deepcopy(self.mol_filtered)
        mol_pars = copy.deepcopy(self.mol_pars)

        # Sort the skeleton of the two individual arms by using the sort_skeleton function
        mol_pars['skel_arm1_sorted'] = analysis.sort_skel(mol_pars, start=mol_pars['skel_eps_pixels'][0, :])
        mol_pars['skel_arm2_sorted'] = analysis.sort_skel(mol_pars, start=mol_pars['skel_eps_pixels'][1, :])

        # Skeletonize the ends of the two arms
        mol_pars['mol_skel'] = analysis.skeletonize_end(mol_filtered, mol_pars, mol_pars['skel_arm1_sorted'][0:4])
        mol_pars['mol_skel'] = analysis.skeletonize_end(mol_filtered, mol_pars, mol_pars['skel_arm2_sorted'][0:4])

        # Recalculate skeleton parameters
        eps_pixels, eps_number, bps_pixels, bps_number, pixels_number = analysis.skel_pars(mol_pars['mol_skel'])
        self.mol_pars.update({'skel_eps_pixels': eps_pixels,
                              'skel_eps_number': eps_number,
                              'skel_bps_pixels': bps_pixels,
                              'skel_bps_number': bps_number,
                              'skel_pixels_number': pixels_number,
                              'mol_skel': mol_pars['mol_skel']})

        # Sort the skeleton of the two individual arms by using the sort_skeleton function
        # Set nucleosome to failed in case that there are no two endpoints anymore after the endpoint skeletonization
        # failed
        if len(eps_pixels) != 2:
            self.results.update({'failed': True,
                                 'failed_reason': 'Endpoints were prolonged wrongfully'})
            self.mol_pars.update({'skel_eps_pixels': np.vstack((eps_pixels, eps_pixels))})

        else:
            mol_pars = copy.deepcopy(self.mol_pars)
            mol_pars['skel_arm1_sorted'] = analysis.sort_skel(mol_pars, start=mol_pars['skel_eps_pixels'][0, :])
            mol_pars['skel_arm2_sorted'] = analysis.sort_skel(mol_pars, start=mol_pars['skel_eps_pixels'][1, :])

            # Update the remaining parameters
            self.mol_pars.update({'skel_arm1_sorted': mol_pars['skel_arm1_sorted'],
                                  'skel_arm2_sorted': mol_pars['skel_arm2_sorted'],
                                  'mol_skel': mol_pars['mol_skel']})

        return self

    def calculate_rog(self):
        # Calculate radius of gyration and center of mass of the whole molecule
        rog, com = analysis.radius_of_gyration(self.mol_filtered)

        # Calculate the same for the nucleosome core particle
        mol_labelled = copy.deepcopy(self.mol_filtered)
        mol_labelled[mol_labelled < self.nuc_min_height] = 0
        mol_labelled[mol_labelled >= self.nuc_min_height] = 1
        mol_labelled = morphology.label(mol_labelled, connectivity=2)
        if np.amax(mol_labelled) != 1:
            mol_labelled = morphology.remove_small_objects(mol_labelled, self.mol_pars['max_area_over_height'])
        mol_core_part = copy.deepcopy(self.mol_filtered)
        mol_core_part[mol_labelled == 0] = 0

        rog_core, com_core = analysis.radius_of_gyration(mol_core_part)

        self.results.update({'radius_of_gyration': rog,
                             'radius_of_gyration_core': rog_core,
                             'center_of_mass': com,
                             'center_of_mass_core': com_core})

        return self

    def ellipsoid_fit(self):

        try:
            coeff, var_matrix, mol_nuc_ellipsoid_cut, ellipsoid_pixels = analysis.\
                nuc_core_ellipsoid_fit(self.mol_filtered,
                                       self.results['center_of_mass_core'])
            self.results.update({'ellipsoid_coeff': coeff,
                                 'ellipsoid_width_a': coeff[2],
                                 'ellipsoid_width_b': coeff[3],
                                 'ellipsoid_height': coeff[4],
                                 'ellipsoid_angle': coeff[5] * 180 / np.pi,
                                 'ellipsoid_var_matrix': var_matrix,
                                 'mol_nuc_ellipsoid_cut': mol_nuc_ellipsoid_cut,
                                 'ellipsoid_pixels': ellipsoid_pixels})
        except:
            self.results.update({'failed': True,
                                 'failed_reason': 'Ellipsoid fit'})

        return self

    def calculate_arm_lengths(self):

        mol_filtered = copy.deepcopy(self.results['mol_nuc_ellipsoid_cut'])
        mol_original = copy.deepcopy(self.mol_original)
        mol_pars = copy.deepcopy(self.mol_pars)
        seg_length = 5/self.pixel_size
        failed = self.results['failed']

        # Find the Wiggins pixels of the arm starting at the top-leftmost arm-endpoint
        pixels_arm1, failed_arm1 = analysis.wiggins(mol_filtered,
                                                    seg_length=seg_length,
                                                    start=mol_pars['skel_arm1_sorted'][:4],
                                                    end=mol_pars['skel_arm1_sorted'][-1],
                                                    mol_type='Nucleosome',
                                                    ellipsoid_coeff=self.results['ellipsoid_coeff'],
                                                    )

        # Apply the new arm tracing close to the nucleosome core particle
        starting_point = None
        if 2 <= len(pixels_arm1) < 5:
            starting_point = len(pixels_arm1)
        elif len(pixels_arm1) >= 5:
            starting_point = 5
        if failed_arm1 is False and starting_point is not None:
            pixels_arm1_end, failed_arm1 = analysis.interp_tracing_end(mol_original,
                                                                       pixels_arm1[-starting_point:(-starting_point + 2)]
                                                                       if starting_point != 2
                                                                       else pixels_arm1[-starting_point:],
                                                                       self.results['ellipsoid_pixels'],
                                                                       seg_length=seg_length
                                                                       )
            del pixels_arm1[-starting_point:]
            pixels_arm1 = pixels_arm1 + pixels_arm1_end

        # pixels_arm1, failed_arm1 = analysis.interp_tracing(mol_original,
        #                                                    mol_pars['skel_arm1_sorted'],
        #                                                    self.results['ellipsoid_pixels']
        #                                                    )

        if failed_arm1 is False:
            # Find the pixel of the arm that lies on the ellipse when connecting the last arm pixels and the ell_center
            ell_pixel = analysis.ellipse_arm_pixel(pixels_arm1, self.results['ellipsoid_coeff'], z_h=0.6, first=True)
            # Remove all arm pixels inside the ellipse
            while np.linalg.norm((self.results['ellipsoid_coeff'][0:2] - ell_pixel)) >= np.linalg.norm(
                    (self.results['ellipsoid_coeff'][0:2] - pixels_arm1[-1])):
                del pixels_arm1[-1]
            # Recalculate the pixel based on the last pixel in the arm
            ell_pixel = analysis.ellipse_arm_pixel(pixels_arm1, self.results['ellipsoid_coeff'], z_h=0.6)
            pixels_arm1.append(ell_pixel)
            length_arm1_60 = np.sum([np.linalg.norm(pixels_arm1[i] - pixels_arm1[i + 1])
                                     for i in range(0, len(pixels_arm1) - 1)]) * self.pixel_size

            ell_pixel_40 = analysis.ellipse_arm_pixel(pixels_arm1[:-1], self.results['ellipsoid_coeff'], z_h=0.4)
            length_arm1_40 = length_arm1_60 - np.linalg.norm(ell_pixel_40 - pixels_arm1[-1]) * self.pixel_size

            ell_pixel_50 = analysis.ellipse_arm_pixel(pixels_arm1[:-1], self.results['ellipsoid_coeff'], z_h=0.5)
            length_arm1_50 = length_arm1_60 - np.linalg.norm(ell_pixel_50 - pixels_arm1[-1]) * self.pixel_size

            ell_pixel_70 = analysis.ellipse_arm_pixel(pixels_arm1[:-1], self.results['ellipsoid_coeff'], z_h=0.7)
            length_arm1_70 = length_arm1_60 + np.linalg.norm(ell_pixel_70 - pixels_arm1[-1]) * self.pixel_size

        else:
            failed = True
            length_arm1_40 = False
            length_arm1_50 = False
            length_arm1_60 = False
            length_arm1_70 = False

        # Find the Wiggins pixels of the arm starting at the bot-rightmost arm-endpoint
        pixels_arm2, failed_arm2 = analysis.wiggins(mol_filtered,
                                                    seg_length=seg_length,
                                                    start=mol_pars['skel_arm2_sorted'][:4],
                                                    end=mol_pars['skel_arm2_sorted'][-1],
                                                    mol_type='Nucleosome',
                                                    ellipsoid_coeff=self.results['ellipsoid_coeff'],
                                                    )

        # Apply the new arm tracing close to the nucleosome core particle
        starting_point = None
        if 2 <= len(pixels_arm2) < 5:
            starting_point = len(pixels_arm2)
        elif len(pixels_arm2) >= 5:
            starting_point = 5
        if failed_arm2 is False and starting_point is not None:
            pixels_arm2_end, failed_arm2 = analysis.interp_tracing_end(mol_original,
                                                                       pixels_arm2[-starting_point:(-starting_point + 2)]
                                                                       if starting_point != 2
                                                                       else pixels_arm2[-starting_point:],
                                                                       self.results['ellipsoid_pixels'],
                                                                       seg_length=seg_length
                                                                       )
            del pixels_arm2[-starting_point:]
            pixels_arm2 = pixels_arm2 + pixels_arm2_end

        # pixels_arm2, failed_arm2 = analysis.interp_tracing(mol_original,
        #                                                    mol_pars['skel_arm2_sorted'],
        #                                                    self.results['ellipsoid_pixels']
        #                                                    )

        if failed_arm2 is False:
            # Find the pixel of the arm that lies on the ellipse when connecting the last arm pixels and the ell_center
            ell_pixel = analysis.ellipse_arm_pixel(pixels_arm2, self.results['ellipsoid_coeff'], z_h=0.6, first=True)
            # Remove all arm pixels inside the ellipse with check that first pixel is not already inside
            while np.linalg.norm((self.results['ellipsoid_coeff'][0:2] - ell_pixel)) >= np.linalg.norm(
                    (self.results['ellipsoid_coeff'][0:2] - pixels_arm2[-1])):
                del pixels_arm2[-1]
            ell_pixel = analysis.ellipse_arm_pixel(pixels_arm2, self.results['ellipsoid_coeff'], z_h=0.6)
            pixels_arm2.append(ell_pixel)
            length_arm2_60 = np.sum([np.linalg.norm(pixels_arm2[i] - pixels_arm2[i + 1])
                                     for i in range(0, len(pixels_arm2) - 1)]) * self.pixel_size

            ell_pixel_40 = analysis.ellipse_arm_pixel(pixels_arm2, self.results['ellipsoid_coeff'], z_h=0.4)
            length_arm2_40 = length_arm2_60 - np.linalg.norm(ell_pixel_40 - pixels_arm2[-1]) * self.pixel_size

            ell_pixel_50 = analysis.ellipse_arm_pixel(pixels_arm2, self.results['ellipsoid_coeff'], z_h=0.5)
            length_arm2_50 = length_arm2_60 - np.linalg.norm(ell_pixel_50 - pixels_arm2[-1]) * self.pixel_size

            ell_pixel_70 = analysis.ellipse_arm_pixel(pixels_arm2, self.results['ellipsoid_coeff'], z_h=0.7)
            length_arm2_70 = length_arm2_60 + np.linalg.norm(ell_pixel_70 - pixels_arm2[-1]) * self.pixel_size

        else:
            failed = True
            length_arm2_40 = False
            length_arm2_50 = False
            length_arm2_60 = False
            length_arm2_70 = False

        self.results.update({'pixels_arm1': pixels_arm1,
                             'pixels_arm2': pixels_arm2,
                             'length_arm1_70': length_arm1_70,
                             'length_arm2_70': length_arm2_70,
                             'length_arm1_60': length_arm1_60,
                             'length_arm2_60': length_arm2_60,
                             'length_arm1_50': length_arm1_50,
                             'length_arm2_50': length_arm2_50,
                             'length_arm1_40': length_arm1_40,
                             'length_arm2_40': length_arm2_40,
                             'length_sum': length_arm1_60 + length_arm2_60 if failed is False else False,
                             'length_etoe': np.linalg.norm(pixels_arm1[0] - pixels_arm2[0]) * self.pixel_size,
                             'failed': failed})
        if self.results['length_sum'] is False:
            self.results.update({'failed_reason': 'Arm Failed'})

        self.results.update({'arm1_end_r': self.results['pixels_arm1'][-1][0],
                             'arm1_end_c': self.results['pixels_arm1'][-1][1],
                             'arm2_end_r': self.results['pixels_arm2'][-1][0],
                             'arm2_end_c': self.results['pixels_arm2'][-1][1],
                             'ell_center_r': self.results['ellipsoid_coeff'][0],
                             'ell_center_c': self.results['ellipsoid_coeff'][1]})

        return self

    def further_analysis(self):
        """ Check the height values along the Wiggins pixels and the slope between individual pixels """

        if self.results['failed'] is False:
            height_pars_arm1 = analysis.wiggins_pixel_height_analysis(self.results['pixels_arm1'],
                                                                      self.mol_filtered, self.pixel_size)

            height_pars_arm2 = analysis.wiggins_pixel_height_analysis(self.results['pixels_arm2'],
                                                                      self.mol_filtered, self.pixel_size)

            orientation_pars_arm1 = analysis.dna_orientation(self.results['pixels_arm1'],
                                                             self.mol_bbox)

            orientation_pars_arm2 = analysis.dna_orientation(self.results['pixels_arm2'],
                                                             self.mol_bbox)

            bending_pars_arm1 = analysis.bending_behaviour(self.results['pixels_arm1'])

            bending_pars_arm2 = analysis.bending_behaviour(self.results['pixels_arm2'])

            # Sum and weigh the height/slope parameters of the two arms based on their length
            height_avg = (height_pars_arm1['height_avg'] * self.results['length_arm1_60'] +
                          height_pars_arm2['height_avg'] * self.results['length_arm2_60']) / self.results['length_sum']
            slope_avg = (height_pars_arm1['slope_avg'] * self.results['length_arm1_60'] +
                         height_pars_arm2['slope_avg'] * self.results['length_arm2_60']) / self.results['length_sum']
            height_std = (height_pars_arm1['height_std'] * self.results['length_arm1_60'] +
                          height_pars_arm2['height_std'] * self.results['length_arm2_60']) / self.results['length_sum']
            slope_std = (height_pars_arm1['slope_std'] * self.results['length_arm1_60'] +
                         height_pars_arm2['slope_std'] * self.results['length_arm2_60']) / self.results['length_sum']
            # Cutout a 28x28 pixel frame around the nuc core
            size = 14
            com_r = int(np.round(self.results['ellipsoid_coeff'][0]))
            com_c = int(np.round(self.results['ellipsoid_coeff'][1]))
            r_rand = int(np.round(np.random.random()))
            c_rand = int(np.round(np.random.random()))
            nuc_cutout = self.mol_filtered[com_r - size + r_rand: com_r + size + r_rand,
                                           com_c - size + c_rand: com_c + size + c_rand]

            self.results.update({'height_avg': height_avg,
                                 'slope_avg': slope_avg,
                                 'height_std': height_std,
                                 'slope_std': slope_std,
                                 'extension_right': orientation_pars_arm1['extension_right'] * self.pixel_size,
                                 'extension_bot': orientation_pars_arm1['extension_bot'] * self.pixel_size,
                                 'rightness_arm1': orientation_pars_arm1['rightness'],
                                 'rightness_arm2': orientation_pars_arm2['rightness'],
                                 'bending_avg_arm1': bending_pars_arm1['bending_avg'],
                                 'bending_avg_arm2': bending_pars_arm2['bending_avg'],
                                 'nuc_cutout': nuc_cutout,
                                 'z_nuc_cutout_str': np.array2string(nuc_cutout, formatter={'float_kind': '{0:.3f}'.format})
                                 })

        return self

    def wrapping_angle(self):

        arm1_vector = self.results['pixels_arm1'][-1] - self.results['ellipsoid_coeff'][0:2]
        arm2_vector = self.results['pixels_arm2'][-1] - self.results['ellipsoid_coeff'][0:2]
        self.results.update({'angle_arms': analysis.angle_between(arm1_vector, arm2_vector)})

        return self

    def nucleosome_volume(self):
        """ Calculate the nucleosome volume based on the pixels within the fitted ground ellipse """
        ell_coeff = self.results['ellipsoid_coeff']
        mol_pixel_locs = np.where(self.mol_filtered != 0)

        # Pre-filter potential nucleosome pixels by selecting the ones within the range of the largest ellipse axis
        mol_pixels = [np.array([r, c]) for r, c in zip(mol_pixel_locs[0], mol_pixel_locs[1])
                      if np.linalg.norm((np.array([r, c]) - ell_coeff[0:2])) <= np.amax(ell_coeff[2:4])]

        # Find all pixels whose centers lie within the ground ellipse of the fitted ellipsoid
        inner_pixels = [pixel for pixel in mol_pixels
                        if np.linalg.norm((analysis.ellipse_arm_pixel([pixel], ell_coeff, z_h=0) - ell_coeff[0:2]))
                        >= np.linalg.norm((pixel - ell_coeff[0:2]))]

        pixel_heights = [self.mol_filtered[r, c] for r, c in inner_pixels]
        self.results.update({'nucleosome_volume': sum(pixel_heights) * self.pixel_size**2})

        # Find all pixels whose centers lie within the z_h=0.6 ellipse of the fitted ellipsoid
        inner_pixels = [pixel for pixel in mol_pixels
                        if np.linalg.norm((analysis.ellipse_arm_pixel([pixel], ell_coeff, z_h=0.6) - ell_coeff[0:2]))
                        >= np.linalg.norm((pixel - ell_coeff[0:2]))]

        pixel_heights = [self.mol_filtered[r, c] for r, c in inner_pixels]
        try: # make this better
            self.results.update({'nucleosome_volume_core': sum(pixel_heights) * self.pixel_size**2,
                                 'nuc_max_height': max(pixel_heights),
                                 'nuc_max_height_avg': np.mean(sorted(pixel_heights)[-5:])})
        except:
            self.results.update({'failed': True})

        return self
