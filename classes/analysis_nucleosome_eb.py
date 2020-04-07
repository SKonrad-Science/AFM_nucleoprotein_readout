"""
Class to analyze categorized nucleosomes
"""

import copy

import numpy as np
from skimage import morphology

from molecule_categorization import AFMMolecule
import analysis_functions as analysis


class NucleosomeEB(AFMMolecule):

    def __init__(self, mol, dna_bp, pixel_size, background_1, background_2, min_area, nuc_min_height, nuc_min_area, mol_pars):
        super().__init__(mol, dna_bp, pixel_size, background_1, background_2, min_area, nuc_min_height, nuc_min_area)
        # Copy the variable, otherwise they are also changed in the AFMMolecule instances
        self.mol_pars = copy.deepcopy(mol_pars)
        self.improve_skel()

        # Calculate all desired parameters
        self.results = {}
        self.results.update({'position_row': self.mol_bbox[0],
                             'position_col': self.mol_bbox[1],
                             'failed': False})
        self.calculate_rog()
        self.ellipsoid_fit()
        if self.results['failed'] is False:
            self.calculate_arm_lengths()
            self.further_analysis()
            self.nucleosome_volume()

    def improve_skel(self):
        mol_filtered = copy.deepcopy(self.mol_filtered)
        mol_pars = copy.deepcopy(self.mol_pars)

        # Sort the skeleton of the arm by using the sort_skeleton function
        mol_pars['skel_arm1_sorted'] = analysis.sort_skel(mol_pars, start=mol_pars['skel_eps_pixels'][0, :])

        # Skeletonize the end of the arm
        mol_pars['mol_skel'] = analysis.skeletonize_end(mol_filtered, mol_pars, mol_pars['skel_arm1_sorted'][0:4])

        # Recalculate skeleton parameters
        eps_pixels, eps_number, bps_pixels, bps_number, pixels_number = analysis.skel_pars(mol_pars['mol_skel'])
        self.mol_pars.update({'skel_eps_pixels': eps_pixels,
                              'skel_eps_number': eps_number,
                              'skel_bps_pixels': bps_pixels,
                              'skel_bps_number': bps_number,
                              'skel_pixels_number': pixels_number,
                              'mol_skel': mol_pars['mol_skel']})

        mol_pars = copy.deepcopy(self.mol_pars)
        # Sort the skeleton of the arm by using the sort_skeleton function
        mol_pars['skel_arm1_sorted'] = analysis.sort_skel(mol_pars, start=mol_pars['skel_eps_pixels'][0, :])

        # Update the remaining parameters
        self.mol_pars.update({'skel_arm1_sorted': mol_pars['skel_arm1_sorted'],
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
            coeff, var_matrix, mol_nuc_ellipsoid_cut, ellipsoid_pixels = analysis.nuc_core_ellipsoid_fit(self.mol_filtered,
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
            print('Ellipsoid Fit failed')
            self.results.update({'failed': True,
                                 'failed_reason': 'Ellipsoid fit'})

        return self

    def calculate_arm_lengths(self):

        mol_filtered = copy.deepcopy(self.results['mol_nuc_ellipsoid_cut'])
        # mol_filtered = copy.deepcopy(self.mol_filtered)
        mol_pars = copy.deepcopy(self.mol_pars)
        seg_length = 5/self.pixel_size
        failed = False

        # Find the Wiggins pixels of the arm starting at the top-leftmost arm-endpoint
        pixels_arm1, failed_arm1 = analysis.wiggins(mol_filtered,
                                                    seg_length=seg_length,
                                                    start=mol_pars['skel_arm1_sorted'][:4],
                                                    end=mol_pars['skel_arm1_sorted'][-1],
                                                    mol_type='Nucleosome',
                                                    ellipsoid_coeff=self.results['ellipsoid_coeff'],
                                                    )

        if failed_arm1 is False:
            # Find the pixel of the arm that lies on the ellipse when connecting the last arm pixels and the ell_center
            ell_pixel = analysis.ellipse_arm_pixel(pixels_arm1, self.results['ellipsoid_coeff'], z_h=0.6)

            # Remove all arm pixels inside the ellipse
            while np.linalg.norm((self.results['ellipsoid_coeff'][0:2] - ell_pixel)) >= np.linalg.norm(
                    (self.results['ellipsoid_coeff'][0:2] - pixels_arm1[-1])):
                del pixels_arm1[-1]
            pixels_arm1.append(ell_pixel)
            length_arm1 = np.sum([np.linalg.norm(pixels_arm1[i] - pixels_arm1[i + 1])
                                 for i in range(0, len(pixels_arm1) - 1)]) * self.pixel_size
        else:
            failed = True
            length_arm1 = False

        self.results.update({'pixels_arm1': pixels_arm1,
                             'length_arm1': length_arm1,
                             'failed': failed})

        return self

    def further_analysis(self):
        """ Check the height values along the Wiggins pixels and the slope between individual pixels """

        if self.results['failed'] is False:
            height_pars_arm1 = analysis.wiggins_pixel_height_analysis(self.results['pixels_arm1'],
                                                                      self.mol_filtered, self.pixel_size)
            orientation_pars_arm1 = analysis.dna_orientation(self.results['pixels_arm1'],
                                                             self.mol_bbox)

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
        self.results.update({'nucleosome_volume_core': sum(pixel_heights) * self.pixel_size**2})

        return self
