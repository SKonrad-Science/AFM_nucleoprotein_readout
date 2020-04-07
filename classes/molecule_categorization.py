"""
Contains a class and the required functions categorize the type of an arbitrary molecule found in
a raw AFM image by using the import_custom import and find_molecules function.
"""

import copy
import numpy as np

from skimage import morphology
from skimage import measure
import skimage

import import_custom
import analysis_functions as analysis


class AFMMolecule:

    def __init__(self, mol, dna_bp, pixel_size, background_1=0.15, background_2=0.25, min_area=300,
                 nuc_min_height=1.25, nuc_min_area=10, categorize=False, manual_detection=False):
        self.mol_original = mol[0]
        self.mol_filtered = mol[1]
        self.mol_bbox = mol[2]
        self.DNA_bp = dna_bp
        self.pixel_size = pixel_size
        self.background_1 = background_1
        self.background_2 = background_2
        self.min_area = min_area
        self.nuc_min_height = nuc_min_height
        self.nuc_min_size = nuc_min_area
        self.mol_pars = {}

        # Categorize molecules if called
        if categorize is True:
            self.categorize_molecule(manual_detection)

    def categorize_molecule(self, manual_detection):
        """
        Uses the calculated parameters of the class instance to categorize its molecule type. Procedure:
        Categories (detected in the sequence as depicted here):

        Trash - reason 'Skeleton too large'
            Depending on the number of base pairs of DNA imaged a certain size of the skeleton is expected. In this
            calculation it is assumed that each pixel contributes 1.2*pixel_size to the length of the molecule (since a
            skeleton pixel can be vertical or diagonal) and the number of skeleton pixels shouldn't be larger than 1.5
            times than the expected amount on pixels based on this pixel length contribution and the DNA base pairs
        Trash - reason 'Skeleton too small'
            Same as for the too large skeleton just with marking everything that's smaller than 0.5* the expected pixels
        Bare DNA
            Has 2 endpoints and no branchpoints
        Nucleosomes - three endpoints
            These are only detected if the parameter 'manual_detection' is set to True.
            Structures that have three endpoints, less than 12 branchpoints and a max_area_over_height bigger than the
            set nuc_min_size. This is done because sometimes nucleosomes have an additional arm in their nucleosome
            circle skeleton and here one gets the chance to remove this arm manually. After manual removal the skeleton
            parameters are updated and recalculated
        Nucleosomes - two endpoints
            Normal detected nucleosomes: Two endpoints, less than 12 branchpoints and a max_area_over_height bigger than
            the set nuc_min_size.
        Nucleosomes - endbound
            These are only detected if the parameter 'manual_detection' is set to True.
            This is done when reanalyzing the trash since otherwise many nucleosomes that have only one endpoint before
            separating its arms manually are counted as endbound nucleosomes wrongfully. (<= 4 branchpoints and minimum
            amount of high pixels)
        Trash - reason 'Endpoints'
            Wrong number of endpoints
        Trash - reason 'Branchpoints'
            Wrong number of branchpoints
        Trash - reason 'Nucleosome pixels'
            Not enough nucleosomes pixels
        Trash - reason 'undefined'
            Don't know the reason

        Input:
            eb_detection - bool
                Turns on the detection of endbound nucleosomes. (Should be done during reanalysis of the trash)

        Output:
            self
        """
        # Calculate molecule properties
        self.mol_props()

        # Skeleton parameters
        eps_pixels, eps_number, bps_pixels, bps_number, pixels_number = analysis.skel_pars(self.mol_pars['mol_skel'])
        self.mol_pars.update({'skel_eps_pixels': eps_pixels,
                              'skel_eps_number': eps_number,
                              'skel_bps_pixels': bps_pixels,
                              'skel_bps_number': bps_number,
                              'skel_pixels_number': pixels_number})

        # Oversized molecule:
        exp_skel_pixels = (self.DNA_bp*0.34)/(self.pixel_size*1.2)
        if self.mol_pars['skel_pixels_number'] >= 2.0*exp_skel_pixels:
            self.mol_pars['type'] = 'Trash'
            self.mol_pars['reason'] = 'Skeleton too large'
            pass

        # Undersized molecule:
        elif self.mol_pars['skel_pixels_number'] <= 0.5*exp_skel_pixels:
            self.mol_pars['type'] = 'Trash'
            self.mol_pars['reason'] = 'Skeleton too small'

        # Bare DNA: 2 Endpoints, no branchpoint
        elif self.mol_pars['skel_eps_number'] == 2 and self.mol_pars['skel_bps_number'] == 0:
            self.mol_pars['type'] = 'Bare DNA'

        # Nucleosomes with three endpoints
        elif self.mol_pars['skel_eps_number'] == 3 and self.mol_pars['skel_bps_number'] <= 15 and \
                self.mol_pars['max_area_over_height'] >= self.nuc_min_size and manual_detection is True:

            # Manually remove pixels in the nucleosome for better skeletonization
            self.mol_pars['mol_skel'] = import_custom.manual_pixel_removal(self.mol_pars['mol_skel'])

            # Update skeleton parameters
            eps_pixels, eps_number, bps_pixels, bps_number, pixels_number = analysis.skel_pars(
                self.mol_pars['mol_skel'])
            self.mol_pars.update({'skel_eps_pixels': eps_pixels,
                                  'skel_eps_number': eps_number,
                                  'skel_bps_pixels': bps_pixels,
                                  'skel_bps_number': bps_number,
                                  'skel_pixels_number': pixels_number})
            if self.mol_pars['skel_eps_number'] == 2:
                self.mol_pars['type'] = 'Nucleosome'
            else:
                self.mol_pars['type'] = 'Trash'
                self.mol_pars['reason'] = 'Undefined'

        # Nucleosomes two endpoints (normal)
        elif self.mol_pars['skel_eps_number'] == 2 and self.mol_pars['skel_bps_number'] <= 11 and \
                self.mol_pars['max_area_over_height'] >= self.nuc_min_size:
            self.mol_pars['type'] = 'Nucleosome'

        # Endbound nucleosomes (one endpoint)
        elif self.mol_pars['skel_eps_number'] == 1 and self.mol_pars['skel_bps_number'] <= 4 and \
                self.mol_pars['max_area_over_height'] >= self.nuc_min_size and manual_detection is True:
            self.mol_pars['type'] = 'Nucleosome endbound'

        # Trash with reasons
        elif self.mol_pars['skel_eps_number'] != 2:
            self.mol_pars['type'] = 'Trash'
            self.mol_pars['reason'] = 'Endpoints'
        elif self.mol_pars['skel_bps_number'] > 11:
            self.mol_pars['type'] = 'Trash'
            self.mol_pars['reason'] = 'Branchpoints'
        elif self.mol_pars['max_area_over_height'] < self.nuc_min_size:
            self.mol_pars['type'] = 'Trash'
            self.mol_pars['reason'] = 'Nucleosome pixels'
        else:
            self.mol_pars['type'] = 'Trash'
            self.mol_pars['reason'] = 'Undefined'

        # # Pauline Ints detection
        # if self.mol_pars['type'] == 'Trash' and (self.mol_pars['skel_bps_number'] >= 15 or self.mol_pars['max_area_over_height'] >= self.nuc_min_size):
        #     self.mol_pars['type'] = 'Int'

        return

    def mol_props(self):
        """
        Function that is called when initializing a new instance of the class. Several parameters of the molecule are
        calculated here:
        area_over_height
            total amount of pixels that have a value higher than self.nuc_min_height
        max_area_over_height
            amount of pixels of the largest connected area with values over self.nuc_min_height
        mol_skel
            Skeleton of the filtered version of the class instance image. Skeletonization is performed on the
            binarized version the filtered molecule after setting all pixels higher than self.nuc_min_height to zero.
            (this helps creating a circle around the nucleosome and thus facilitates categorization of the molecules)

        Input:
            self

        Output:
            self
        """

        mol_filtered = copy.deepcopy(self.mol_filtered)

        # area_over_height - calculate the number of pixels with values over self.nuc_min_height in the molecule
        # max_area_over_height - calculate the largest area of connected pixels with a value over self.nuc_min_height
        if np.amax(mol_filtered) > self.nuc_min_height:
            mol_over_height = copy.deepcopy(mol_filtered)
            mol_over_height[mol_over_height < self.nuc_min_height] = 0
            mol_over_height[mol_over_height > self.nuc_min_height] = 1
            img_labelled = morphology.label(mol_over_height, connectivity=2)
            max_area_over_height = max(region.area for region in measure.regionprops(img_labelled) if region.area)
        else:
            max_area_over_height = 0
        self.mol_pars['max_area_over_height'] = max_area_over_height

        # mol_skel - skeletonization of the molecule with values higher than self.nuc_min_height set to 0
        mol_bw = copy.deepcopy(self.mol_filtered)
        mol_bw[mol_bw > 0] = 1
        mol_bw[self.mol_original > self.nuc_min_height] = 0

        # Set pixels with height values higher than the nuc_min_height but an area of less than nuc_min_size to 1
        if np.amax(mol_filtered) > self.nuc_min_height:
            for region in measure.regionprops(img_labelled):
                if region.area < self.nuc_min_size:
                    for r, c in region.coords:
                        mol_bw[r, c] = 1

        mol_skel = skimage.img_as_float(morphology.skeletonize(mol_bw))
        self.mol_pars['mol_skel'] = mol_skel

        return


def manual_trash_analysis(afm_molecules):
    """
    Function that allows to manually separate all afm_molecules that are tagged as 'trashed' based on their structure
    parameters. The trashed molecules are then reanalyzed after manually separating them or kept the same in case
    there is nothing to separate and added to the afm_molecules list. This way, more biological structures can be
    detected in general and lead to better statistics per image and in total.

    Input:
        afm_molecules - list of molecules
            Give a list of AFMMolecule class instances to the manual trash analysis function.

    Output:
        afm_molecules - list of molecules
            Updated list of the input with hopefully less 'trash' molecules
    """

    mol_trash = [mol for mol in afm_molecules if mol.mol_pars['type'] == 'Trash']

    for mol in mol_trash:
        # give chance to separate the trashed molecules manually
        mol_filtered, separate_mols = import_custom.separate_molecules(mol)

        # call new class instances for the separated molecules
        for separate_mol in separate_mols:
            afm_molecules.append(AFMMolecule(separate_mol, mol.DNA_bp, mol.pixel_size,
                                             background_1=mol.background_1,
                                             background_2=mol.background_2,
                                             min_area=mol.min_area,
                                             nuc_min_height=mol.nuc_min_height, nuc_min_area=mol.nuc_min_size,
                                             categorize=True, manual_detection=True))

        # Delete the previously unseparated AFM molecules
        delete_id = id(mol)
        for item in afm_molecules:
            if id(item) == delete_id:
                afm_molecules.remove(item)

    return afm_molecules
