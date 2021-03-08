"""
Contains a class and the required functions to categorize the type of an arbitrary molecule found in
a raw AFM image by using the import_custom and find_molecules functions.
"""

import copy
import numpy as np

from skimage import morphology
from skimage import measure
from skimage import filters
from skimage import segmentation
from skimage import util
import skimage
import cv2

import molecule_categorization as cat
import analysis_functions as analysis
import plot_functions as plotting


class AFMMolecule:

    def __init__(self, mol, img_meta_data, anal_pars, manual_detection=False):
        self.mol_original = mol['mol_original']
        self.mol_filtered = mol['mol_filtered']
        self.img_meta_data = img_meta_data
        self.anal_pars = anal_pars
        self.mol_pars = {'mol_bbox': mol['mol_bbox']}

        # Categorize molecules
        self.mol_props()
        self.categorize_molecules(manual_detection)

    def categorize_molecules(self, manual_detection):
        """

        Returns:

        """
        # Skeleton parameters
        self.mol_pars.update(analysis.skel_pars(self.mol_pars['mol_skel']))
        mol_pars = self.mol_pars
        anal_pars = self.anal_pars
        expected_pixels = anal_pars['dna_length_bp'] * 0.33 / self.img_meta_data['pixel_size']

        # Undersized skeleton
        if mol_pars['skel_pixels_number'] <= expected_pixels / 3:
            mol_pars['type'] = 'Trash'
            mol_pars['reason'] = 'Skeleton too small'

        # Molecules with a too large high spot (and only one spot - otherwise potentially connected good molecule)
        elif mol_pars['max_area_over_height'] > anal_pars['nuc_max_area'] \
                and mol_pars['regions_over_nuc_min_area'] == 1 and mol_pars['skel_pixels_number'] < 1.3*expected_pixels:
            mol_pars['type'] = 'Trash'
            mol_pars['reason'] = 'High area too large'

        # Two large blobs for with too small skeleton
        elif mol_pars['regions_over_nuc_min_area'] >= 2 and mol_pars['skel_pixels_number'] < expected_pixels:
            mol_pars['type'] = 'Trash'
            mol_pars['reason'] = 'Two high areas with small skeleton'

        # Oversized skeleton - potentially connected good molecules
        elif mol_pars['skel_pixels_number'] > expected_pixels * 1.5:
            mol_pars['type'] = 'Potential'
            mol_pars['reason'] = 'Large skeleton'

        # Bare DNA
        elif mol_pars['skel_eps_number'] == 2 and self.mol_pars['skel_bps_number'] == 0 and \
                mol_pars['regions_over_nuc_min_area'] == 0:
            mol_pars['type'] = 'Bare DNA'

        # Bare DNA with branch
        elif mol_pars['skel_eps_number'] in [2, 3] and self.mol_pars['skel_bps_number'] == 1 and \
                mol_pars['regions_over_nuc_min_area'] == 0:
            mol_pars['type'] = 'Potential'
            mol_pars['reason'] = '2 EPS 1 BP no high area'

        # Nucleosomes with three endpoints
        elif mol_pars['skel_eps_number'] == 3 and mol_pars['skel_bps_number'] == 3 and \
                mol_pars['max_area_over_height'] >= anal_pars['nuc_min_area'] and manual_detection is True:

            # Manually remove pixels in the nucleosome for better skeletonization and update skel pars
            mol_pars['mol_skel'] = cat.manual_pixel_removal(mol_pars['mol_skel'])
            mol_pars.update(analysis.skel_pars(mol_pars['mol_skel']))

            if mol_pars['skel_eps_number'] == 2 and mol_pars['skel_bps_number'] == 2:
                mol_pars['type'] = 'Nucleosome'
            else:
                mol_pars['type'] = 'Trash'
                mol_pars['reason'] = 'Test for three endpoint nuc. failed'

        # Nucleosomes two endpoints (normal)
        elif mol_pars['skel_eps_number'] == 2 and mol_pars['skel_bps_number'] == 2 and \
                mol_pars['max_area_over_height'] >= anal_pars['nuc_min_area']:
            mol_pars['type'] = 'Nucleosome'

        elif mol_pars['skel_eps_number'] == 2 and mol_pars['skel_bps_number'] == 1 and \
                mol_pars['max_area_over_height'] >= anal_pars['nuc_min_area']:
            mol_pars['type'] = 'Potential'
            mol_pars['reason'] = '2 EPS 1 BP and high area'

        # Endbound nucleosomes (one endpoint)
        elif mol_pars['skel_eps_number'] == 1 and mol_pars['skel_bps_number'] == 1 and \
                mol_pars['max_area_over_height'] >= anal_pars['nuc_min_area'] and manual_detection is True and \
                mol_pars['skel_pixels_number'] > expected_pixels * 0.6:
            mol_pars['type'] = 'Nucleosome endbound'

        # May be DNA where one end touches the strand itself
        elif mol_pars['skel_eps_number'] == 1 and mol_pars['skel_bps_number'] == 1 and \
                mol_pars['skel_pixels_number'] > expected_pixels * 0.6:
            mol_pars['type'] = 'Potential'
            mol_pars['reason'] = '1 BP 1 EP and minimum size'

        # Trash with reasons
        elif mol_pars['max_area_over_height'] >= anal_pars['nuc_min_area'] and mol_pars['skel_bps_number'] in [3, 4]:
            mol_pars['type'] = 'Potential'
            mol_pars['reason'] = 'Large enough high area and 3 BPS'
        # elif mol_pars['skel_eps_number'] != 2:
        #     mol_pars['type'] = 'Potential'
        #     mol_pars['reason'] = 'Not 2 endpoints'
        elif mol_pars['skel_bps_number'] >= 5 and mol_pars['skel_pixels_number'] < 1.3 * expected_pixels:
            mol_pars['type'] = 'Trash'
            mol_pars['reason'] = 'At least 5 BP and too small skeleton'
        elif mol_pars['max_area_over_height'] < anal_pars['nuc_min_area']:
            mol_pars['type'] = 'Trash'
            mol_pars['reason'] = 'Largest height area too small for a nucleosome'
        else:
            mol_pars['type'] = 'Trash'
            mol_pars['reason'] = 'Undefined'

        self.mol_pars.update(mol_pars)

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

        # max_area_over_height - calculate the largest area of connected pixels with a value over self.nuc_min_height
        if np.amax(mol_filtered) > self.anal_pars['nuc_min_height']:
            mol_over_height = copy.deepcopy(mol_filtered)
            mol_over_height[mol_over_height < self.anal_pars['nuc_min_height']] = 0
            mol_over_height[mol_over_height > self.anal_pars['nuc_min_height']] = 1
            img_labelled = morphology.label(mol_over_height, connectivity=2)
            max_area_over_height = max(region.area for region in measure.regionprops(img_labelled) if region.area)
        else:
            max_area_over_height = 0
        self.mol_pars['max_area_over_height'] = max_area_over_height

        # mol_skel - skeletonization of the molecule with values higher than self.nuc_min_height set to 0
        mol_bw = copy.deepcopy(self.mol_filtered)
        mol_bw[mol_bw > 0] = 1
        mol_bw[self.mol_original > self.anal_pars['nuc_min_height']] = 0

        # Set pixels with height values higher than the nuc_min_height but an area of less than nuc_min_size to 1
        regions_over_nuc_min_area = 0
        if np.amax(mol_filtered) > self.anal_pars['nuc_min_height']:
            for region in measure.regionprops(img_labelled):
                if region.area > self.anal_pars['nuc_min_area']:
                    regions_over_nuc_min_area += 1
                if region.area < self.anal_pars['nuc_min_area']:
                    for r, c in region.coords:
                        mol_bw[r, c] = self.anal_pars['nuc_min_height'] - 0.01
        self.mol_pars['regions_over_nuc_min_area'] = regions_over_nuc_min_area

        mol_skel = skimage.img_as_float(morphology.skeletonize(mol_bw))
        self.mol_pars['mol_skel'] = mol_skel

        return


def find_molecules(img_original, img_meta_data, manual=False, mol_bbox_prev=None, mol_skel=None,
                   back_thresh=0.08, mol_min_area=300, nuc_min_area=12, nuc_min_height=1.00, nuc_max_area=200,
                   dna_length_bp=486):
    """
    Applies the first filtering steps that are required for every AFM image
    1. Gaussian filter with sigma = 1
    2. Background removal according to 'background_1'
    3. Gaussian filter with sigma = 1
    4. Background removal according to 'background_2'
    5. All molecules that touch the image border are removed
    6. Small molecule filter is applied according to 'min_area'
    7. If 'manual' is set to True, one can manually set pixels to zero
    8. Each molecule and its boundary region are stored in a list
       (parts of other molecules in this boundary region are removed)

    Input:
        img_original - 2D numpy array
            Greyscale AFM image that was generated when importing the ascii file
        x_pixels - int
            Amount of pixels in x direction
        y_pixels - int
            Amount of pixels in y direction
        background_1 - float
            Flat value. All pixels below this value are set to 0 in order to
            remove the background.
        background_2 - float (Preset 0.25)
            Flat value. All pixels below this value are set to 0 in order to
            remove the background.
        min_area - int
            Flat value. All molecule that have less pixels are removed
        manual - bool
            Decision whether one want to manually set pixels to zero before connected structures are defined
            as molecules -> helps separating structures that are close together.

    Output:
        img_filtered - array
            Filtered image with height values
        molecules - list
            Each list entry contains one molecule found in the image. Each entry contains
            a close up shot of the filtered image and the original image and the bounds
            of the molecule. Thus the list has as many entries as there are molecules in the
            image after applying the first filtering steps.
    """

    # Set very high and low values (AFM measurement errors) to zero
    img_original[img_original < -0.5] = 0

    img_gaussian = filters.gaussian(img_original, sigma=1.0)

    # Remove background
    img_no_background = img_gaussian[:]
    img_no_background[img_no_background < back_thresh] = 0

    # Apply another Gaussian filter with background removal to separate molecules better
    img_gaussian_final = filters.gaussian(img_no_background, sigma=1.0)
    img_gaussian_final[img_gaussian_final < 2 * back_thresh] = 0

    # Remove molecules touching the border
    img_bw = img_gaussian_final[:]
    img_bw[img_bw > 0] = 1
    img_no_border = segmentation.clear_border(img_bw)

    # Filter small molecules
    labels = morphology.label(img_no_border, connectivity=2)
    img_filtered_bw = morphology.remove_small_objects(labels, mol_min_area)

    # Copy original image and set filtered parts to zero
    img_filtered = copy.deepcopy(img_original)
    img_filtered[img_filtered_bw == 0] = 0
    img_filtered[img_filtered < 0] = 0

    if manual is True:
        # Remove molecules that sneaked into the image (happened under certain conditions even if not connected)
        img_labelled_manual = copy.deepcopy(img_filtered)
        img_labelled_manual[img_labelled_manual != 0] = 1
        img_labelled_manual = morphology.label(img_labelled_manual, connectivity=2)
        if np.amax(img_labelled_manual != 1):
            img_labelled_manual[img_labelled_manual != 1] = 0
            img_filtered[img_labelled_manual == 0] = 0

        # Remove pixels manually
        img_filtered = manual_pixel_removal(img_filtered, mol_skel)
        img_filtered_bw[img_filtered == 0] = 0

    # Extract the individual molecules
    molecules = []
    img_labelled = morphology.label(img_filtered_bw, connectivity=2)
    for region in measure.regionprops(img_labelled):
        curr_molecule = []
        minr, minc, maxr, maxc = region.bbox

        # Remove parts of other molecules that are in the same box
        mol_filtered_box = copy.deepcopy(img_filtered[minr:maxr, minc:maxc])
        mol_labelled_box = img_labelled[minr:maxr, minc:maxc]
        mol_filtered_box[mol_labelled_box != region.label] = 0

        # Pad each molecule with a 10 pixel border of the original image or 0 if at the edges of the orig image
        if np.amin([minr, minc]) < 10 or maxr > (img_meta_data['y_pixels']-10) or maxc > (img_meta_data['x_pixels']-10):
            curr_molecule.append(util.pad(img_original[minr:maxr, minc:maxc], pad_width=10, mode='constant'))
        else:
            curr_molecule.append(img_original[minr-10:maxr+10, minc-10:maxc+10])
        # mol_filtered_box = copy.deepcopy(img_filtered[minr:maxr, minc:maxc])
        curr_molecule.append(util.pad(mol_filtered_box, pad_width=10, mode='constant'))
        if manual is False or mol_bbox_prev is None:
            curr_molecule.append([minr, minc, maxr, maxc])
        if manual is True and mol_bbox_prev is not None:
            minr, minc, maxr, maxc = np.array([minr, minc, maxr, maxc]) \
                                     + np.array([mol_bbox_prev[0], mol_bbox_prev[1],
                                                 mol_bbox_prev[0], mol_bbox_prev[1]])\
                                     - np.array([10, 10, 10, 10])
            curr_molecule.append([minr, minc, maxr, maxc])
        molecules.append({'mol_original': curr_molecule[0],
                          'mol_filtered': curr_molecule[1],
                          'mol_bbox': curr_molecule[2]})

    return img_filtered, molecules


def manual_pixel_removal(img_filtered, mol_skel=None):
    """
    Function that allows to manually set pixels to zero

    Input:
        img_filtered - array
            the image for which certain pixels are supposed to be set to zero

    Output:
        img_filtered_new - array
            the new version of the image after editing
    """

    # Import my own colormap to make it look nicer
    my_colormap = plotting.create_custom_colormap()

    # Convert image to uint8 and apply colormap (cv_apply_custom_colormap only takes uint8)
    img_uint8 = copy.deepcopy(img_filtered)
    # Scale with 2.5 to have more contrast in the colormap
    if np.amax(img_uint8) >= 2.5:
        img_uint8 = img_uint8 / np.amax(img_uint8) * 254.9  # scaling to 255 will make the max value equal to 0
    else:
        img_uint8 = img_uint8 / 2.5 * 255
    img_uint8 = plotting.cv_apply_custom_colormap(img_uint8.astype('uint8'), cmap=my_colormap)

    if mol_skel is not None:
        # Mark the skeleton pixels in the color image
        for r, c in zip(*np.where(mol_skel != 0)):
            img_uint8[r, c] = np.array([42, 49, 50])

    mode = True

    # mouse callback function
    def draw(event, x, y, flags, param):

        global ix, iy, drawing
        drawing = False
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing is True:
                if mode is True:
                    cv2.rectangle(img_uint8, (ix, iy), (x, y), (0, 110, 0), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if mode is True:
                cv2.rectangle(img_uint8, (ix, iy), (x, y), (0, 110, 0), -1)

    # Create an image, a window and bind the function to window
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    window_size = (img_filtered.shape / np.amax(img_filtered.shape) * 600).astype('int')     # Rescaled to 600
    cv2.resizeWindow('image', window_size[1], window_size[0])
    cv2.setMouseCallback('image', draw)

    while 1:
        cv2.imshow('image', img_uint8)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            mode = False
        elif k == ord('s'):
            mode = True
        elif k == 27:
            break
    cv2.destroyAllWindows()

    # In my drawing function only the R values are set to zero when clicking on them -> Set all values in the
    # filtered image to zero where the R values are 0
    img_filtered_new = copy.deepcopy(img_filtered)
    img_filtered_new[img_uint8[:, :, 0] == 0] = 0

    return img_filtered_new


def separate_molecules(afm_molecule):
    """
    Function to manually separate close molecules: molecules that are not nicely defined structures like bare DNA
    nucleosomes etc. are stored as 'trash' and can be edited manually. In some cases a manual separation of the
    molecules by setting some pixels that connect two individual structures helps to detect more defined structure.
    In this function the manual pixel removal is called for each trash molecule and the separated molecules are then re-
    analyzed.

    Input:
        afm_molecule - molecule
            Instance of the AFMMolecule class (defined in molecule_categorization.py)

    Output:
        molecules - list of list of arrays
            Returns the separated structures as individual arrays which can then be analyzed again by creating a new
            AFMmolecule instance for each separated structure
    """

    mol_original = afm_molecule.mol_original

    # Use the find molecules function with manual pixel removal set to True to separate the connected structures
    mol_filtered, molecules = find_molecules(mol_original, afm_molecule.img_meta_data, manual=True,
                                             mol_bbox_prev=afm_molecule.mol_pars['mol_bbox'],
                                             mol_skel=afm_molecule.mol_pars['mol_skel'],
                                             **afm_molecule.anal_pars)

    return mol_filtered, molecules


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

    mol_trash = [mol for mol in afm_molecules if mol.mol_pars['type'] == 'Potential']

    for mol in mol_trash:
        # give chance to separate the trashed molecules manually
        mol_filtered, separate_mols = cat.separate_molecules(mol)

        # call new class instances for the separated molecules
        for separate_mol in separate_mols:
            afm_molecules.append(AFMMolecule(separate_mol, mol.img_meta_data, mol.anal_pars, manual_detection=True))

        # Delete the previously unseparated AFM molecules
        delete_id = id(mol)
        for item in afm_molecules:
            if id(item) == delete_id:
                afm_molecules.remove(item)

    return afm_molecules
