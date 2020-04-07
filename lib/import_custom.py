"""
Import functions for different ASCII AFM files and detection of the molecules in the image
"""

from tkinter.filedialog import askopenfilename
import numpy as np
import copy
import cv2

from skimage import filters
from skimage import segmentation
from skimage import morphology
from skimage import measure
from skimage import util

import plot_functions as plotting


def import_ascii(file_path=None):
    """
    Function to import an ASCII file with AFM surface height measurements and some
    headerlines. Afterwards converts the floats to uint-8 format

    Input:
        file_path - String
            If no file_path is given to the function, a window opens to select the
            desired file manually

    Output:
        img_original - array
            Original image as array with height values as float
        file_name - string
            Name of the file that was imported
        x_pixels - int
            Number of pixels in x-direction
        y_pixels - int
            Number of pixels in y-direction
        x_length - float
            Length of the image in x-direction -> can be used to calculate the resolution (x_length/x_pixels)
    """

    if file_path is None:
        file_path = askopenfilename(title='Select AFM image ASCII file', filetypes=(("ASCII files", "*.asc"),))
    file_name = file_path.split('/')[-1]
    f = open(file_path, 'r')

    # Read each line, discriminate between header line and height value line by checking if the
    # content of the first entry of the line is a digit or not
    img = []
    for line in f:
        try:
            first_entry = line.strip().split()[0][-5:]
            meas_par = line.split()[1]

            if first_entry.isdigit() or first_entry[-5:-3] == 'e-' or first_entry[-4:-2] == 'e-':
                line = line.strip()
                floats = [float(x) for x in line.split()]
                img.append(np.asarray(floats))

            # Find the required measurement information
            elif meas_par == 'x-pixels':
                x_pixels = float(line.split()[-1])

            # Find the required measurement information
            elif meas_par == 'y-pixels':
                y_pixels = float(line.split()[-1])

            elif meas_par == 'x-length':
                x_length = float(line.split()[-1])

        except IndexError:
            pass

    if 'x_pixels' not in locals():
        x_pixels = 'unknown'
        print('The amount of x-pixels was not found in the header')

    if 'y_pixels' not in locals():
        y_pixels = 'unknown'
        print('The amount of y-pixels was not found in the header')

    if 'x_length' not in locals():
        x_length = 'unknown'
        print('The size of the image was not found in the header')

    img = np.asarray(img)

    return np.asarray(img), file_name, file_path, y_pixels, x_pixels, x_length


def find_molecules(img_original, x_pixels, y_pixels, background_1=0.15, background_2=0.25, min_area=300, manual=False,
                   afm_type='nanoscope', mol_bbox_prev=None, mol_skel=None):
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
    # img_original[img_original > 6.0] = 0

    if afm_type == 'nanoscope':
        img_gaussian = filters.gaussian(img_original, sigma=1.0)

        # Remove background
        img_no_background = img_gaussian[:]
        img_no_background[img_no_background < background_1] = 0

        # Apply another Gaussian filter with background removal to separate molecules better
        img_gaussian_final = filters.gaussian(img_no_background, sigma=1.0)
        img_gaussian_final[img_gaussian_final < background_2] = 0

    elif afm_type == 'jpk':

        img_no_background = copy.deepcopy(img_original)
        img_no_background[img_no_background < background_2] = 0
        img_gaussian = filters.gaussian(img_no_background, sigma=1.0)
        img_gaussian[img_gaussian < background_2] = 0
        img_gaussian_final = filters.gaussian(img_gaussian, sigma=1.0)
        img_gaussian_final[img_gaussian_final < background_1] = 0

    # Remove molecules touching the border
    img_bw = img_gaussian_final[:]
    img_bw[img_bw > 0] = 1
    img_no_border = segmentation.clear_border(img_bw)

    # Filter small molecules
    labels = morphology.label(img_no_border, connectivity=2)
    img_filtered_bw = morphology.remove_small_objects(labels, min_area)

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
        if np.amin([minr, minc]) < 10 or maxr > (x_pixels-10) or maxc > (y_pixels-10):
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
        molecules.append(curr_molecule)

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
    back_1 = afm_molecule.background_1
    back_2 = afm_molecule.background_2
    min_area = afm_molecule.min_area

    # Use the find molecules function with manual pixel removal set to True to separate the connected structures
    mol_filtered, molecules = find_molecules(mol_original, x_pixels=np.shape(mol_original)[0],
                                             y_pixels=np.shape(mol_original)[1],
                                             background_1=back_1, background_2=back_2,
                                             min_area=min_area, manual=True, mol_bbox_prev=afm_molecule.mol_bbox,
                                             mol_skel=afm_molecule.mol_pars['mol_skel'])

    # Set boundary box to the original image size
    # for i in range(len(molecules)):
    #    molecules[i][2] = afm_molecule.mol_bbox

    return mol_filtered, molecules
