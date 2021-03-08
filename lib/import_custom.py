"""
Import functions for different ASCII AFM files and detection of the molecules in the image
"""

from tkinter.filedialog import askopenfilename
import numpy as np


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
    img_meta_data = {'file_name': file_name,
                     'file_path': file_path,
                     'x_pixels': x_pixels,
                     'x_length': x_length,
                     'y_pixels': y_pixels,
                     'pixel_size': x_length/x_pixels}

    return np.asarray(img), img_meta_data
