""" Functions to plot the image analysis results """

import copy
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import cv2

import analysis_functions as analysis


def plot_overview_image(img_filtered,
                        file_name,
                        results_final,
                        analyze_bare_DNA=False,
                        analyze_nucleosomes=False,
                        analyze_nucleosomes_eb=False,
                        analyze_ints=False):
    """ Plot over image with all analyzed molecules marked """

    my_colormap = create_custom_colormap()
    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.add_subplot(1, 1, 1)
    plt.imshow(img_filtered, interpolation='None', cmap=my_colormap)

    # Plot the analyzed bare DNA
    if analyze_bare_DNA is True:
        analyzed_bare_DNA = results_final['analyzed_bare_DNA']

        dna_succeeded = [mol for mol in analyzed_bare_DNA if mol.results['failed'] is False]
        dna_failed = [mol for mol in analyzed_bare_DNA if mol.results['failed'] is True]
        for mol in dna_succeeded:
            if mol.results['length_fwd'] is not False:
                wiggins_pixels = copy.deepcopy(mol.results['wiggins_pixels_fwd'])
            else:
                wiggins_pixels = copy.deepcopy(mol.results['wiggins_pixels_bwd'])
            pixels_img = np.asarray([np.array([r - 10 + mol.mol_bbox[0], c - 10 + mol.mol_bbox[1]])
                                     for r, c in wiggins_pixels])
            plt.plot(pixels_img[:, 1], pixels_img[:, 0], color='#FB6542', linewidth=1.5)

        for mol in dna_failed:
            wiggins_pixels = copy.deepcopy(mol.results['wiggins_pixels_fwd'])
            pixels_img = np.asarray([np.array([r - 10 + mol.mol_bbox[0], c - 10 + mol.mol_bbox[1]])
                                     for r, c in wiggins_pixels])
            plt.plot(pixels_img[:, 1], pixels_img[:, 0], color='#FF420E')
            wiggins_pixels = copy.deepcopy(mol.results['wiggins_pixels_bwd'])
            pixels_img = np.asarray([np.array([r - 10 + mol.mol_bbox[0], c - 10 + mol.mol_bbox[1]])
                                     for r, c in wiggins_pixels])
            plt.plot(pixels_img[:, 1], pixels_img[:, 0], color='#FF420E')

    # Plot all nucleosomes
    if analyze_nucleosomes is True:
        analyzed_nucleosomes = results_final['analyzed_nucleosomes']

        nuc_succeeded = [mol for mol in analyzed_nucleosomes if mol.results['failed'] is False]
        failed = [mol for mol in analyzed_nucleosomes if mol.results['failed'] is True]
        for mol in nuc_succeeded:

            # Plot the Wiggins trace
            pixels_arm1 = copy.deepcopy(mol.results['pixels_arm1'])
            pixels_img = np.asarray([np.array([r - 10 + mol.mol_bbox[0], c - 10 + mol.mol_bbox[1]])
                                     for r, c in pixels_arm1])
            plt.plot(pixels_img[:, 1], pixels_img[:, 0], color='#FFBB00', linewidth=1.5)
            pixels_arm2 = copy.deepcopy(mol.results['pixels_arm2'])
            pixels_img = np.asarray([np.array([r - 10 + mol.mol_bbox[0], c - 10 + mol.mol_bbox[1]])
                                     for r, c in pixels_arm2])
            plt.plot(pixels_img[:, 1], pixels_img[:, 0], color='#FFBB00', linewidth=1.5)

            # Plot the angle
            ellipsoid_coeff = mol.results['ellipsoid_coeff']
            pixel_center = ellipsoid_coeff[0:2]
            pixels_angle = np.array([[pixels_arm1[-1][0] - 10 + mol.mol_bbox[0],
                                      pixels_arm1[-1][1] - 10 + mol.mol_bbox[1]],
                                     [pixel_center[0] - 10 + mol.mol_bbox[0],
                                      pixel_center[1] - 10 + mol.mol_bbox[1]],
                                     [pixels_arm2[-1][0] - 10 + mol.mol_bbox[0],
                                      pixels_arm2[-1][1] - 10 + mol.mol_bbox[1]]])
            plt.plot(pixels_angle[:, 1], pixels_angle[:, 0], color='#FFBB00', linewidth=1.5)

            # Plot the nucleosome ellipses
            pixels_ellipse = ellipse_points(x0=ellipsoid_coeff[0], y0=ellipsoid_coeff[1],
                                            a=ellipsoid_coeff[2], b=ellipsoid_coeff[3],
                                            phi=ellipsoid_coeff[5], z_h=0)
            plt.plot(pixels_ellipse[:, 1] - 10 + mol.mol_bbox[1],
                     pixels_ellipse[:, 0] - 10 + mol.mol_bbox[0], color='#FFBB00', linewidth=1.5)

            pixels_ellipse = ellipse_points(x0=ellipsoid_coeff[0], y0=ellipsoid_coeff[1],
                                            a=ellipsoid_coeff[2], b=ellipsoid_coeff[3],
                                            phi=ellipsoid_coeff[5], z_h=0.6)
            plt.plot(pixels_ellipse[:, 1] - 10 + mol.mol_bbox[1],
                     pixels_ellipse[:, 0] - 10 + mol.mol_bbox[0], color='#FFBB00', linewidth=1.5)

            # plt.scatter(mol.results['ellipsoid_coeff'][1] - 10 + mol.mol_bbox[1],
            #             mol.results['ellipsoid_coeff'][0] - 10 + mol.mol_bbox[0])

        for mol in failed:
            # Check that the ellipsoid fit worked, otherwise don't try plotting since arms weren't traced
            if 'pixels_arm1' and 'pixels_arm2' in mol.results:
                # Plot the Wiggins trace
                pixels_arm1 = copy.deepcopy(mol.results['pixels_arm1'])
                if pixels_arm1 is not False:
                    pixels_img = np.asarray([np.array([r - 10 + mol.mol_bbox[0], c - 10 + mol.mol_bbox[1]])
                                             for r, c in pixels_arm1])
                    plt.plot(pixels_img[:, 1], pixels_img[:, 0], color='#FF420E')

                pixels_arm2 = copy.deepcopy(mol.results['pixels_arm2'])
                if pixels_arm2 is not False:
                    pixels_img = np.asarray([np.array([r - 10 + mol.mol_bbox[0], c - 10 + mol.mol_bbox[1]])
                                             for r, c in pixels_arm2])
                    plt.plot(pixels_img[:, 1], pixels_img[:, 0], color='#FF420E')

    # Plot all endbound nucleosomes
    if analyze_nucleosomes_eb is True:
        analyzed_nucleosomes_eb = results_final['analyzed_nucleosomes_eb']

        nuc_succeeded = [mol for mol in analyzed_nucleosomes_eb if mol.results['failed'] is False]
        failed = [mol for mol in analyzed_nucleosomes_eb if mol.results['failed'] is True]

        for mol in nuc_succeeded:

            # Plot the Wiggins trace
            pixels_arm1 = copy.deepcopy(mol.results['pixels_arm1'])
            pixels_img = np.asarray([np.array([r - 10 + mol.mol_bbox[0], c - 10 + mol.mol_bbox[1]])
                                     for r, c in pixels_arm1])
            plt.plot(pixels_img[:, 1], pixels_img[:, 0], color='#F98866')

            # Plot the nucleosome ellipses
            ellipsoid_coeff = mol.results['ellipsoid_coeff']
            pixels_ellipse = ellipse_points(x0=ellipsoid_coeff[0], y0=ellipsoid_coeff[1],
                                            a=ellipsoid_coeff[2], b=ellipsoid_coeff[3],
                                            phi=ellipsoid_coeff[5], z_h=0)
            plt.plot(pixels_ellipse[:, 1] - 10 + mol.mol_bbox[1],
                     pixels_ellipse[:, 0] - 10 + mol.mol_bbox[0], color='#F98866')

            pixels_ellipse = ellipse_points(x0=ellipsoid_coeff[0], y0=ellipsoid_coeff[1],
                                            a=ellipsoid_coeff[2], b=ellipsoid_coeff[3],
                                            phi=ellipsoid_coeff[5], z_h=0.6)
            plt.plot(pixels_ellipse[:, 1] - 10 + mol.mol_bbox[1],
                     pixels_ellipse[:, 0] - 10 + mol.mol_bbox[0], color='#F98866')

    plt.show()
    fig.savefig(file_name + '_overview.png', bbox_inches='tight')

    return


def plot_save_close_ups(results_final,
                        file_name,
                        analyze_bare_DNA=False,
                        analyze_nucleosomes=False,
                        analyze_nucleosomes_eb=False,
                        analyze_ints=False,
                        plot_trash=False):

    my_colormap = create_custom_colormap()
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    # Plot the individual analyzed bare DNA strands
    if analyze_bare_DNA is True:
        analyzed_bare_DNA = results_final['analyzed_bare_DNA']
        if not os.path.exists(file_name + '/bare_DNA'):
            os.makedirs(file_name + '/bare_DNA')
            os.makedirs(file_name + '/bare_DNA/failed')
        count = 0
        count_failed = 0

        for mol in analyzed_bare_DNA:

            fig = plt.figure()
            mol_filtered_rgb = convert_to_rgb(mol.mol_filtered, my_colormap)

            if mol.results['length_fwd'] is not False:
                wiggins_pixels = copy.deepcopy(mol.results['wiggins_pixels_fwd'])
            elif mol.results['length_bwd'] is not False:
                wiggins_pixels = copy.deepcopy(mol.results['wiggins_pixels_bwd'])

            plt.imshow(mol_filtered_rgb, interpolation='None')

            if mol.results['failed'] is False:
                plot_wiggins_pixels_close_up(wiggins_pixels)
                name = file_name + '/bare_DNA/' + repr(count) + '.png'
                plt.text(2, 6, 'Length: ' + repr(np.round(mol.results['length_avg'], decimals=1))
                         + ' nm', color='white', fontsize=12)
                count += 1
            else:
                name = file_name + '/bare_DNA/failed/' + repr(count_failed) + '_failed.png'
                if mol.results['failed_reason'] == 'Discarded manually':
                    plot_wiggins_pixels_close_up(wiggins_pixels)
                    plt.text(2, 6, 'Length: ' + repr(np.round(mol.results['length_avg'], decimals=1))
                             + ' nm', color='white', fontsize=12)
                plt.text(2, 3, 'Discard reason: ' + mol.results['failed_reason'], color='white', fontsize=12)
                count_failed += 1
            fig.savefig(name, bbox_inches='tight')

            plt.close('all')

    # Plot the nucleosomes
    if analyze_nucleosomes is True:
        analyzed_nucleosomes = results_final['analyzed_nucleosomes']
        if not os.path.exists(file_name + '/nucleosomes'):
            os.makedirs(file_name + '/nucleosomes')
            os.makedirs(file_name + '/nucleosomes/failed')
        count = 0
        count_failed = 0

        for mol in analyzed_nucleosomes:
            # Check that the ellipsoid fit worked, otherwise don't try plotting since arms weren't traced
            if 'pixels_arm1' and 'pixels_arm2' in mol.results:
                fig = plt.figure()
                plt.axis('off')
                mol_filtered_rgb = convert_to_rgb(mol.mol_filtered, my_colormap)

                plot_wiggins_pixels_close_up(mol.results['pixels_arm1'])
                plot_wiggins_pixels_close_up(mol.results['pixels_arm2'])
                plot_ellipse_close_up(mol.results['ellipsoid_coeff'])
                plt.scatter(mol.results['ellipsoid_coeff'][1], mol.results['ellipsoid_coeff'][0], color='#2A3132')
                plt.imshow(mol_filtered_rgb, interpolation='None')

                if mol.results['failed'] is False:
                    name = file_name + '/nucleosomes/' + repr(count) + '.png'
                    plt.text(2, 3, 'Arm sum: ' + repr(np.round(mol.results['length_sum'], decimals=1))
                             + ' nm', color='white', fontsize=10)
                    plt.text(2, 6, 'Angle: ' + repr(np.round(mol.results['angle_arms'], decimals=1))
                             + ' Degree', color='white', fontsize=10)
                    plt.text(2, 9, 'Volume: ' + repr(np.round(mol.results['nucleosome_volume'], decimals=1))
                             + ' nm^3', color='white', fontsize=10)
                    count += 1
                else:
                    name = file_name + '/nucleosomes/failed/' + repr(count_failed) + '_failed.png'
                    plt.text(2, 3, 'Discard reason: ' + mol.results['failed_reason'], color='white', fontsize=12)
                    if mol.results['failed_reason'] == 'Discarded manually':
                        plt.text(2, 6, 'Arm sum: ' + repr(np.round(mol.results['length_sum'], decimals=1))
                                 + ' nm', color='white', fontsize=10)
                        plt.text(2, 9, 'Angle: ' + repr(np.round(mol.results['angle_arms'], decimals=1))
                                 + ' Degree', color='white', fontsize=10)
                        plt.text(2, 12, 'Volume: ' + repr(np.round(mol.results['nucleosome_volume'], decimals=1))
                                 + ' nm^3', color='white', fontsize=10)
                    count_failed += 1
                fig.savefig(name, bbox_inches='tight')
            else:
                count += 1

            plt.close()

    if analyze_nucleosomes_eb is True:
        analyzed_nucleosomes_eb = results_final['analyzed_nucleosomes_eb']
        if not os.path.exists(file_name + '/nucleosomes_eb'):
            os.makedirs(file_name + '/nucleosomes_eb')
            os.makedirs(file_name + '/nucleosomes_eb/failed')
        count = 0
        count_failed = 0

        for mol in analyzed_nucleosomes_eb:

            fig = plt.figure()
            plt.axis('off')
            mol_filtered_rgb = convert_to_rgb(mol.mol_filtered, my_colormap)
            plt.imshow(mol_filtered_rgb, interpolation='None')

            if mol.results['failed'] is False:
                plot_wiggins_pixels_close_up(mol.results['pixels_arm1'])
                plot_ellipse_close_up(mol.results['ellipsoid_coeff'])
                plt.scatter(mol.results['ellipsoid_coeff'][1], mol.results['ellipsoid_coeff'][0], color='#2A3132')

                name = file_name + '/nucleosomes_eb/' + repr(count) + '.png'
                plt.text(2, 3, 'Arm: ' + repr(np.round(mol.results['length_arm1'], decimals=1))
                         + ' nm', color='white', fontsize=10)
                plt.text(2, 6, 'Volume: ' + repr(np.round(mol.results['nucleosome_volume'], decimals=1))
                         + ' nm^3', color='white', fontsize=10)
                count += 1
            else:
                name = file_name + '/nucleosomes_eb/failed/' + repr(count_failed) + '_failed.png'
                plt.text(2, 3, 'Discard reason: ' + mol.results['failed_reason'], color='white', fontsize=12)
                if mol.results['failed_reason'] == 'Discarded manually':
                    plot_wiggins_pixels_close_up(mol.results['pixels_arm1'])
                    plot_ellipse_close_up(mol.results['ellipsoid_coeff'])
                    plt.scatter(mol.results['ellipsoid_coeff'][1], mol.results['ellipsoid_coeff'][0], color='#2A3132')
                    plt.text(2, 6, 'Arm: ' + repr(np.round(mol.results['length_arm1'], decimals=1))
                             + ' nm', color='white', fontsize=10)
                    plt.text(2, 9, 'Volume: ' + repr(np.round(mol.results['nucleosome_volume'], decimals=1))
                             + ' nm^3', color='white', fontsize=10)
                count_failed += 1
            fig.savefig(name, bbox_inches='tight')

            plt.close()

    if analyze_ints is True:
        analyzed_ints = results_final['analyzed_ints']
        if not os.path.exists(file_name + '/ints'):
            os.makedirs(file_name + '/ints')
            os.makedirs(file_name + '/ints/failed')
            os.makedirs(file_name + '/ints/ascii')
            os.makedirs(file_name + '/ints/failed/ascii')
        count = 0
        count_failed = 0

        for mol in analyzed_ints:
            header = 'File Format = ASCII\n'
            header = header + 'x-pixels = ' + repr(np.shape(mol.mol_filtered)[1]) + '\n'
            header = header + 'y-pixels = ' + repr(np.shape(mol.mol_filtered)[0]) + '\n'
            header = header + 'x-length = ' + repr(
                np.round(np.shape(mol.mol_filtered)[1] * mol.pixel_size, decimals=2)) + '\n'
            header = header + 'y-length = ' + repr(
                np.round(np.shape(mol.mol_filtered)[0] * mol.pixel_size, decimals=2)) + '\n'
            header = header + 'x-offset = ' + repr(mol.mol_bbox[1]) + '\n'
            header = header + 'y-offset = ' + repr(mol.mol_bbox[0]) + '\n'
            header = header + 'z-unit = nm\n'
            header = header + 'Start of Data:'

            fig = plt.figure()
            plt.axis('off')
            mol_filtered_rgb = convert_to_rgb(mol.mol_filtered, my_colormap)
            plt.imshow(mol_filtered_rgb, interpolation='None')

            if mol.results['failed'] is False:
                plot_ellipse_close_up([mol.results['com_r'], mol.results['com_c'],
                                       mol.results['radius_of_gyration'], mol.results['radius_of_gyration'],
                                       0, 0], double=False)
                if mol.results['com_core_r'] != 0:
                    plot_ellipse_close_up([mol.results['com_core_r'], mol.results['com_core_c'],
                                           mol.results['radius_of_gyration_core'], mol.results['radius_of_gyration_core'],
                                           0, 0], double=False)
                if mol.results['ellipsoid_height'] != 0:
                    plot_ellipse_close_up(mol.results['ellipsoid_coeff'], color='black')
                    plt.scatter(mol.results['ellipsoid_coeff'][1], mol.results['ellipsoid_coeff'][0],
                                color='#2A3132')

                name = file_name + '/ints/' + repr(count) + '.png'
                plt.text(2, 6, 'RoG: ' + repr(np.round(mol.results['radius_of_gyration'], decimals=1)),
                         color='white', fontsize=10)
                plt.text(2, 21, 'RoG Core: ' + repr(np.round(mol.results['radius_of_gyration_core'], decimals=1)),
                         color='white', fontsize=10)

                # save ascii
                ascname = file_name + '/ints/ascii/' + repr(count) + '.asc'
                np.savetxt(ascname, mol.mol_filtered, delimiter='\t', header=header)
                count += 1

            else:
                name = file_name + '/ints/failed/' + repr(count_failed) + '_failed.png'
                ascname = file_name + '/ints/failed/ascii/' + repr(count_failed) + '.asc'
                np.savetxt(ascname, mol.mol_filtered, delimiter='\t', header=header)
                count_failed += 1

            fig.savefig(name, bbox_inches='tight')
            plt.close()

    # Plot all molecules that were categorized as trash
    if plot_trash is True:
        mol_trash = results_final['mol_trash']
        if not os.path.exists(file_name + '/trash'):
            os.makedirs(file_name + '/trash')
        count = 0

        for mol in mol_trash:

            fig = plt.figure()
            plt.axis('off')
            mol_filtered_rgb = convert_to_rgb(mol.mol_filtered, my_colormap)

            # Mark parameters like the skeleton and endpoints
            for r, c in zip(*np.where(mol.mol_pars['mol_skel'] != 0)):
                mol_filtered_rgb[r, c] = np.array([42, 49, 50]) / 255
            for r, c in mol.mol_pars['skel_bps_pixels']:
                mol_filtered_rgb[r, c] = np.array([255, 187, 0]) / 255
            for r, c in mol.mol_pars['skel_eps_pixels']:
                mol_filtered_rgb[r, c] = np.array([255, 187, 0]) / 255

            # Write reason for disposal
            plt.text(2, 6, 'Discard reason: ' + mol.mol_pars['reason'], color='white', fontsize=12)
            plt.imshow(mol_filtered_rgb, interpolation='None')

            name = file_name + '/trash/' + repr(count) + '.png'
            count += 1
            fig.savefig(name, bbox_inches='tight')

            plt.close()

    return


def create_custom_colormap(midpoint=0.35):
    """ Define a self-made colormap starting at #375E97 going to 1.0 and ending at #D61800 """

    color_dict = {'red': ((0.0, 0.0, 0.261),
                          (midpoint, 1.0, 1.0),
                          (1.0, 0.839, 0)),

                  'green': ((0.0, 0.0, 0.369),
                            (midpoint, 1.0, 1.0),
                            (1.0, 0.094, 0)),

                  'blue': ((0.0, 0.0, 0.592),
                           (midpoint, 1.0, 1.0),
                           (1.0, 0, 0))
                  }

    my_colormap = LinearSegmentedColormap('my_colormap', color_dict)
    plt.register_cmap(cmap=my_colormap)

    return my_colormap


def convert_to_rgb(mol_filtered, my_colormap):
    """ Converts a greyscale img to RGB and scales it properly to a colormap"""

    # Shrink the image to values between 0.0 and 1.0 for rgb conversion
    mol_filtered = copy.deepcopy(mol_filtered)
    # Set one pixel high to make the colormap a little more spread out
    if np.amax(mol_filtered) <= 2.5:
        mol_filtered[0, 0] = 2.5
    mol_filtered *= 1.0 / mol_filtered.max()

    # Convert the greyscale image to a color image
    # colormap = plt.get_cmap('coolwarm')
    mol_filtered_rgba = my_colormap(mol_filtered)
    mol_filtered_rgb = np.delete(mol_filtered_rgba, 3, 2)

    return mol_filtered_rgb


def cv_apply_custom_colormap(image_gray, cmap=plt.get_cmap('seismic')):
    """ Used to apply a custom colormap to a image I want to use for openCV (required in manual pixel removal)"""
    # I CAN MAKE THIS NICER, DON'T REALLY THINK THAT I NEED THIS FUNCTION, I CAN MAYBE ALSO USE MY CONVERT TO RGB FUNC?

    assert image_gray.dtype == np.uint8, 'must be np.uint8 image'
    if image_gray.ndim == 3:
        image_gray = image_gray.squeeze(-1)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256))[:,0:3]    # color range RGBA => RGB
    color_range = (color_range*255.0).astype(np.uint8)         # [0,1] => [0,255]
    color_range = np.squeeze(np.dstack([color_range[:,2], color_range[:,1], color_range[:,0]]), 0)  # RGB => BGR

    # Apply colormap for each channel individually
    channels = [cv2.LUT(image_gray, color_range[:,i]) for i in range(3)]
    return np.dstack(channels)


def ellipse_points(x0, y0, a, b, phi=0, z_h=0.67):
    """ Calculate the points along the ellipse at a certain height of the ellipsoid """

    # define possible x-values, add/substract small number at the ends to prevent negative sqrt
    x = np.linspace(-a * (1 - z_h ** 2) + 0.0001, a * (1 - z_h ** 2) - 0.0001, 5000)
    y_pos = np.sqrt(b ** 2 * (1 - z_h ** 2) * (1 - x ** 2 / (a ** 2 * (1 - z_h ** 2)) - z_h ** 2))
    points = np.hstack((np.array([np.cos(-phi) * x - np.sin(-phi) * y_pos + x0, -np.sin(-phi) * x + np.cos(-phi) * y_pos + y0]),
                        np.flip(np.array([np.cos(-phi) * x - np.sin(-phi) * -y_pos + x0, -np.sin(-phi) * x + np.cos(-phi) * -y_pos + y0]), axis=1)))

    return points.T


def plot_wiggins_pixels_close_up(pixels):

    wiggins_pixels = copy.deepcopy(pixels)
    pixels_img = np.asarray([np.array([r, c])
                             for r, c in wiggins_pixels])
    plt.plot(pixels_img[:, 1], pixels_img[:, 0], color='#2A3132')

    return


def plot_ellipse_close_up(ellipsoid_coeff, double=True, color='#FFBB00'):

    # Plot the nucleosome ellipses
    pixels_ellipse = ellipse_points(x0=ellipsoid_coeff[0], y0=ellipsoid_coeff[1],
                                    a=ellipsoid_coeff[2], b=ellipsoid_coeff[3],
                                    phi=ellipsoid_coeff[5], z_h=0)
    plt.plot(pixels_ellipse[:, 1],
             pixels_ellipse[:, 0], color=color)

    if double is True:
        pixels_ellipse = ellipse_points(x0=ellipsoid_coeff[0], y0=ellipsoid_coeff[1],
                                        a=ellipsoid_coeff[2], b=ellipsoid_coeff[3],
                                        phi=ellipsoid_coeff[5], z_h=0.6)
        plt.plot(pixels_ellipse[:, 1],
                 pixels_ellipse[:, 0], color=color)

    return


def dna_height_plots(molecules):
    """ Plot a few useful diagram to check the DNA height """

    height_avg = [mol.results['height_avg'] for mol in molecules if mol.results['failed'] is False]

    return
