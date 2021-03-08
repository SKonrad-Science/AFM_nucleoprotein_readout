""" Functions to plot the image analysis results """

import copy
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import cv2


def plot_overview_image(img_filtered,
                        file_name,
                        results_final,
                        analyze_bare_DNA=False,
                        analyze_nucleosomes=False,
                        analyze_nucleosomes_eb=False
                        ):
    """ Plot overview image with all analyzed molecules marked """

    my_colormap = create_custom_colormap_2()
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(img_filtered, interpolation='None', cmap=my_colormap)

    # Plot the analyzed bare DNA
    if analyze_bare_DNA is True:

        analyzed_bare_DNA = results_final['analyzed_bare_DNA']
        dna_succeeded = [mol for mol in analyzed_bare_DNA if mol.results['failed'] is False]
        dna_failed = [mol for mol in analyzed_bare_DNA if mol.results['failed'] is True]

        for mol in dna_succeeded:
            if mol.results['length_fwd'] is not False:
                trace_points = copy.deepcopy(mol.results['wigg_fwd'])
            else:
                trace_points = copy.deepcopy(mol.results['wigg_bwd'])
            plot_trace_points(trace_points, mol, color='#FFDB5C', linewidth=1.5)

        # for mol in dna_failed:
        #     plot_trace_points(copy.deepcopy(mol.results['wigg_fwd']), mol, color='#FB6542', linewidth=1.5)
        #     plot_trace_points(copy.deepcopy(mol.results['wigg_bwd']), mol, color='#FB6542', linewidth=1.5)

    # Plot all nucleosomes
    if analyze_nucleosomes is True:

        analyzed_nucleosomes = results_final['analyzed_nucleosomes']
        nuc_succeeded = [mol for mol in analyzed_nucleosomes if mol.results['failed'] is False]
        nuc_failed = [mol for mol in analyzed_nucleosomes if mol.results['failed'] is True]

        for mol in nuc_succeeded:

            # Plot the Wiggins trace
            points_arm1 = copy.deepcopy(mol.results['pixels_arm1'])
            points_arm2 = copy.deepcopy(mol.results['pixels_arm2'])
            plot_trace_points(points_arm1, mol, color='#FA812F', linewidth=1.5)
            plot_trace_points(points_arm2, mol, color='#FA812F', linewidth=1.5)

            # Plot the angle
            ell_data = mol.results['ell_data']
            center = ell_data['center']
            points_angle = np.array([[points_arm1[-1][0] - 10 + mol.mol_pars['mol_bbox'][0],
                                      points_arm1[-1][1] - 10 + mol.mol_pars['mol_bbox'][1]],
                                     [center[0] - 10 + mol.mol_pars['mol_bbox'][0],
                                      center[1] - 10 + mol.mol_pars['mol_bbox'][1]],
                                     [points_arm2[-1][0] - 10 + mol.mol_pars['mol_bbox'][0],
                                      points_arm2[-1][1] - 10 + mol.mol_pars['mol_bbox'][1]]])
            plt.plot(points_angle[:, 1], points_angle[:, 0], color='#FA812F', linewidth=1.5)

            # Plot the nucleosome ellipses
            ax.add_patch(plot_ellipse(ell_data, mol, ell_cutoff=0, edgecolor='#FA812F'))
            ax.add_patch(plot_ellipse(ell_data, mol, ell_cutoff=0.6, edgecolor='#FA812F'))

        # for mol in nuc_failed:
        #     # Check that the ellipsoid fit worked, otherwise don't try plotting since arms weren't traced
        #     if 'pixels_arm1' and 'pixels_arm2' in mol.results:
        #         plot_trace_points(copy.deepcopy(mol.results['pixels_arm1']), mol, color='#FF420E', linewidth=1.5)
        #         plot_trace_points(copy.deepcopy(mol.results['pixels_arm2']), mol, color='#FF420E', linewidth=1.5)

    # Plot all endbound nucleosomes
    if analyze_nucleosomes_eb is True:
        analyzed_nucleosomes_eb = results_final['analyzed_nucleosomes_eb']
        nuc_succeeded = [mol for mol in analyzed_nucleosomes_eb if mol.results['failed'] is False]

        for mol in nuc_succeeded:

            # Plot the Wiggins trace
            plot_trace_points(copy.deepcopy(mol.results['pixels_arm1']), mol, color='green', linewidth=1.5)

            ell_data = mol.results['ell_data']
            ax.add_patch(plot_ellipse(ell_data, mol, ell_cutoff=0, edgecolor='green'))
            ax.add_patch(plot_ellipse(ell_data, mol, ell_cutoff=0.6, edgecolor='green'))

    plt.show()
    fig.savefig(file_name + '_overview.png', bbox_inches='tight')

    return


def plot_save_close_ups(results_final,
                        file_name,
                        analyze_bare_DNA=False,
                        analyze_nucleosomes=False,
                        analyze_nucleosomes_eb=False,
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
                wiggins_pixels = copy.deepcopy(mol.results['wigg_fwd'])
            elif mol.results['length_bwd'] is not False:
                wiggins_pixels = copy.deepcopy(mol.results['wigg_bwd'])

            plt.imshow(mol_filtered_rgb, interpolation='None')

            if mol.results['failed'] is False:
                plot_trace_points_close_up(wiggins_pixels)
                name = file_name + '/bare_DNA/' + repr(count) + '.png'
                plot_text(['length_avg'], mol, pos=6)
                count += 1
            else:
                name = file_name + '/bare_DNA/failed/' + repr(count_failed) + '_failed.png'
                if mol.results['failed_reason'] == 'Discarded manually':
                    plot_trace_points_close_up(wiggins_pixels)
                    plot_text(['length_avg'], mol, pos=6)
                plot_text(['failed_reason'], mol)
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
                fig, ax = plt.subplots(1, 1)
                plt.axis('off')
                mol_filtered_rgb = convert_to_rgb(mol.mol_filtered, my_colormap)

                plot_trace_points_close_up(mol.results['pixels_arm1'])
                plot_trace_points_close_up(mol.results['pixels_arm2'])

                ell_data = mol.results['ell_data']
                ax.add_patch(plot_ellipse_close_up(ell_data, ell_cutoff=0, edgecolor='#FFBB00'))
                ax.add_patch(plot_ellipse_close_up(ell_data, ell_cutoff=0.6, edgecolor='#FFBB00'))
                plt.imshow(mol_filtered_rgb, interpolation='None')

                if mol.results['failed'] is False:
                    name = file_name + '/nucleosomes/' + repr(count) + '.png'
                    plot_text(['length_sum', 'angle_arms', 'nucleosome_volume'], mol)
                    count += 1
                else:
                    name = file_name + '/nucleosomes/failed/' + repr(count_failed) + '_failed.png'
                    plot_text(['failed_reason'], mol)
                    if mol.results['failed_reason'] == 'Discarded manually':
                        plot_text(['length_sum', 'angle_arms', 'nucleosome_volume'], mol, pos=6)
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

            fig, ax = plt.subplots(1, 1)
            plt.axis('off')
            mol_filtered_rgb = convert_to_rgb(mol.mol_filtered, my_colormap)
            plt.imshow(mol_filtered_rgb, interpolation='None')

            if mol.results['failed'] is False:
                ell_data = mol.results['ell_data']
                plot_trace_points_close_up(mol.results['pixels_arm1'])
                ax.add_patch(plot_ellipse_close_up(ell_data, ell_cutoff=0, edgecolor='#FFBB00'))
                ax.add_patch(plot_ellipse_close_up(ell_data, ell_cutoff=0.6, edgecolor='#FFBB00'))

                name = file_name + '/nucleosomes_eb/' + repr(count) + '.png'
                plot_text(['length_arm1_60', 'nucleosome_volume'], mol)
                count += 1
            else:
                name = file_name + '/nucleosomes_eb/failed/' + repr(count_failed) + '_failed.png'
                if 'failed_reason' in mol.results:
                    plot_text(['failed_reason'], mol)
                    if mol.results['failed_reason'] == 'Discarded manually':
                        plot_trace_points_close_up(mol.results['pixels_arm1'])
                        ax.add_patch(plot_ellipse_close_up(ell_data, ell_cutoff=0, edgecolor='#FFBB00'))
                        ax.add_patch(plot_ellipse_close_up(ell_data, ell_cutoff=0.6, edgecolor='#FFBB00'))
                        plot_text(['length_arm1_60', 'nucleosome_volume'], mol, pos=6)
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


def create_custom_colormap_2(midpoint=0.2):
    """ Define a self-made colormap starting at #375E97 going to 1.0 and ending at #D61800 """

    color_dict = {'red': ((0.0, 0.0, 0.024),
                          (midpoint, 1.0, 1.0),
                          (1.0, 0.976, 0)),

                  'green': ((0.0, 0.0, 0.220),
                            (midpoint, 1.0, 1.0),
                            (1.0, 0.651, 0)),

                  'blue': ((0.0, 0.0, 0.322),
                           (midpoint, 1.0, 1.0),
                           (1.0, 0.012, 0))
                  }

    my_colormap = LinearSegmentedColormap('my_colormap', color_dict)
    plt.register_cmap(cmap=my_colormap)

    return my_colormap


def create_custom_colormap_3(midpoint=0.30):
    """ Define a self-made colormap starting at #375E97 going to 1.0 and ending at #D61800 """

    color_dict = {'red': ((0.0, 0.0, 0.754),
                          (midpoint, 1.0, 1.0),
                          (1.0, 0.976, 0)),

                  'green': ((0.0, 0.0, 0.816),
                            (midpoint, 1.0, 1.0),
                            (1.0, 0.651, 0)),

                  'blue': ((0.0, 0.0, 0.922),
                           (midpoint, 1.0, 1.0),
                           (1.0, 0.012, 0))
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


def plot_trace_points(arm_pixels, mol, color='#FB6542', linewidth=1.5):

    points = np.asarray([np.array([r - 10 + mol.mol_pars['mol_bbox'][0], c - 10 + mol.mol_pars['mol_bbox'][1]])
                         for r, c in arm_pixels])
    plt.plot(points[:, 1], points[:, 0], color=color, linewidth=linewidth)

    return


def plot_ellipse(ell_data, mol, ell_cutoff=0, edgecolor='#FFBB00'):

    center = ell_data['center']
    a, b, c = ell_data['abc']
    ell_plot_patch = matplotlib.patches.Ellipse((center[1] - 10 + mol.mol_pars['mol_bbox'][1],
                                                 center[0] - 10 + mol.mol_pars['mol_bbox'][0]),
                                                2 * a * (1 - ell_cutoff ** 2), 2 * b * (1 - ell_cutoff ** 2),
                                                angle=-ell_data['rot_angle'] * 180 / np.pi,
                                                facecolor='None', edgecolor=edgecolor)
    return ell_plot_patch


def plot_trace_points_close_up(pixels):

    wiggins_pixels = copy.deepcopy(pixels)
    pixels_img = np.asarray([np.array([r, c])
                             for r, c in wiggins_pixels])
    plt.plot(pixels_img[:, 1], pixels_img[:, 0], color='#2A3132')

    return


def plot_ellipse_close_up(ell_data, ell_cutoff=0, edgecolor='#FFBB00'):

    center = ell_data['center']
    a, b, c = ell_data['abc']
    ell_plot_patch = matplotlib.patches.Ellipse((center[1],
                                                 center[0]),
                                                2 * a * (1 - ell_cutoff ** 2), 2 * b * (1 - ell_cutoff ** 2),
                                                angle=-ell_data['rot_angle'] * 180 / np.pi,
                                                facecolor='None', edgecolor=edgecolor)
    plt.scatter(center[1], center[0], color=edgecolor)

    return ell_plot_patch


def plot_text(key_list, mol, pos=3):

    for key in key_list:
        if key == 'failed_reason':
            plt.text(2, pos, key + ': ' + mol.results[key], color='white', fontsize=8)
        else:
            plt.text(2, pos, key + ': ' + repr(np.round(mol.results[key], decimals=1)), color='white', fontsize=8)
        pos += 3

    return
