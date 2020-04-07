"""
Analysis functions
"""

import copy
import numpy as np
import scipy
import scipy.interpolate as interp
from scipy.optimize import curve_fit
import math

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import plot_functions as plot

neighbour_matrix = np.array([[1., 1., 1.],
                            [1., 0., 1.],
                            [1., 1., 1.]])


def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def skel_pars(mol_skeleton):
    """
    Method to calculate parameters of a skeleton: Endpoints, branch points and the amount of pixels
    Endpoints - End pixels of the skeleton
    Branchpoints - Pixels that have three or more non-zero neighbours

    Input:
        mol_skeleton - array
            Binary image with values 0 and 1. 1 represents the one-pixel wide skeleton of the image produced by
            skimage morphology.skeletonize

    Output:
        eps_pixels - array
            x any y coordinated of the pixels of the endpoints
        eps_number - int
            Amount of endpoints of the skeleton
        bps_pixels - array
            x any y coordinated of the pixels of the branchpoints
        bps_number - int
            Amount of branchpoints of the skeleton
        pixels_number - int
            Amount of all pixels in the skeleton. Can be used to estimate the length of the structure and thus
            helps to classify whether a structure is too small or too large to be a proper molecule
    """

    mol_skel = copy.deepcopy(mol_skeleton)

    # Calculate the number of non-zero neighbouring pixels for each pixel of the skeleton
    mol_neighbours = np.zeros_like(mol_skel)
    for r, c in zip(*np.where(mol_skel == 1)):
        mol_neighbours[r, c] = np.sum(np.multiply(mol_skel[r - 1:r + 2, c - 1:c + 2], neighbour_matrix))

    # Calculate the different numbers
    eps_pixels = np.array(np.where(mol_neighbours == 1)).T
    eps_number = np.shape(np.array(np.where(mol_neighbours == 1)))[1]
    bps_pixels = np.array(np.where(mol_neighbours >= 3)).T
    bps_number = np.shape(np.array(np.where(mol_neighbours >= 3)))[1]
    pixels_number = len(mol_skel[mol_skel != 0])

    return eps_pixels, eps_number, bps_pixels, bps_number, pixels_number


def sort_skel(mol_pars, start):
    """
     Sort the pixels of the skeleton given to the function. It starts at the point given as "start" to the function
     and ends at the first point that does not have exactly two neighbours (1 neighbour == endpoint and more than
     2 neigbours == branchpoint)

     Input:
        mol_pars - dict
            Contains the skeleton of the molecule as a the key 'mol_skel' (array)
        start - array
            row and column value of the start pixel

    Output:
        skel_sorted - list of tuples
            the list contains one tuple for each pixel of the sorted skeleton with the first entry of the list being
            the start point and the last entry being the endpoint/branchpoint of the skeleton. The tuples in between
            represent the skeleton pixels in order from start to end
     """

    mol_skel = copy.deepcopy(mol_pars['mol_skel'])
    # Calculate the number of non-zero neighbouring pixels for each pixel of the skeleton
    mol_neighbours = np.zeros_like(mol_skel)
    for r, c in zip(*np.where(mol_skel == 1)):
        mol_neighbours[r, c] = np.sum(np.multiply(mol_skel[r - 1:r + 2, c - 1:c + 2], neighbour_matrix))

    curr_row, curr_col = start
    skel_sorted = [tuple((curr_row, curr_col))]
    mol_neighbours[curr_row, curr_col] = 2

    while mol_neighbours[curr_row, curr_col] == 2:
        # Set current location to zero such that this point of the skeleton is not used again
        mol_neighbours[curr_row, curr_col] = 0

        # Calculate distance of the current pixel to each remaining pixel of the skeleton
        pixels_remaining = np.where(mol_neighbours != 0)
        distances = [np.sqrt(abs((row - curr_row) ** 2 + (col - curr_col) ** 2))
                     for row, col in zip(*pixels_remaining)]

        # Pixel with the minimal distance is the next one in the skeleton
        index = distances.index(min(distances))
        curr_row = pixels_remaining[0][index]
        curr_col = pixels_remaining[1][index]
        skel_sorted.append(tuple((curr_row, curr_col)))

    return skel_sorted


def skeletonize_end(mol_filtered, mol_pars, skel_end_pixels):
    """
    Skeletonize the end of a skeleton given to the function until you reach the end of the filtered molecule
    (to get the skeleton pixels are eroded from the filtered molecule from all sides, thus also from the two ends.
    To not lose molecule length at the ends this additional skeletonization is done).
    The skeleton is prolonged into the direction of the vector between the end point and a few skeleton points before
    the skeleton end.

    Input:
        mol_filtered - array
            The numpy array of the filtered molecule with its height values
        mol_pars - dict
            Contains the skeleton of the molecule with key 'mol_skel'
        skel_end_pixels - array
            The pixels that should be used as end pixel and the following pixels to calculate the direction of the
            additional skeleton pixels

    Output:
        mol_skel - array
            Skeleton with the skeletonized end included
    """

    mol_skel = copy.deepcopy(mol_pars['mol_skel'])
    mol_filtered = copy.deepcopy(mol_filtered)

    # Find the directions of the vectors between the end and the pixels before. Then calculate the avg. direction
    skel_directions = []
    for i in range(1, len(skel_end_pixels)):
        vector = np.subtract(skel_end_pixels[0], skel_end_pixels[i])
        skel_directions.append(vector/np.linalg.norm(vector))
    skel_directions = np.asarray(skel_directions)
    skel_mean_direction = np.array([sum(skel_directions[:, 0]), sum(skel_directions[:, 1])])
    skel_mean_direction = skel_mean_direction / np.linalg.norm(skel_mean_direction)

    # Produce a set of the potential new skeleton pixels by using the skel_mean_direction vector
    potential_pixels = [tuple(np.round((skel_end_pixels[0] + skel_mean_direction*x))) for x in np.linspace(0, 10, 100)]
    # Remove the excessive pixels once one pixel of the potential pixels touches 0 to prevent skeleton self-crossing
    for i in range(len(potential_pixels)):
        if mol_filtered[int(potential_pixels[i][0]), int(potential_pixels[i][1])] == 0:
            potential_pixels = copy.deepcopy(potential_pixels[0:i])
            break
    potential_pixels = set([(int(row), int(col)) for row, col in potential_pixels])

    # Only pixels that are within the filtered object can be new skeleton pixels (also remove current skel endpoint)
    pixel_copy = copy.deepcopy(potential_pixels)
    for pixel in pixel_copy:
        if mol_filtered[pixel[0], pixel[1]] == 0 or pixel == skel_end_pixels[0]:
            potential_pixels.remove(pixel)
    potential_pixels = list(potential_pixels)

    # Check for all potential pixels whether they can be added ( e.g. have only one neighbour - the previous endpoint)
    curr_row, curr_col = skel_end_pixels[0]
    while len(potential_pixels) != 0:
        prev_row, prev_col = copy.deepcopy((curr_row, curr_col))

        # Add skeleton point where the distance of the current endpoint is the smallest to the potential new pixel
        distances = [np.sqrt(abs((pixel[0] - curr_row) ** 2 + (pixel[1] - curr_col) ** 2))
                     for pixel in potential_pixels]
        index = distances.index(min(distances))
        curr_row, curr_col = potential_pixels[index]
        mol_skel[curr_row, curr_col] = 1

        # Calculate the amount of neighbors for each skeleton pixel
        mol_neighbours = np.zeros_like(mol_skel)
        for r, c in zip(*np.where(mol_skel == 1)):
            mol_neighbours[r, c] = np.sum(np.multiply(mol_skel[r - 1:r + 2, c - 1:c + 2], neighbour_matrix))

        # In case the new skeleton pixel has two neighbours, remove the previously added pixel -> 1-pixel wide skeleton
        if mol_neighbours[curr_row, curr_col] == 2:
            mol_skel[prev_row, prev_col] = 0
        potential_pixels.pop(index)

    return mol_skel


def wiggins(mol_filtered, seg_length, start, end, mol_type, ellipsoid_coeff=None, failed=False):
    """ Wiggin's algorithm to trace the DNA length """

    seg_length_orig = copy.deepcopy(seg_length)
    rot_matrix = np.array([[0, -1], [1, 0]])
    num_interp_values = 50

    curr_point = np.asarray(start[0])
    next_point = np.asarray(start[-1])
    end_point = np.asarray(end)
    wiggins_points = [curr_point]

    conditions = {
        lambda: mol_type == 'Bare DNA' and np.linalg.norm(end_point - curr_point) > seg_length_orig,
        # lambda: mol_type == 'Nucleosome' and
        #         (np.linalg.norm(end_point - curr_point) > 1*seg_length_orig or
        #          1.5*np.linalg.norm(end_point - curr_point) > np.linalg.norm(end_point - next_point))
        lambda: mol_type == 'Nucleosome' and
        np.linalg.norm(ellipsoid_coeff[0:2] - curr_point) > np.amax(ellipsoid_coeff[2:4])
    }
    while any(cond() for cond in conditions):

        if np.linalg.norm(end_point - curr_point) <= 0.5*seg_length_orig:
            seg_length = 1.5

        # Direction and perpendicular direction of the first segment
        direction = (next_point - curr_point)/np.linalg.norm(next_point - curr_point) * seg_length
        direction_perp = rot_matrix.dot(direction) / np.linalg.norm(direction)

        # Create the meshgrid on which the interpolation will be carried out
        curr_point_int = np.round(curr_point).astype(int)
        curr_row, curr_col = curr_point_int
        grid_size = math.ceil(seg_length) + 2
        rr = np.linspace(curr_row - grid_size, curr_row + grid_size, 2 * grid_size + 1)
        cc = np.linspace(curr_col - grid_size, curr_col + grid_size, 2 * grid_size + 1)
        cc, rr = np.meshgrid(cc, rr)

        # Calculate interpolation function
        height_grid = copy.deepcopy(mol_filtered[int(np.amin(rr)):int(np.amax(rr)) + 1,
                                                 int(np.amin(cc)):int(np.amax(cc)) + 1])
        interp_function = interp.Rbf(rr, cc, height_grid, function='linear')

        # Repeat the interpolation several times to get the best position
        for _ in range(3):
            # Use interpolation function to calculate a perpend. height profile and find best position along profile
            next_row, next_col = next_point
            r_linspace = np.linspace(next_row - 4*direction_perp[0], next_row + 4*direction_perp[0], num_interp_values)
            c_linspace = np.linspace(next_col - 4*direction_perp[1], next_col + 4*direction_perp[1], num_interp_values)
            height_profile = interp_function(r_linspace, c_linspace)
            profile_position = np.linspace(1, num_interp_values, num_interp_values)
            best_position = np.mean(height_profile * profile_position) / np.mean(height_profile)
            if best_position < 1 or best_position > 50:
                failed = True
                break

            # Find the according row and column location for this best position via linear interpolation
            next_point = np.array([interp.interp1d(profile_position, r_linspace, kind='linear')(best_position),
                                   interp.interp1d(profile_position, c_linspace, kind='linear')(best_position)])
            direction = (next_point - curr_point) / np.linalg.norm(next_point - curr_point) * seg_length
            direction_perp = rot_matrix.dot(direction) / np.linalg.norm(direction)
            next_point = curr_point + direction

        # Set the new current location and the next location for the next segment
        curr_point = copy.deepcopy(next_point)
        next_point = curr_point + direction
        wiggins_points.append(curr_point)

        if mol_filtered[int(np.round(curr_point[0])), int(np.round(curr_point[1]))] == 0 or failed is True:
            failed = True
            break

        if len(wiggins_points) >= 1000:
            failed = True
            break

    if len(wiggins_points) <= 1:
        failed = True

    if failed is False:
        angle = angle_between(end_point - curr_point, curr_point - wiggins_points[-2])

    if failed is False and mol_type == 'Bare DNA' and angle < 60:
        wiggins_points.append(end_point)

    return wiggins_points, failed


def radius_of_gyration(mol_filtered):
    """ Calculate the center of mass and the radius of gyration of the molecule """
    # Improve to handel the case of Integrase where no core molecule is existent to not give 'nan'
    mol_img = copy.deepcopy(mol_filtered)

    center_of_mass = scipy.ndimage.center_of_mass(mol_img)
    pixel_heights = mol_img[mol_img != 0]
    distances = np.asarray([np.linalg.norm(pixel - center_of_mass) for pixel in np.argwhere(mol_img)])

    # Radius of gyration formula according to Wikipedia
    radius = np.sqrt((np.sum(pixel_heights*(distances**2) / np.sum(pixel_heights))))

    return radius, center_of_mass


def ellipsoid(xy, x0, y0, a, b, c):
    """ Function for an ellipsoid fit based on the given parameters - 2 variables, 5 parameters """
    x, y = xy[0, :], xy[1, :]
    result = []
    for x_i, y_i in zip(x, y):
        if (1 - (x_i-x0)**2/a**2 - (y_i-y0)**2/b**2) <= 0:
            result.append(0)
        else:
            result.append(c*np.sqrt((1 - (x_i-x0)**2/a**2 - (y_i-y0)**2/b**2)))
    return np.asarray(result)


def ellipsoid_phi(xy, x0, y0, a, b, c, phi):
    """ Function for an ellipsoid fit based on the given parameters - 2 variables, 5 parameters """
    # How to calculate the function: Formula for an ellipsoid is on https://de.wikipedia.org/wiki/Ellipsoid
    # Solve x²/a² + y²/b² + z²/c² = 1 for z. In our case x = x_i - x0 and y = y_1 - y0 since our ellipsoid is not in
    # the origin of the coordinate system
    # This way you would already have an ellipsoid but it can not rotate along the z axis
    # To implement the rotation use the 2D rotation matrix R = (cos(phi) -sin(phi), sin(phi) cos(phi)) and apply it to
    # the base plane coordinates x and y :
    # x -> np.cos(phi) * (x_i - x0) - np.sin(phi) * (y_i - y0)
    # y -> -np.sin(phi) * (x_i - x0) + np.cos(phi) * (y_i - y0)
    # Now we have a function that can use x_i and y_i as input for the fit (the locations for which I have the height
    # values measured in my AFM image).
    # Additionally, there are 6 parameters that are fitted by scipy:
    # x0, y0: The center of the ellipsoid along the xy plane
    # a, b, c: The extension parameters of the ellipsoid
    # phi: the rotation of the ellipsoid around the z-axis

    x, y = xy[0, :], xy[1, :]
    result = []
    for x_i, y_i in zip(x, y):
        if (1 - ((np.cos(phi) * (x_i - x0) - np.sin(phi) * (y_i - y0))**2/a**2 +
                 (-np.sin(phi) * (x_i - x0) + np.cos(phi) * (y_i - y0))**2/b**2)) <= 0:
            result.append(0)
        else:
            result.append(c*np.sqrt((1 - ((np.cos(phi) * (x_i - x0) - np.sin(phi) * (y_i - y0))**2/a**2 +
                                          (-np.sin(phi) * (x_i - x0) + np.cos(phi) * (y_i - y0))**2/b**2))))

    return np.asarray(result)


def nuc_core_ellipsoid_fit(mol_filtered, center_of_mass_core, grid_size=10, start=[5., 5., 2.]):
    """ Fit half an ellipsoid to the nucleosome core particle """
    # Define parameters
    com_int = np.round(center_of_mass_core).astype(int)
    rr = np.linspace(com_int[0] - grid_size, com_int[0] + grid_size, 2 * grid_size + 1)
    cc = np.linspace(com_int[1] - grid_size, com_int[1] + grid_size, 2 * grid_size + 1)
    cc, rr = np.meshgrid(cc, rr)
    height_grid = copy.deepcopy(mol_filtered[int(np.amin(rr)):int(np.amax(rr)) + 1,
                                int(np.amin(cc)):int(np.amax(cc)) + 1])
    # height_grid[height_grid < 1] = 0
    rc_stack = np.vstack((rr.flatten(), cc.flatten()))

    # Try fit
    coeff_start = [com_int[0], com_int[1], start[0], start[1], start[2], 0.]
    coeff, var_matrix = curve_fit(ellipsoid_phi, rc_stack, height_grid.flatten(), p0=coeff_start)

    # Cut out the part from the mol_filtered where the z_values of the fitted ellipsoid are > 0
    # This image is then used to apply the Wiggins algorithm to it
    z_values = ellipsoid_phi(rc_stack, *coeff)
    ellipsoid_pixels = np.asarray([rc_stack[:, i] for i in range(0, len(z_values)) if z_values[i] != 0])
    mol_nuc_ellipsoid_cut = copy.deepcopy(mol_filtered)
    for r, c in ellipsoid_pixels:
        mol_nuc_ellipsoid_cut[int(r), int(c)] = np.mean(mol_filtered[mol_filtered != 0])

    # Plot results
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(rr, cc, height_grid, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # surf = ax.plot_surface(rr, cc, z_values.reshape((grid_size*2 + 1, grid_size*2 + 1)), cmap=cm.afmhot,
    #                        linewidth=0, antialiased=False, shade=True)
    # plt.show()

    return coeff, var_matrix, mol_nuc_ellipsoid_cut, ellipsoid_pixels


def angle_between(v1, v2):
    """ Calculate the angle between two vectors """
    # Calculate unit vectors
    v1_unit = v1 / np.linalg.norm(v1)
    v2_unit = v2 / np.linalg.norm(v2)

    # Calculate the dotproduct between the vectors and the angle (rad) afterwards
    dotproduct = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
    angle = np.arccos(dotproduct) * 180 / np.pi
    return angle


def ellipse_arm_pixel(pixels_arm, ellipsoid_coeff, z_h=0.5, first=False):
    """ Add an additional pixel to the nucleosome arm where the vector between the last arm pixel and the ellipsoid
    center intersect with the ellipse at a certain height. """

    arm_vector = pixels_arm[-1] - ellipsoid_coeff[0:2]      # vector between center of ellipsoid and last arm pixel
    phi = ellipsoid_coeff[5]

    # Define parameters to calculate the r and c position along the ellipse
    slope = arm_vector[0]/arm_vector[1]                     # row = slope * column
    a_dash = ellipsoid_coeff[2] * np.sqrt(1 - z_h ** 2)
    b_dash = ellipsoid_coeff[3] * np.sqrt(1 - z_h ** 2)
    col = np.sqrt((b_dash ** 2 * (1 - z_h ** 2)) / (1 + b_dash ** 2 * slope ** 2 / a_dash ** 2))
    row = np.sqrt(a_dash ** 2 * (1 - (col / b_dash) ** 2 - z_h ** 2))

    # Depending on relative position of the arm pixel and the ellipse center the signs have to be adjusted
    if arm_vector[0] <= 0:
        row = -row
    if arm_vector[1] <= 0:
        col = -col

    # Apply the rotation angle of the ellipsoid
    ell_pixel = np.array([np.cos(-phi) * row - np.sin(-phi) * col + ellipsoid_coeff[0],
                         -np.sin(-phi) * row + np.cos(-phi) * col + ellipsoid_coeff[1]])

    # This is part of my ellipsoid fit problem fix - look at this to improve it
    # Main problem: Rotation parameter phi of the fit is not really an angle. Can not find its physical meaning but
    # The problem was that pixels on a rotated ellipse had somewhat wrong angles
    # This fix doesn't solve it completely. Still yields a length error of about 0.01-0.05nm per arm. Angles are exact!
    if first is False and np.linalg.norm(pixels_arm[-1] - ellipsoid_coeff[0:2]) >= np.linalg.norm(
            ell_pixel - ellipsoid_coeff[0:2]):
        ell_pixel = pixels_arm[-1] - arm_vector * np.linalg.norm(pixels_arm[-1] - ell_pixel)/np.linalg.norm(arm_vector)
    elif first is False and np.linalg.norm(pixels_arm[-1] - ellipsoid_coeff[0:2]) < np.linalg.norm(
            ell_pixel - ellipsoid_coeff[0:2]):
        ell_pixel = pixels_arm[-1] + arm_vector * np.linalg.norm(pixels_arm[-1] - ell_pixel) / np.linalg.norm(
            arm_vector)

    # makes the ellipse pixels more accurately but also computationally heavy
    # if first is False:
    #     ellipse_points = plot.ellipse_points(x0=ellipsoid_coeff[0], y0=ellipsoid_coeff[1],
    #                                          a=ellipsoid_coeff[2], b=ellipsoid_coeff[3],
    #                                          phi=ellipsoid_coeff[5], z_h=z_h)
    #     dists = [np.linalg.norm(ell_pixel - point) for point in ellipse_points]
    #     ell_pixel = ell_pixel + (ellipsoid_coeff[0:2] - ell_pixel) / np.linalg.norm(ellipsoid_coeff[0:2] - ell_pixel) * min(dists)

    # ellipse_pixel = np.array([row + ellipsoid_coeff[0], col + ellipsoid_coeff[1]])

    return ell_pixel


def wiggins_pixel_height_analysis(pixels, mol_filtered, pixel_size):

    heights = [mol_filtered[int(np.round(r, decimals=0)), int(np.round(c, decimals=0))] for r, c in pixels]
    lengths = [np.linalg.norm(pixels[i] - pixels[i+1]) * pixel_size for i in range(0, len(pixels)-1)]
    slopes = [(heights[i + 1] - heights[i]) / lengths[i] for i in range(0, len(heights)-1)]

    height_pars = {'heights': heights,
                   'lengths': lengths,
                   'slopes': slopes,
                   'height_avg': np.mean(abs(np.asarray(heights))),
                   'slope_avg': np.mean(abs(np.asarray(slopes))),
                   'height_std': np.std(heights),
                   'slope_std': np.std(slopes)}

    return height_pars


def dna_orientation(wiggins_pixels, mol_bbox):
    """ Calculate the average orientation of the DNA in the image """

    distances = [np.linalg.norm(wiggins_pixels[i + 1] - wiggins_pixels[i]) for i in range(0, len(wiggins_pixels) - 1)]
    vectors = [wiggins_pixels[i + 1] - wiggins_pixels[i] for i in range(0, len(wiggins_pixels) - 1)]

    # Calculate the angles between the dna trace vectors and a vector facing to the right
    angles = [angle_between(vectors[i], np.array([0, 1])) for i in range(0, len(vectors))]
    for i in range(0, len(angles)):
        if angles[i] > 90:
            angles[i] = angles[i] - (2 * abs(angles[i] - 90))
    topness = np.asarray(angles) / 90
    # Weigh the topness since distances between angles are not always the same
    topness_weighted = np.sum(topness * np.asarray(distances)) / np.sum(distances)
    rightness_weighted = 1 - topness_weighted

    orientation_pars = {'rightness': rightness_weighted,
                        'extension_right': mol_bbox[3] - mol_bbox[1],
                        'extension_bot': mol_bbox[2] - mol_bbox[0]}

    return orientation_pars


def bending_behaviour(wiggins_pixels):

    distances = [np.linalg.norm(wiggins_pixels[i + 1] - wiggins_pixels[i]) for i in range(0, len(wiggins_pixels) - 1)]
    vectors = [wiggins_pixels[i + 1] - wiggins_pixels[i] for i in range(0, len(wiggins_pixels) - 1)]
    angles = [angle_between(vectors[i + 1], vectors[i]) for i in range(0, len(vectors) - 1)]

    bending_avg = np.sum(angles[:-1])/len(angles[:-1])

    bending_pars = {'bending_avg': bending_avg}

    return bending_pars


def rotate_vector(vector, theta):

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    vec_rotated = np.dot(R, vector)

    return vec_rotated


def interp_tracing(mol_original, skel_arm_pixels, ellipsoid_pixels):

    interp_function, failed = get_interp_function(mol_original, skel_arm_pixels[0])
    # strand_points = [find_best_position(interp_function, skel_arm_pixels[0])]
    strand_points = [np.asarray(skel_arm_pixels[0])]
    direction_guess = skel_arm_pixels[3] - strand_points[-1]
    next_position = find_next_position(interp_function, strand_points[-1], direction_guess)
    next_best = find_best_position(interp_function, next_position)
    strand_points.append(
        strand_points[-1] + (next_best - strand_points[-1]) / np.linalg.norm(next_best - strand_points[-1]))
    while not (ellipsoid_pixels == np.round(strand_points[-1])).all(1).any() and not failed:
        interp_function, failed = get_interp_function(mol_original, strand_points[-1])
        next_position = find_next_position(interp_function, strand_points[-1], strand_points[-1] - strand_points[-2])
        next_best = find_best_position(interp_function, next_position, strand=strand_points)
        strand_points.append(
            strand_points[-1] + (next_best - strand_points[-1]) / np.linalg.norm(next_best - strand_points[-1]))
        if len(strand_points) > 200:
            failed = True
            break
        if failed is True:
            break

    return strand_points, failed


def interp_tracing_end(mol_original, strand_points, ellipsoid_pixels, failed=False, seg_length=3.41):

    try: # make this try stuff better
        while not (ellipsoid_pixels == np.round(strand_points[-1])).all(1).any() and not failed:
            interp_function, failed = get_interp_function(mol_original, strand_points[-1])
            next_position = find_next_position(interp_function, strand_points[-1], strand_points[-1] - strand_points[-2])
            next_best = find_best_position(interp_function, next_position, strand=strand_points)
            strand_points.append(
                strand_points[-1] + (next_best - strand_points[-1]) / np.linalg.norm(next_best - strand_points[-1]))
            if len(strand_points) > 200:
                failed = True
                break
            if failed is True:
                break
    except:
        failed = True

    # Now place 5 nm segments along the strand_points
    strand_points_segs = [strand_points[0]]
    points_itp_trace = copy.deepcopy(strand_points[1:])
    seg_length = 2
    while points_itp_trace:
        distances = [np.linalg.norm(points_itp_trace[i] - strand_points_segs[-1]) for i in range(0, len(points_itp_trace))]
        for i in range(0, len(distances)):
            if distances[i] >= seg_length:
                direction = points_itp_trace[i] - points_itp_trace[i - 1]
                line_coords = [np.array([points_itp_trace[i - 1][0] + factor * direction[0],
                                         points_itp_trace[i - 1][1] + factor * direction[1]])
                               for factor in np.linspace(0, 1, 11)]
                distances_line_coords = [np.linalg.norm(line_coords[j] - points_itp_trace[0])
                                         for j in range(0, len(line_coords))]
                index = np.argmin(abs(np.asarray(distances_line_coords) - seg_length))
                strand_points_segs.append(strand_points_segs[-1] + seg_length * (line_coords[index] - strand_points_segs[-1])/np.linalg.norm(line_coords[index] - strand_points_segs[-1]))
                break
        del points_itp_trace[0:i+1]

    return strand_points_segs, failed


def find_best_position(interp_function, next_position, strand=None, line_points=11, thetas_num=13):
    """ At a given point, use lines to interpolate the height_profile and compute the best position along those """
    if strand is None:
        unit_vector = np.array([1., 0.])
        thetas = np.linspace(0, np.pi - np.pi / 18, thetas_num)
    else:
        unit_vector = strand[-1] - strand[-2]
        thetas = np.linspace(np.pi*3.5 / 9, np.pi*5.5 / 9, thetas_num)

    # Define the lines and coordinates/height values along those lines
    line_straight = [np.array([next_position[0] + factor * unit_vector[0], next_position[1] + factor * unit_vector[1]])
                     for factor in np.linspace(-2.0, 2.0, line_points)]
    lines_coords = [np.asarray([rotate_vector(point - next_position, theta) + next_position for point in line_straight])
                    for theta in thetas]
    height_values = [interp_function(line[:, 0], line[:, 1]) for line in lines_coords]

    # Use the mean of the slopes of each line to find the one with the maximum slope, this one should be the best
    slopes = [np.mean(abs(np.asarray([heights[i + 1] - heights[i] for i in range(0, len(heights) - 1)])))
              for heights in height_values]
    # Apply weights: the perpendicular line should be weighed the most and the ones rotated the most have less weight
    if strand is not None:
        weights = np.linspace(0.5, 1.0, int(np.ceil(thetas_num/2)))
        weights = np.hstack((weights, weights[::-1][1::]))
        slopes = weights * np.asarray(slopes)
    line_best = lines_coords[np.argmax(slopes)]
    height_values_best = height_values[np.argmax(slopes)]


    # Fit Gaussian to the best height values and use peak position to calculate new position
    try:
        coeff, var_matrix = curve_fit(gauss_function, np.linspace(0, 10, line_points), height_values_best, p0=[1., 5., 2.])
        best_position = line_best[0, :] + coeff[1]/10 * (line_best[-1, :] - line_best[0, :])
    except:
        best_position = next_position

    if strand is not None:
        angle = angle_between(strand[-1] - strand[-2], best_position - strand[-1])
        if angle >= 20:
            overrotation = (angle - 19.99)/180 * np.pi
            best_position_new = strand[-1] + rotate_vector(best_position - strand[-1], overrotation)
            if angle_between(strand[-1] - strand[-2], best_position_new - strand[-1]) >= 20:
                best_position_new = strand[-1] + rotate_vector(best_position - strand[-1], -overrotation)
            best_position = copy.deepcopy(best_position_new)

    return best_position


def find_next_position(interp_function, curr_position, direction_guess, line_points=11, thetas_num=13):

    direction_guess = direction_guess/np.linalg.norm(direction_guess)
    line_straight = [np.array([curr_position[0] + factor * direction_guess[0],
                               curr_position[1] + factor * direction_guess[1]])
                     for factor in np.linspace(0, 2, line_points)]
    thetas = np.linspace(-np.pi/6, np.pi/6, thetas_num)
    lines_coords = [np.asarray([rotate_vector(point - curr_position, theta) + curr_position for point in line_straight])
                    for theta in thetas]
    height_values = [interp_function(line[:, 0], line[:, 1]) for line in lines_coords]

    # Calculate slopes, the way it is done they can be negative as well thus add the negative threshold before weighing
    slopes = [np.mean(np.asarray([heights[i + 1] - heights[i] for i in range(0, len(heights) - 1)]))
              for heights in height_values]
    slopes = np.asarray(slopes) + abs(max(slopes))
    weights = np.linspace(0.5, 1.0, int(np.ceil(thetas_num/2)))
    weights = np.hstack((weights, weights[::-1][1::]))
    theta = thetas[np.argmax(slopes * weights)]
    direction = rotate_vector(direction_guess, theta)

    return curr_position + direction/np.linalg.norm(direction)


def get_interp_function(mol_original, position, grid_size=5, failed=False):

    r = int(position[0])
    c = int(position[1])
    if mol_original[r, c] <= 0.05:
        failed = True
    rr = np.linspace(r - grid_size, r + grid_size, 2 * grid_size + 1)
    cc = np.linspace(c - grid_size, c + grid_size, 2 * grid_size + 1)
    cc, rr = np.meshgrid(cc, rr)
    height_grid = copy.deepcopy(mol_original[r - grid_size: r + grid_size + 1,
                                c - grid_size: c + grid_size + 1])

    return interp.Rbf(rr, cc, height_grid, function='linear'), failed
