"""
Analysis functions
"""

import copy
import numpy as np
import scipy
import scipy.interpolate as interp
from scipy.optimize import curve_fit
from skimage import morphology
from scipy.ndimage import rotate
import math

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
            x any y coordinates of the pixels of the endpoints
        eps_number - int
            Amount of endpoints of the skeleton
        bps_pixels - array
            x any y coordinated of the pixels of the branchpoints
        bps_number - int
            Amount of branchpoints of the skeleton (connected areas where one or more skeleton pixels have > 2 nb)
        pixels_number - int
            Amount of all pixels in the skeleton. Can be used to estimate the length of the structure and thus
            helps to classify whether a structure is too small or too large to be a proper molecule
    """

    mol_skel = copy.deepcopy(mol_skeleton)

    # Calculate the number of non-zero neighbouring pixels for each pixel of the skeleton
    mol_neighbours = np.zeros_like(mol_skel)
    for r, c in zip(*np.where(mol_skel == 1)):
        mol_neighbours[r, c] = np.sum(np.multiply(mol_skel[r - 1:r + 2, c - 1:c + 2], neighbour_matrix))

    # Calculate the number of endpoints
    eps_pixels = np.array(np.where(mol_neighbours == 1)).T
    eps_number = np.shape(np.array(np.where(mol_neighbours == 1)))[1]
    pixels_number = len(mol_skel[mol_skel != 0])

    # Calculate the number of branches
    bps_pixels = np.array(np.where(mol_neighbours >= 3)).T
    mol_branches_conn = copy.deepcopy(mol_neighbours)
    mol_branches_conn[mol_neighbours <= 2] = 0
    mol_branches_conn[mol_branches_conn != 0] = 1
    bps_number = np.amax(morphology.label(mol_branches_conn, connectivity=2))

    skel_dict = {'skel_eps_pixels': eps_pixels,
                 'skel_eps_number': eps_number,
                 'skel_bps_pixels': bps_pixels,
                 'skel_bps_number': bps_number,
                 'skel_pixels_number': pixels_number}

    return skel_dict


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


def wiggins(mol_filtered, seg_length, start, end, mol_type, ell_data=None, failed=False):
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
        lambda: mol_type == 'Nucleosome' and
        np.linalg.norm(ell_data['center'] - curr_point) > np.amax(ell_data['abc'] * 1)
    }
    while any(cond() for cond in conditions):

        if ell_data is None:
            width = 4
            if np.linalg.norm(end_point - curr_point) <= 0.5*seg_length_orig:
                seg_length = 1.5
        else:
            width = 3
            if np.linalg.norm(end_point - curr_point) <= 0.5*seg_length_orig or np.linalg.norm(ell_data['center'] - curr_point) < np.amax(ell_data['abc'] * 1.5):
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
            r_linspace = np.linspace(next_row - width*direction_perp[0], next_row + width*direction_perp[0],
                                     num_interp_values)
            c_linspace = np.linspace(next_col - width*direction_perp[1], next_col + width*direction_perp[1],
                                     num_interp_values)
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


def radius_of_gyration(mol_filtered, pixel_size):
    """ Calculate the center of mass and the radius of gyration of the molecule """
    # Improve to handel the case of Integrase where no core molecule is existent to not give 'nan'
    mol_img = copy.deepcopy(mol_filtered)

    center_of_mass = scipy.ndimage.center_of_mass(mol_img)
    pixel_heights = mol_img[mol_img != 0]
    distances = np.asarray([np.linalg.norm(pixel - center_of_mass) for pixel in np.argwhere(mol_img)])

    # Radius of gyration formula according to Wikipedia
    radius = np.sqrt((np.sum(pixel_heights*(distances**2) / np.sum(pixel_heights))))

    rog_dict = {'rog': radius * pixel_size,
                'com': center_of_mass}
    return rog_dict


def ellipsoid_fct(xy, a, b, c, rot_angle, x0, y0):
    """
    Ellipsoid from https://en.wikipedia.org/wiki/Ellipsoid
    Parametrized x = a*sin(theta)*cos(phi)  (I)
                 y = b*sin(theta)*sin(phi)  (II)
                 z = c*cos(theta)
    Solve (I) and (II) for phi and theta depending on x, y. Then calculate z.
    Rotation of the ellipsoid and shift along x, y plane is applied by rotating the input coordinates by rot_angle
    around the z-axis and afterwards moving the coordinates by constant amounts x0 and y0.
    Args:
        xy:
        a:
        b:
        c:
        rot_angle:
        x0:
        y0:

    Returns:

    """

    # Rotation matrix and rotation of the offsets
    result = []
    rot_z = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                      [np.sin(rot_angle), np.cos(rot_angle)]])
    x0, y0 = rot_z.dot(np.asarray([x0, y0]))

    for x, y in zip(xy[0, :], xy[1, :]):
        # Rotation of the coordinates
        x, y = rot_z.dot(np.asarray([x, y])) - np.array([x0, y0])

        # Calculation of the z_value
        if (b * x) != 0:
            phi = np.arctan((a * y) / (b * x))
        elif (b * x) == 0 and y != 0:
            phi = np.pi / 2
        else:
            phi = 0
        if phi != 0 and -1 <= (y / (b * np.sin(phi))) <= 1:
            theta = np.arccos(y / (b * np.sin(phi)))
        else:
            theta = 0
        if y == 0 and -1 <= (x / a) <= 1:
            z = c * np.sin(np.arccos(x / a))
        else:
            z = c * np.sin(theta)
        result.append(z)

    return np.asarray(result)


def ellipsoid_plot(xy, ellipsoid_coeffs):

    # Calculate the ellipsoid height values
    x_arr, y_arr = xy[0, :], xy[1, :]
    x_mid = (np.amax(x_arr) + np.amin(x_arr))/2
    y_mid = (np.amax(y_arr) + np.amin(y_arr))/2
    a, b, c = ellipsoid_coeffs['abc']
    ell_heights = ellipsoid_fct(xy, a, b, c, rot_angle=0, x0=x_mid, y0=y_mid)
    x_range = int(np.amax(x_arr) - np.amin(x_arr) + 1)
    y_range = int(np.amax(y_arr) - np.amin(y_arr) + 1)
    ell_heights = ell_heights.reshape((x_range, y_range))

    # Rotate - in the fit the coordinates are rotated, here the height array is rotated
    ell_heights_rot = rotate(ell_heights, angle=-ellipsoid_coeffs['rot_angle'] * 180 / np.pi, reshape=False)
    ell_heights_rot[ell_heights_rot < 0.03] = 0     # Rotation interpolates and thus causes slight height changes
    x_center, y_center = ellipsoid_coeffs['center']
    ell_data = {'ell_heights_rot': ell_heights_rot,
                'rr_shifted': x_arr.reshape((x_range, y_range)) - (x_mid - x_center),
                'cc_shifted': y_arr.reshape((x_range, y_range)) - (y_mid - y_center)}

    return ell_data


def ellipsoid_fit(mol_filtered, center_of_mass_core, grid_size=10):

    # Set up fit input
    com_int = np.round(center_of_mass_core).astype(int)
    rr = np.linspace(com_int[0] - grid_size, com_int[0] + grid_size, 2 * grid_size + 1)
    cc = np.linspace(com_int[1] - grid_size, com_int[1] + grid_size, 2 * grid_size + 1)
    cc, rr = np.meshgrid(cc, rr)
    height_grid = copy.deepcopy(mol_filtered[int(np.amin(rr)):int(np.amax(rr)) + 1,
                                int(np.amin(cc)):int(np.amax(cc)) + 1])
    rc_stack = np.vstack((rr.flatten(), cc.flatten()))

    # Fit rotating half ellipsoid
    coeff_start = [5, 5, 2, 0, com_int[0], com_int[1]]
    bounds = ([0, 0, 0, -np.pi / 2, 0, 0], [15, 15, 5, np.pi / 2, np.amax(rr), np.amax(cc)])
    # weights = copy.deepcopy(height_grid)
    # weights[height_grid < 1] = 0.5
    # weights[height_grid > 1] = 1.0
    try:
        coeff, var_matrix = curve_fit(ellipsoid_fct, rc_stack, height_grid.flatten(), p0=coeff_start, bounds=bounds)
    except:
        return {'failed': True}

    ell_coeffs = {'abc': np.asarray([coeff[1], coeff[0], coeff[2]]),   # Had to change order to fit my ellipse orientat.
                  'rot_angle': -coeff[3],
                  'center': coeff[4:6]}
    ell_data = ellipsoid_plot(rc_stack, ell_coeffs)
    ell_data.update(ell_coeffs)

    mol_ellipsoid_cut = copy.deepcopy(mol_filtered)
    indices = np.where(ell_data['ell_heights_rot'] != 0)
    rows_1 = np.floor(ell_data['rr_shifted'][indices]).astype(int)
    cols_1 = np.floor(ell_data['cc_shifted'][indices]).astype(int)
    rows_2 = np.ceil(ell_data['rr_shifted'][indices]).astype(int)
    cols_2 = np.ceil(ell_data['cc_shifted'][indices]).astype(int)
    rows, cols = np.hstack((rows_1, rows_2)), np.hstack((cols_1, cols_2))
    mol_ellipsoid_cut[rows, cols] = 0

    ell_data.update({'mol_ellipsoid_cut': mol_ellipsoid_cut,
                     'ell_indices': (rows, cols)})

    return ell_data


def ellipse_arm_pixel(pixels_arm, ell_data, ell_cutoff=0.6):

    # Define all necessary input parameters
    a, b, c = ell_data['abc']
    a = a * (1 - ell_cutoff ** 2)   # ell_cutoff defines the height that should be reached of the ellipsoid before
    b = b * (1 - ell_cutoff ** 2)   # cutting off the DNA arm
    phi = -ell_data['rot_angle']
    center = ell_data['center']
    last_point = pixels_arm[-1]

    # Parameters of the line that goes through the ellipse
    m = (center[0] - last_point[0]) / (center[1] - last_point[1])
    b_line = center[0] - m * center[1]
    b_line2 = b_line + m * center[1] - center[0]    # Adjust the y-offset to make the ellipse the coordinate center

    # Solve Mitternachtsformula to get the two x values
    q1 = b ** 2 * (np.cos(phi) ** 2 + 2 * m * np.cos(phi) * np.sin(phi) + m ** 2 * np.sin(phi) ** 2) \
         + a ** 2 * (m ** 2 * np.cos(phi) ** 2 - 2 * m * np.cos(phi) * np.sin(phi) + np.sin(phi) ** 2)
    q2 = 2 * b ** 2 * b_line2 * (np.cos(phi) * np.sin(phi) + m * np.sin(phi) ** 2) \
         + 2 * a ** 2 * b_line2 * (m * np.cos(phi) ** 2 - np.cos(phi) * np.sin(phi))
    q3 = b_line2 ** 2 * (b ** 2 * np.sin(phi) ** 2 + a ** 2 * np.cos(phi) ** 2) - a ** 2 * b ** 2

    x1 = (-q2 + np.sqrt(q2 ** 2 - 4 * q1 * q3)) / (2 * q1) + center[1]
    x2 = (-q2 - np.sqrt(q2 ** 2 - 4 * q1 * q3)) / (2 * q1) + center[1]
    y1 = m * x1 + b_line
    y2 = m * x2 + b_line
    intersection_1 = np.array([y1, x1])
    intersection_2 = np.array([y2, x2])

    if np.linalg.norm(intersection_1 - last_point) < np.linalg.norm(intersection_2 - last_point):
        return intersection_1
    else:
        return intersection_2


def angle_between(v1, v2):
    """ Calculate the angle between two vectors """
    # Calculate unit vectors
    v1_unit = v1 / np.linalg.norm(v1)
    v2_unit = v2 / np.linalg.norm(v2)

    # Calculate the dotproduct between the vectors and the angle (rad) afterwards
    dotproduct = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
    angle = np.arccos(dotproduct) * 180 / np.pi
    return angle


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


def rotate_vector(vector, theta):

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    vec_rotated = np.dot(R, vector)

    return vec_rotated
