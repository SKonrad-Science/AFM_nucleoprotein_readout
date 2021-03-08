import numpy as np
import copy

from scipy.interpolate import Rbf
from scipy.spatial.ckdtree import cKDTree
from skimage import restoration


def estimate_tip_from_DNA(analyzed_bare_DNA):
    """
    Estimates the shape of the AFM cantilever tip from a list of analyzed bare DNA objects

    Args:
        analyzed_bare_DNA (list): Analyzed bare DNA objects that contain a Wiggin's trace and the filtered molecule to
                                  estimate the tip shape
    Returns:
        tip_shape: Dict with the tip shape function, an array that represents the tip shape and the tip's excentricity
    """

    rel_coords, heights = [], []
    for bare_DNA in analyzed_bare_DNA:
        trace_wigg = np.asarray(bare_DNA.results['wigg_fwd'])
        trace_fine = trace_finer(trace_wigg)
        pixels, trace_id, rel_coord_list, height_list = rel_coords_to_trace(bare_DNA.mol_filtered, trace_fine)
        rel_coords += list(rel_coord_list)
        heights += list(height_list)

    tip_shape_function, tip_shape_arr, tip_norm_arr = tip_shape_estimation(rel_coords, heights)
    excentricity = tip_excentricity(tip_shape_function)

    tip_shape = {'tip_shape_fct': tip_shape_function,
                 'tip_shape_arr': tip_shape_arr,
                 'tip_norm_arr': tip_norm_arr,
                 'tip_excentricity': excentricity}

    return tip_shape


def trace_finer(trace, steps_per_pixel=10):
    """
    Makes the initial trace that is given to the function more fine grained to increase nearest neighbour accuracy

    Args:
        trace ([N, 2] array): Trace of the analyzed bare DNA according to Wiggin's tracing
        steps_per_pixel (int): Number of steps to interpolate the trace with per pixel
    Returns:
        trace_fine: More fine grained positions of points along the trace
    """

    fine_factor = (np.linalg.norm(trace[:-1] - trace[1:], axis=1).mean() / (1/steps_per_pixel)).astype(int)
    r_fine = np.asarray([np.linspace(trace[i, 0], trace[i + 1, 0], fine_factor)[:-1]
                         for i in range(0, len(trace)-1)]).flatten()
    c_fine = np.asarray([np.linspace(trace[i, 1], trace[i + 1, 1], fine_factor)[:-1]
                         for i in range(0, len(trace)-1)]).flatten()
    trace_fine = np.vstack((r_fine, c_fine)).T
    trace_fine = np.vstack((trace_fine, trace[-1, :]))

    return trace_fine


def rel_coords_to_trace(mol_filtered, trace, distance_limit=5.0):
    """
    Finds the pixels in the image that are within the 'distance_limit' of the 'trace' points. For those pixels the
    relative coordinates to the closest trace point is calculated.

    Args:
        mol_filtered ([N, M] array): Image of the molecule for tip estimation - can also be the unfiltered image
        trace ([N, 2] array): Initial trace of the DNA strand
        distance_limit (float): Maximum distance a pixel can have from the trace to be taken into account
    Returns:
        pixels: Array with row/column coordinates of the pixels within the distance limit from the trace
        trace_id: Int relating each pixel from the 'pixels' array to the point in the 'trace' it is closest to
        relative_coords ([N, 2] array): Relative x and y distances of all pixels from the closest point of the trace
        heights([N, ] array): Height of the image at the position of the pixel
    """

    min_r, min_c = np.floor(trace.min(axis=0) - distance_limit).astype(int).clip(min=0)
    max_r, max_c = np.ceil(trace.max(axis=0) + distance_limit).astype(int).clip(max=mol_filtered.shape)
    pixels_pos = np.mgrid[min_r:max_r, min_c:max_c].reshape([2, -1]).T      # all potential pixels

    # kdTree finds the nearest neighbour between a specific pixel and all trace points
    # Returns distances between pixels and nn and the id of the nn. Distances are inf if bigger than distance_limit
    kdtree = cKDTree(trace)
    distances, trace_id = kdtree.query(pixels_pos, k=1, distance_upper_bound=distance_limit)
    pixels = pixels_pos[distances != np.inf]
    trace_id = trace_id[distances != np.inf]
    rel_coords = pixels - trace[trace_id]

    # mask = (trace_id != 0) & (trace_id != len(trace)-1)
    # pixels, trace_id, rel_coords = pixels[mask], trace_id[mask], rel_coords[mask]

    heights = np.asarray([mol_filtered[pixel[0], pixel[1]] for pixel in pixels])

    return pixels, trace_id, rel_coords, heights


def tip_shape_estimation(rel_coords, heights):
    """
    Estimate the shape of the AFM cantilever tip

    Args:
        rel_coords ([N, 2] array): Contains the relative position of each pixel to the trace
        heights ([N, ] array: Contains the height of each pixel
    Returns:
         tip_shape_fct: The Rbf fitted function to the tip shape
         tip_shape_norm: The normalized shape of the tip from the pixel data
         data_quantity: The normalization array based on pixel occurrences
    """

    # Calculate the four closest pixel coordinates of a point that can lie between pixels
    coords_nn_int = [np.array([[np.floor(coord[0]), np.floor(coord[1])],
                               [np.floor(coord[0]), np.ceil(coord[1])],
                               [np.ceil(coord[0]), np.floor(coord[1])],
                               [np.ceil(coord[0]), np.ceil(coord[1])]]).astype(int) for coord in rel_coords]

    offsets = abs(rel_coords - np.ceil(rel_coords))
    weights_coords_nn = np.asarray([np.array([offset[0] * offset[1],    # x/y dists weighed betw. point and nn pixels
                                              offset[0] * (1 - offset[1]),
                                              (1 - offset[0]) * offset[1],
                                              (1 - offset[0]) * (1 - offset[1])]) for offset in offsets])

    heights = np.vstack((heights, heights, heights, heights)).T
    heights, weights_coords_nn = heights.flatten(), weights_coords_nn.flatten()
    coords_nn_int = np.asarray(coords_nn_int).reshape([-1, 2])

    # Get the tip shape by adding up all heights of the relative positions in a 2D histogram
    max_rel_coords = np.max(coords_nn_int, axis=0)
    min_rel_coords = np.min(coords_nn_int, axis=0)
    bins = [np.arange(low, high + 2) for low, high in zip(min_rel_coords, max_rel_coords)]  # +2 to align tip/hist cent.
    tip_shape = np.histogramdd(coords_nn_int, weights=heights * weights_coords_nn, bins=bins)[0]
    data_quantity = np.histogramdd(coords_nn_int, weights=weights_coords_nn, bins=bins)[0]
    tip_shape_norm = np.divide(tip_shape, data_quantity, out=np.zeros_like(tip_shape), where=data_quantity!=0)

    # Get tip filter function by fitting Rbf to the tip shape of the histogram
    bins = [np.arange(low, high + 1) for low, high in zip(min_rel_coords, max_rel_coords)]  # here only +1 to center it
    rr_mesh, cc_mesh = np.meshgrid(*bins)
    rr_mesh, cc_mesh = rr_mesh.T, cc_mesh.T
    tip_filter_fct = Rbf(rr_mesh, cc_mesh, tip_shape_norm, function='linear')

    return tip_filter_fct, tip_shape_norm, data_quantity


def tip_excentricity(tip_shape_fct):
    """
    Calculates the excentricity of the tip

    Args:
        tip_shape_fct: function that estimated the AFM cantilever tip
    Returns:
        excentricity:
    """
    steps = 101
    linspace = np.linspace(-10, 10, steps)
    x_shape = tip_shape_fct([0]*steps, linspace)
    y_shape = tip_shape_fct(linspace, [0]*steps)
    x_shape -= np.amin(x_shape)
    y_shape -= np.amin(y_shape)
    std1 = np.sqrt((x_shape * linspace ** 2).mean())     # better fit Gaussian or 2D Gaussian -> test?
    std2 = np.sqrt((y_shape * linspace ** 2).mean())

    excentricity = std1/std2

    return excentricity


def decon_mol(mol_filtered, tip_shape):

    mol_decon = copy.deepcopy(mol_filtered)
    mol_decon[mol_decon == 0] = 0.001
    tip = tip_shape['tip_shape_arr']
    mol_decon = restoration.richardson_lucy(mol_decon, tip/np.sum(tip), iterations=10)
    mol_decon[mol_filtered == 0] = 0    # Set Outer parts to zero since deconvolution gives it small non-zero values

    return mol_decon
