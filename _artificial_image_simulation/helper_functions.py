import copy
import numpy as np
import math

import skimage.morphology as morph
import skimage.filters as filters
import skimage.util as util
from PolymerCpp.helpers import getCppWLC
from PolymerCpp.helpers import getCppWLC2D


def create_simulated_nuc(bare_DNA_length, persis_length, rise_per_bp, uw_angle,
                         simulation_type='2D',
                         arm_length_1=106, arm_length_2=233):

    nuc_diameter = 8/rise_per_bp
    img_simulated = np.zeros((2*bare_DNA_length, 2*bare_DNA_length))

    # Calculate the start point of the chain simulation for the two arms, offset fixed for the stable arm
    row_offset_fixed = np.sin(((180 - 66.5) / 2) / 180 * np.pi) * (nuc_diameter - 1/rise_per_bp) - 1
    col_offset_fixed = np.cos(((180 - 66.5) / 2) / 180 * np.pi) * (nuc_diameter - 1/rise_per_bp) - 1
    row_offset_uw = np.sin(((180 - 66.5) / 2 - uw_angle) / 180 * np.pi) * (nuc_diameter - 1/rise_per_bp) - 1
    col_offset_uw = np.cos(((180 - 66.5) / 2 - uw_angle) / 180 * np.pi) * (nuc_diameter - 1/rise_per_bp) - 1

    # Simulate the arms
    if simulation_type == '2D':
        arm1 = np.asarray(getCppWLC2D(arm_length_1, persis_length))
        arm2 = np.asarray(getCppWLC2D(arm_length_2, persis_length))
    elif simulation_type == '3D':
        arm1 = np.asarray(getCppWLC(arm_length_1, persis_length))
        arm2 = np.asarray(getCppWLC(arm_length_2, persis_length))

    # Create a rotation matrix to rotate the simulated arm pixels
    # theta = np.radians((angle_between_arms - 120) / 2)
    theta_arm1 = np.radians(-47.7/2)
    theta_arm2 = np.radians(-47.7/2 + uw_angle)

    # Mark pixels in the image
    c, s = np.cos(-theta_arm1), np.sin(-theta_arm1)
    R = np.array(((c, -s), (s, c)))
    arm1_pixels_rotated = [np.dot(R, point) for point in arm1[:, 0:2]]
    for row, col in arm1_pixels_rotated:
        img_simulated[int(np.round(row + row_offset_fixed + bare_DNA_length)),
                      int(np.round(col - col_offset_fixed + bare_DNA_length - 1))] = 1
    c, s = np.cos(theta_arm2), np.sin(theta_arm2)
    R = np.array(((c, -s), (s, c)))
    arm2_pixels_rotated = [np.dot(R, point) for point in arm2[:, 0:2]]
    for row, col in arm2_pixels_rotated:
        img_simulated[int(np.round(row + row_offset_uw + bare_DNA_length)),
                      int(np.round(col + col_offset_uw + bare_DNA_length - 1))] = 1

    return img_simulated


def create_simulated_dna(bare_DNA_length, persis_length,
                         simulation_type='2D'):

    img_simulated = np.zeros((2*bare_DNA_length, 2*bare_DNA_length))

    # Simulate the arms
    if simulation_type == '2D':
        dna = np.asarray(getCppWLC2D(bare_DNA_length, persis_length))
    elif simulation_type == '3D':
        dna = np.asarray(getCppWLC(bare_DNA_length, persis_length))
    pixels = dna[:, 0:2]
    for row, col in pixels:
        img_simulated[int(np.round(row + bare_DNA_length - 1)),
                      int(np.round(col + bare_DNA_length - 1))] = 1

    return img_simulated


def apply_convolution(img_simulated, dilation_iterations, gaussian_sigma, bare_DNA_length, rise_per_bp):

    img_convolved = copy.deepcopy(img_simulated)
    # Dilate the pixels to get 2 nm wide DNA
    for i in range(0, dilation_iterations):
       img_convolved = morph.binary_dilation(img_convolved)

    # Set DNA height to 1.2
    img_convolved = img_convolved.astype(float)
    img_convolved[img_convolved != 0] = 1.3

    nuc_diameter = 8/rise_per_bp

    # Mark the nucleosome core pixels and set height to 1.8
    for row in range(int(bare_DNA_length - (nuc_diameter+5)), int(bare_DNA_length + (nuc_diameter+5))):
        for col in range(int(bare_DNA_length - (nuc_diameter+5)), int(bare_DNA_length + (nuc_diameter+5))):
            if np.sqrt((bare_DNA_length - row - 1) ** 2 + (bare_DNA_length - col - 1) ** 2) <= nuc_diameter:
                img_convolved[row, col] = 1.8

    # Add random noise
    img_noisy = util.random_noise(np.empty_like(img_convolved))
    img_convolved = img_convolved + img_noisy

    # Convolve image with a Gaussian and some noise
    img_convolved = filters.gaussian(img_convolved, gaussian_sigma)

    return img_convolved
