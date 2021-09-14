import numpy as np
import random

import matplotlib.pyplot as plt
from skimage.measure import block_reduce

import helper_functions as helper
from tqdm import tqdm

# Quick guide:
#

# Simulate image
rise_per_bp = 0.165      # Problem of 0.33 is that I can't get to 1.5 nm/pixel resolution with reduce. Use 0.165 and
# just double the DNA length to still have 486 "real" bp, reduce by (9, 9) -> roughly 1.5 nm/pixel (the resolution of my "real" AFM images)
reduction = 9
bare_DNA_length = (486 - 4) * 2         # -4 segments per side due to dilation (checked for the 6 dil steps)
# The numbers in the brackets denote the bp values that the two arms can have in length for unwrapping of up to 35 bp
# Unwrapping only occurs from one side at the same time - the DNA construct has one 106 bp arm and one 233 bp arm
# when fully wrapped.
arm_length_1 = (106 * 2) - 4            # [106, 106, 106, 106, 106, 106, 106, 106, 233, 233, 233, 233, 233, 233, 233]
arm_length_2 = (268 * 2) - 4            # [233, 238, 243, 248, 253, 258, 263, 268, 111, 116, 121, 126, 131, 136, 141]

bp_uw = 35
persis_length = 40
unwrapping_angle = 155.4        # [0, 22.2, 44.4, 66.6, 88.8, 111, 133.2, 155.4, 22.2, 44.4, 66.6, 88.8, 111, 133.2, 155.4]
# The unwrapping angle increases by 22.2° for 5 bp unwrapping according to crystal structure
dil_iterations = 6              # Dilation is performed to get the DNA to 2 nm width.
# One pixel has 0.165 nm so dilating 6 times into both directions yields 2 nm
gaussian_sigma = 12
number_molecules = 2                                        # number of molecules per line, only even numbers
sim_type = '2D'                                             # WLC dimensionality - '2D' or '3D'
output_file = 'nuc_35_bp_uw.asc'                            # Name of the output file
sim_molecule = 'nucleosome'                                 # Choose which molecule type to simulate 'bare_DNA' or 'nucleosome'

images = []
for i in tqdm(range(0, int((number_molecules**2)))):

    # Random 5 bp unwrapping - This part is for the case that unwrapping can occur from both sides simultaneously
    # predefined_list = [0, 5, 10, 15, 20, 25, 30, 35] # Adjust the numbers based on how many bp are unwrapped - right now 35 bp are unwrapped in total
    # random_index = random.randint(0, len(predefined_list)-1)
    # bp_uw_1 = predefined_list[random_index]
    # bp_uw_2 = bp_uw - bp_uw_1
    # arm_length_1 = ((106 + bp_uw_1) * 2) - 4
    # arm_length_2 = ((233 + bp_uw_2) * 2) - 4

    # Simulate DNA
    if sim_molecule == 'bare_DNA':
        dna_simulated = helper.create_simulated_dna(bare_DNA_length=bare_DNA_length,
                                                    persis_length=int(persis_length/rise_per_bp),
                                                    simulation_type=sim_type)
        dna_convolved = helper.apply_convolution(dna_simulated, dilation_iterations=dil_iterations,
                                                 gaussian_sigma=gaussian_sigma, bare_DNA_length=bare_DNA_length,
                                                 rise_per_bp=rise_per_bp)
        dna_reduced = block_reduce(dna_convolved, block_size=(reduction, reduction), func=np.mean)
        images.append(dna_reduced)

    elif sim_molecule == 'nucleosome':
        nuc_simulated = helper.create_simulated_nuc(bare_DNA_length=bare_DNA_length,
                                                    persis_length=int(persis_length/rise_per_bp),
                                                    rise_per_bp=rise_per_bp,
                                                    uw_angle=unwrapping_angle,
                                                    simulation_type=sim_type,
                                                    arm_length_1=arm_length_1,
                                                    arm_length_2=arm_length_2)
        nuc_convolved = helper.apply_convolution(nuc_simulated, dilation_iterations=dil_iterations,
                                                 gaussian_sigma=gaussian_sigma, bare_DNA_length=bare_DNA_length,
                                                 rise_per_bp=rise_per_bp)
        nuc_reduced = block_reduce(nuc_convolved, block_size=(reduction, reduction), func=np.mean)
        images.append(nuc_reduced)

# Stack images vertically
images_vertical = []
for i in range(0, number_molecules):
    image_vertical = images[i * number_molecules]
    for j in range(1 + i * number_molecules, number_molecules + i * number_molecules):
        image_vertical = np.vstack((image_vertical, images[j]))
    images_vertical.append(image_vertical)

# Stack images horizontally
large_image = images_vertical[0]
for i in range(0, number_molecules - 1):
    large_image = np.hstack((large_image, images_vertical[i+1]))

plt.figure()
plt.imshow(large_image, interpolation='None', cmap='coolwarm')

# Export the data to the file
np.savetxt(output_file, large_image, delimiter='\t', fmt='%-8.6f',
           header='x-pixels = ' + repr(large_image.shape[0]) + '\n' +
                  'y-pixels = ' + repr(large_image.shape[0]) + '\n' +
                  'x-length = ' + repr(int(large_image.shape[0]*rise_per_bp*reduction)) + '\n' +
                  'y-length = ' + repr(int(large_image.shape[0]*rise_per_bp*reduction)))
