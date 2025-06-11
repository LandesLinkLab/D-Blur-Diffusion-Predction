#Import external packages
import numpy as np
from scipy.ndimage import maximum_filter, label, center_of_mass
import math


#Import internal functions: 
from utils.data_loader import load_file



def re_group_exp(result_save,w,h,f, patch_size,overlap,channel ):


    # Start to crop 
    step_size = math.floor(patch_size * (1 - overlap))    # The steps with the overlay included.
    # initiate the x y cropping index
    x_ind = []
    y_ind = []

    # crop in x: 
    for start in range(0, w - patch_size + 1, step_size):
        x_ind.append(start)

    # Take care of the edge: 
    if (x_ind[-1] + patch_size ) < w:
        x_ind.append(w - patch_size)


    # crop in y: 
    for start in range(0, h - patch_size + 1, step_size):
        y_ind.append(start)

    if (y_ind[-1] + patch_size ) < h:
        y_ind.append(h - patch_size)

    # devide in frames, same as xy but no overlay:
    f_ind = []

    for start in range(0, f - channel + 1, channel ):
        f_ind.append(start)
    if (f_ind[-1] + channel ) < f: 
        f_ind.append(f - channel) 
    index = 0

    reconstructed = np.zeros([w,h,14,len(f_ind)])

    # Create a weight array to track how many times each pixel is written
    weights = np.zeros((w, h,14,len(f_ind)))

    for xi, i in enumerate(x_ind):
        for yj, j in enumerate(y_ind):
            for k in range(len(f_ind)):
                # Get the patch

                result_file = result_save + 'result_' + str(index) + '.npz'
                pred = load_file(result_file)
                #patch = torch.sigmoid(pred[index]).to(dtype=torch.float32).permute(1,2,0).squeeze().detach().cpu().numpy()
                reconstructed[i:i + patch_size, j:j + patch_size, :,k] += pred
                weights[i:i + patch_size, j:j + patch_size, :,k] += 1
                index += 1
    
    # Average the overlaps
    #reconstructed_mask /= weights
    reconstructed /= weights
    reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())
    return reconstructed


def find_particle_locations(gaussian_map, neighborhood_size = 3, threshold=0.1):
    
    """
    Recover particle locations from a Gaussian probability map.
    
    Args:
    - gaussian_map: 3D numpy array of the Gaussian probability map.
    - neighborhood_size: Size of the neighborhood to consider for local maxima.
    - threshold: Minimum probability to consider a peak as valid.

    Returns:
    - refined_coords: List of tuples with (x, y) subpixel particle locations.
    """
    
    # Step 1: Find local max 
    neighborhood = np.ones((neighborhood_size, neighborhood_size,neighborhood_size))
    local_max = (gaussian_map == maximum_filter(gaussian_map, footprint=neighborhood))
    
    # Step 2: Apply threshold to filter noise
    threshold_mask = gaussian_map > threshold
    valid_peaks = local_max & threshold_mask
    
    # Step 3: Label connected regions (for subpixel refinement)
    labeled_array, num_features = label(valid_peaks)
    
    # Step 4: Subpixel refinement using center of mass
    refined_coords = []
    for i in range(1, num_features + 1):  # Label indices start at 1
        mask = (labeled_array == i)
        if mask.any():
            y, x, z = center_of_mass(gaussian_map, mask)  # Compute center of mass
            refined_coords.append([x, y, z])
    
    return refined_coords
