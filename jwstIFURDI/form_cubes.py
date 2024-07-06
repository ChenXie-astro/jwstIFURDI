import numpy as np
from scipy import ndimage
from scipy import optimize

####################################################################

def make_spike_mask(input_cube, filter_size):
    nz, ny, nx = input_cube.shape
    median_filtered_cube = np.zeros(input_cube.shape)
    input_cube[np.isnan(input_cube)] = 0
    for z in range(nz):
        median_filtered_cube[z,:,:] = input_cube[z,:,:] - ndimage.median_filter(input_cube[z,:,:], filter_size, mode='nearest')
    med_filtered_combined_img = np.nanmean(median_filtered_cube, axis=0)  # HD181327
    # med_filtered_combined_img = np.nanmedian(median_filtered_cube, axis=0)  # betapic current IFU-sligned cube has outliers! So median combinition
    spike_mask = np.ones((ny,nx))
    spike_mask[med_filtered_combined_img>40] = 0  # HD181327
    # spike_mask[med_filtered_combined_img>100] = 0  # beta pic
    spike_mask[spike_mask==0] = np.nan 

    return spike_mask


def r2_correction_map_2D(input_radius_map, ref_radius, function_index):
    # center is the image center
    #############################
    #    weight_function = r**a
    #############################
    # input 2D array
    a = function_index
    ny, nx = input_radius_map.shape 
    # x_center = nx//2
    # y_center = ny//2
    weight_map = np.ones(input_radius_map.shape)
    for x in range(nx):
        for y in range(ny):
            # r = np.sqrt( (x-x_center)**2 + (y-y_center)**2 ) * pixel_size
            r = (input_radius_map[y,x] / ref_radius)
            weight_map[y,x] = r ** a 

    return weight_map


def single_frame_sub(sci_cube, ref_cube, mask_cube, ):
    
    """
    Returns the residual images by using the single ref frame RDI approch.
    Used for processing JWST/NIRSpec IFU data cube. 

    Args:
        sci_img: science images
        ref_img: reference images with the same shape as the sci_img
        mask: mask

    Returns:
        res_img: the residual images
        nu: the scaling factors 
        cost: minimum cost

    Written: Chen Xie, 2023-10.
    """
    
    def cost_function_subtraction(nu,):
        """
        Returns the vaule of the cost function used for the single ref frame RDI approch.

        Args:
            nu: scaling factor 
        Returns:
            cost

        Written: Chen Xie, 2023-10.

        Note that: 'sci_image' (sci image), 'ref_img' (ref), 'mask_img' (mask) are global variables in this nested function that will be updated in each interation.
        """
        return np.log(np.nansum( ((nu * ref_img  - sci_img) * mask_img)**2  , axis=(0,1)))

    nz, ny, nx = sci_cube.shape
    res_cube = np.zeros(sci_cube.shape)
    scaling_factor = np.zeros((nz))
    cost = np.zeros((nz))
    if mask_cube is None:
        mask_cube = np.ones(sci_cube.shape)
        print('*******   Note: using the entire FoV in scaling the reference image in RDI   *******')

    for z in range(nz):
        nu_0 = np.nansum(sci_cube[z] * mask_cube[z], axis=(0,1)) / np.nansum(ref_cube[z] * mask_cube[z], axis=(0,1)) 
        ref_img = ref_cube[z]
        sci_img = sci_cube[z]
        mask_img = mask_cube[z]

        minimum = optimize.fmin(cost_function_subtraction, nu_0, disp=False)
        scaling_factor[z] = minimum[0]
        res_cube[z] = sci_cube[z] - minimum[0] * ref_cube[z]
        cost[z] = cost_function_subtraction(minimum[0]) 

    return res_cube, scaling_factor, cost


def extract_spec(disk_residual_cube, residual_cube_disk_subtracted, disk_model_raw, disk_extract_mask,  stellar_spec_1D, weight_map, PSF_convolution_ratio, z_wave, extract_region):

    disk_residual_cube = disk_residual_cube * disk_extract_mask
    residual_cube_disk_subtracted = residual_cube_disk_subtracted* disk_extract_mask
    disk_model_raw = disk_model_raw  * disk_extract_mask      
    disk_spec_r2 = np.nansum(disk_residual_cube * weight_map , axis=(1,2)) #- np.nansum(halo_cube_r2_corrected, axis=(1,2))  
    noise_region_1D = np.nansum(residual_cube_disk_subtracted* weight_map , axis=(1,2))

    nz = disk_residual_cube.shape[0] 
    ######  calculate noise  ######
    band_width = 100
    new_std_1D = np.zeros(noise_region_1D.shape)
    for z in range(nz):
        if z <= band_width//2:
            new_std_1D[z] = np.nanstd(noise_region_1D[0:band_width], axis=0)
        elif (nz-z) <= band_width//2:
            new_std_1D[z] = np.nanstd(noise_region_1D[(nz-band_width): nz], axis=0)
        else: 
            new_std_1D[z] = np.nanstd(noise_region_1D[(z-band_width//2) : (z+band_width//2) ], axis=0)

    ######  throughput estimation in 1D  ######
    disk_spec = np.nansum(disk_residual_cube, axis=(1,2))
    residual_spec_disk_sub = np.nansum(residual_cube_disk_subtracted, axis=(1,2))
    disk_model_1D = np.nansum(disk_model_raw, axis=(1,2))
    throughput_factor_1D = (disk_spec-residual_spec_disk_sub) / disk_model_1D 

    ######################################
    # applying corrections
    output_disk_spec_r2_corr_throughput_corr = disk_spec_r2 / stellar_spec_1D / PSF_convolution_ratio  / throughput_factor_1D
    new_std_1D = new_std_1D / stellar_spec_1D / PSF_convolution_ratio  / throughput_factor_1D 

    output_disk_spec_r2_corr_throughput_corr[-1] = np.nan # removing the last channel  
    new_std_1D[-1] = np.nan # removing the last channel  

    output_disk_spectrum = np.zeros((nz, 3))
    if disk_extract_mask.shape[0]>2:
        num_region = np.count_nonzero(disk_extract_mask[0])
    else:
        num_region = np.count_nonzero(disk_extract_mask)

    # if outpur total flux within the aperture in the unit of Jy
    output_disk_spectrum[:,0] = z_wave
    output_disk_spectrum[:,1] = output_disk_spec_r2_corr_throughput_corr *  2.352e-5 *0.1**2   # pixel size 0.1" x 0.1"
    output_disk_spectrum[:,2] = new_std_1D * 2.352e-5 *0.1**2    # 1 MJy/sr = 2.352e-5 Jy/arcsec^2

    ########################################
    # remove channels affected by outliers
    output_disk_spectrum[712:714,:] = np.ones((2, 3)) * (output_disk_spectrum[711,:])
    output_disk_spectrum[715,:] = np.ones((1, 3)) * (output_disk_spectrum[716,:])
    output_disk_spectrum[597:599,:] = np.ones((2, 3)) * (output_disk_spectrum[596,:])

    throughput_factor_1D[712:714] = np.ones((2)) * (throughput_factor_1D[711]) 
    throughput_factor_1D[715] = np.ones((1)) * (throughput_factor_1D[716]) 
    throughput_factor_1D[597:599 ] = np.ones((2)) * (throughput_factor_1D[711])
    
    return output_disk_spectrum, throughput_factor_1D



def extract_spec_no_stellar_color(disk_residual_cube, residual_cube_disk_subtracted, disk_model_raw, disk_extract_mask,  stellar_spec_1D, weight_map, PSF_convolution_ratio, z_wave, extract_region):

    disk_residual_cube = disk_residual_cube * disk_extract_mask
    residual_cube_disk_subtracted = residual_cube_disk_subtracted* disk_extract_mask
    disk_model_raw = disk_model_raw  * disk_extract_mask      
    disk_spec_r2 = np.nansum(disk_residual_cube * weight_map , axis=(1,2)) #- np.nansum(halo_cube_r2_corrected, axis=(1,2))  
    noise_region_1D = np.nansum(residual_cube_disk_subtracted* weight_map , axis=(1,2))

    nz = disk_residual_cube.shape[0] 
    band_width = 100
    ######  calculate noise  ######
    new_std_1D = np.zeros(noise_region_1D.shape)
    for z in range(nz):
        if z <= band_width//2:
            new_std_1D[z] = np.nanstd(noise_region_1D[0:band_width], axis=0)
        elif (nz-z) <= band_width//2:
            new_std_1D[z] = np.nanstd(noise_region_1D[(nz-band_width): nz], axis=0)
        else: 
            new_std_1D[z] = np.nanstd(noise_region_1D[(z-band_width//2) : (z+band_width//2) ], axis=0)

    ######  throughput estimation in 1D  ######
    disk_spec = np.nansum(disk_residual_cube, axis=(1,2))
    residual_spec_disk_sub = np.nansum(residual_cube_disk_subtracted, axis=(1,2))
    disk_model_1D = np.nansum(disk_model_raw, axis=(1,2))
    throughput_factor_1D = (disk_spec-residual_spec_disk_sub) / disk_model_1D 

    ######################################
    # applying corrections
    output_disk_spec_r2_corr_throughput_corr = disk_spec_r2  / PSF_convolution_ratio  / throughput_factor_1D
    new_std_1D = new_std_1D  / PSF_convolution_ratio  / throughput_factor_1D 

    output_disk_spec_r2_corr_throughput_corr[-1] = np.nan # removing the last channel  
    new_std_1D[-1] = np.nan # removing the last channel  

    output_disk_spectrum = np.zeros((nz, 3))
    if disk_extract_mask.shape[0]>2:
        num_region = np.count_nonzero(disk_extract_mask[0])
    else:
        num_region = np.count_nonzero(disk_extract_mask)

    # if outpur total flux within the aperture in the unit of Jy
    output_disk_spectrum[:,0] = z_wave
    output_disk_spectrum[:,1] = output_disk_spec_r2_corr_throughput_corr *  2.352e-5 *0.1**2  
    output_disk_spectrum[:,2] = new_std_1D * 2.352e-5 *0.1**2    # 1 MJy/sr = 2.352e-5 Jy/arcsec^2


    ########################################
    # remove channels affected by outliers
    output_disk_spectrum[712:714,:] = np.ones((2, 3)) * (output_disk_spectrum[711,:])
    output_disk_spectrum[715,:] = np.ones((1, 3)) * (output_disk_spectrum[716,:])
    output_disk_spectrum[597:599,:] = np.ones((2, 3)) * (output_disk_spectrum[596,:])

    throughput_factor_1D[712:714] = np.ones((2)) * (throughput_factor_1D[711]) 
    throughput_factor_1D[715] = np.ones((1)) * (throughput_factor_1D[716]) 
    throughput_factor_1D[597:599 ] = np.ones((2)) * (throughput_factor_1D[711])


    return output_disk_spectrum, throughput_factor_1D



