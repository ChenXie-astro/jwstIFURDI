import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy import optimize
from jwstIFURDI.toolbox import plot_scaling_factor, plot_throughput, plot_spectrum_errorbar, plot_multiple_spectrum_errorbar_old, plot_four_spectrum_errorbar_old

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
    
    # if disk_extract_mask.shape[0]>2:
    #     for z in range(nz):
    #         disk_residual_cube[z] = disk_residual_cube[z] * disk_extract_mask[z]
    #         residual_cube_disk_subtracted[z] = residual_cube_disk_subtracted[z]* disk_extract_mask[z]
    # else:
    disk_residual_cube = disk_residual_cube * disk_extract_mask
    residual_cube_disk_subtracted = residual_cube_disk_subtracted* disk_extract_mask
    disk_model_raw = disk_model_raw  * disk_extract_mask      
    disk_spec_r2 = np.nansum(disk_residual_cube * weight_map , axis=(1,2)) #- np.nansum(halo_cube_r2_corrected, axis=(1,2))  
    noise_region_1D = np.nansum(residual_cube_disk_subtracted* weight_map , axis=(1,2))

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
    # throughput_factor_1D = (disk_spec) / (disk_model_1D + residual_spec_disk_sub) 
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
    output_disk_spectrum[:,1] = output_disk_spec_r2_corr_throughput_corr *  2.352e-5 *0.1**2  
    output_disk_spectrum[:,2] = new_std_1D * 2.352e-5 *0.1**2    # 1 MJy/sr = 2.352e-5 Jy/arcsec^2


    # if output the average surface brighness within the aperture
    # output_disk_spectrum[:,0] = z_wave
    # output_disk_spectrum[:,1] = disk_spec_r2  / PSF_convolution_ratio  / throughput_factor_1D *  2.352e-5  /num_region #*0.1**2  
    # output_disk_spectrum[:,2] = new_std_1D *stellar_spec_1D  * 2.352e-5 /num_region#*0.1**2    # 1 MJy/sr = 2.352e-5 Jy/arcsec^2



    ########################################
    # remove channels affected by outliers
    output_disk_spectrum[712:714,:] = np.ones((2, 3)) * (output_disk_spectrum[711,:])
    output_disk_spectrum[715,:] = np.ones((1, 3)) * (output_disk_spectrum[716,:])
    output_disk_spectrum[597:599,:] = np.ones((2, 3)) * (output_disk_spectrum[596,:])

    throughput_factor_1D[712:714] = np.ones((2)) * (throughput_factor_1D[711]) 
    throughput_factor_1D[715] = np.ones((1)) * (throughput_factor_1D[716]) 
    throughput_factor_1D[597:599 ] = np.ones((2)) * (throughput_factor_1D[711])

    # for outmost region that is also affected by outliers 
    if extract_region == 'outx2_E' or extract_region == 'outx2':
        output_disk_spectrum[297:302,:] = np.ones((5, 3)) * (output_disk_spectrum[296,:]) 
        throughput_factor_1D[297:302] = np.ones((5)) * (throughput_factor_1D[296]) 
 
    
    return output_disk_spectrum, throughput_factor_1D



def extract_spec_no_stellar_color(disk_residual_cube, residual_cube_disk_subtracted, disk_model_raw, disk_extract_mask,  stellar_spec_1D, weight_map, PSF_convolution_ratio, z_wave, extract_region):
    
    # if disk_extract_mask.shape[0]>2:
    #     for z in range(nz):
    #         disk_residual_cube[z] = disk_residual_cube[z] * disk_extract_mask[z]
    #         residual_cube_disk_subtracted[z] = residual_cube_disk_subtracted[z]* disk_extract_mask[z]
    # else:
    disk_residual_cube = disk_residual_cube * disk_extract_mask
    residual_cube_disk_subtracted = residual_cube_disk_subtracted* disk_extract_mask
    disk_model_raw = disk_model_raw  * disk_extract_mask      
    disk_spec_r2 = np.nansum(disk_residual_cube * weight_map , axis=(1,2)) #- np.nansum(halo_cube_r2_corrected, axis=(1,2))  
    noise_region_1D = np.nansum(residual_cube_disk_subtracted* weight_map , axis=(1,2))

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
    # throughput_factor_1D = (disk_spec) / (disk_model_1D + residual_spec_disk_sub) 
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


    # if output the average surface brighness within the aperture
    # output_disk_spectrum[:,0] = z_wave
    # output_disk_spectrum[:,1] = disk_spec_r2  / PSF_convolution_ratio  / throughput_factor_1D *  2.352e-5  /num_region #*0.1**2  
    # output_disk_spectrum[:,2] = new_std_1D *stellar_spec_1D  * 2.352e-5 /num_region#*0.1**2    # 1 MJy/sr = 2.352e-5 Jy/arcsec^2



    ########################################
    # remove channels affected by outliers
    output_disk_spectrum[712:714,:] = np.ones((2, 3)) * (output_disk_spectrum[711,:])
    output_disk_spectrum[715,:] = np.ones((1, 3)) * (output_disk_spectrum[716,:])
    output_disk_spectrum[597:599,:] = np.ones((2, 3)) * (output_disk_spectrum[596,:])

    throughput_factor_1D[712:714] = np.ones((2)) * (throughput_factor_1D[711]) 
    throughput_factor_1D[715] = np.ones((1)) * (throughput_factor_1D[716]) 
    throughput_factor_1D[597:599 ] = np.ones((2)) * (throughput_factor_1D[711])

    # for outmost region that is also affected by outliers 
    if extract_region == 'outx2_E' or extract_region == 'outx2':
        output_disk_spectrum[297:302,:] = np.ones((5, 3)) * (output_disk_spectrum[296,:]) 
        throughput_factor_1D[297:302] = np.ones((5)) * (throughput_factor_1D[296]) 
 
    
    return output_disk_spectrum, throughput_factor_1D




if __name__ == "__main__":

    # directory path 
    sci_target_name = 'HD181327_IFU_align'
    root = '/Users/sxie/Desktop/JWST/ms_post_processing'
    path_model = root + '/anadisk_modeling/output_cubes/'
    savepath = root + "/reflectance_measurement/"

    # input data cubes 
    sci_cube_raw, header_sci = fits.getdata(root + '/centering/sci_cube_expend_{1}_aligned.fits'.format(sci_target_name, sci_target_name), header=True)
    ref_cube_raw = fits.getdata(root + '/centering/ref_cube_expend_{1}_aligned.fits'.format(sci_target_name, sci_target_name))
    # Apparent stellar photosphere model specturm 
    stellar_spec_1D = fits.getdata(root + '/stellar_model_spec/BT-Settl_6400K_logg4.5_meta_-0.0.fits') # HD181327 host star
    # stellar_spec_1D = fits.getdata(root + '/stellar_model_spec/BT-Settl_6400K_logg4.5_meta_-0.0_mean_combined.fits')[1] # HD181327 host star
    nz, ny, nx = sci_cube_raw.shape
    z_wave = np.arange(header_sci['CRVAL3'], header_sci['CRVAL3']+ (nz)*header_sci['CDELT3'], header_sci['CDELT3'], )

    # input disk models 
    mcmc_papa = fits.getdata(path_model + '{0}_mcmc_para.fits'.format(sci_target_name))
    chi2v_scaling = fits.getdata(path_model + '{0}_chi2v_scaling.fits'.format(sci_target_name))
    residual_cube = fits.getdata(path_model + '{0}_residual_cube_after_FM.fits'.format(sci_target_name))
    disk_model_raw = fits.getdata(path_model + '{0}_best_fit_disk_model.fits'.format(sci_target_name))

    fits.writeto(savepath + '{0}_residual_cube_after_FM_mean.fits'.format(sci_target_name), np.nanmean(residual_cube, axis=0), header_sci, overwrite=True )

    # input spike masks 
    mask = fits.getdata(root + '/make_disk_mask/spike_mask_{0}.fits'.format(sci_target_name))
    mask_cube =  fits.getdata(root+ '/anadisk_modeling/masks/{0}_mask_cube.fits'.format(sci_target_name))

    # creating weight map and spike mask
    ref_radius = 81.656
    radius_map = fits.getdata(root + '/r2_scale/HD181327_IFU_align_radius_shfit_ceter_56pix.fits')
    weight_map = r2_correction_map_2D(radius_map, ref_radius, function_index=2)
    fits.writeto(savepath + '{0}_r2_correction_weight_map_2D_ref_radius_{1}au.fits'.format(sci_target_name, ref_radius), weight_map, overwrite=True )
    # spike_mask = make_spike_mask(sci_cube_raw, 15)
    
    # masking the inner region (<10 pix) 
    x_center = nx//2
    y_center = ny//2
    for y in range(ny):
        for x in range(nx):
            if (abs(x - x_center)) ** 2 + (abs(y - y_center)) ** 2 < 10 ** 2:
                mask[y,x] = np.nan


    ##  performing post-processing
    new_res_cube = np.zeros(sci_cube_raw.shape)
    new_std_cube = np.zeros(sci_cube_raw.shape)
    throughput_cube_3D = np.zeros(sci_cube_raw.shape)

    spectral_res = 4  # pixels in spectral direction
    number_of_sampling = 25
    band_width = spectral_res * number_of_sampling 


    # replace the outliers in the disk-free residual cubes for the uncertainty estimation
    # outliers that were unremoved by pre-processing
    residual_cube[712:714, :,:] = np.ones((2,ny,nx))* (residual_cube[711]+residual_cube[714])/2


    plot_scaling_factor(z_wave[20:-21], chi2v_scaling[20:-21, 0], 'scaling_factor (disk-free)',  savepath, ) # scaling factor obtained using disk-free cube
    new_res_cube, scaling_factor, cost = single_frame_sub(sci_cube_raw, ref_cube_raw, mask_cube)
    plot_scaling_factor(z_wave[20:-21], scaling_factor[20:-21], 'scaling_factor',  savepath, )
    # new_res_cube = np.zeros(sci_cube_raw.shape)

    for z in range(nz):
        # new_res_cube[z] = sci_cube_raw[z] -  chi2v_scaling[z, 0] * ref_cube_raw[z]
        # throughput_cube_3D[z] = (new_res_cube[z] - residual_cube[z] )/disk_model_raw[z]

        # output unccertainty map
        if z <= band_width//2:
            new_std_cube[z] = np.nanstd(residual_cube[0:band_width], axis=0)
        elif (nz-z) <= band_width//2:
            new_std_cube[z] = np.nanstd(residual_cube[(nz-band_width): nz], axis=0)
        else: 
            new_std_cube[z] = np.nanstd(residual_cube[(z-band_width//2) : (z+band_width//2) ], axis=0)

    new_std_cube_sigle =  np.nanstd(residual_cube[0:-1], axis=0) 

    # SN_cube = np.zeros(sci_cube_raw.shape)
    # for z in range(nz):
    #     SN_cube[z] = new_res_cube[z]/new_std_cube[z]

    throughput_cube_3D[throughput_cube_3D==0] = np.nan
    fits.writeto(savepath + '{0}_throughput_cube_3D.fits'.format(sci_target_name), throughput_cube_3D, header_sci, overwrite=True )

    # fits.writeto(savepath + 'residual_single_frame_sub_RDI_SN_cube_{0}.fits'.format(sci_target_name), SN_cube, header_sci, overwrite=True )
    fits.writeto(savepath + 'residual_single_frame_sub_RDI_{0}.fits'.format(sci_target_name), new_res_cube, header_sci, overwrite=True )
    fits.writeto(savepath + 'residual_single_frame_sub_RDI_{0}_spike_masked.fits'.format(sci_target_name), new_res_cube*mask, header_sci, overwrite=True )
    fits.writeto(savepath + 'residual_single_frame_sub_RDI_median_{0}.fits'.format(sci_target_name), np.nanmedian(new_res_cube, axis=0), header_sci, overwrite=True )
    fits.writeto(savepath + 'residual_single_frame_sub_RDI_mean_{0}.fits'.format(sci_target_name), np.nanmean(new_res_cube, axis=0), header_sci, overwrite=True )
    fits.writeto(savepath + 'residual_single_frame_sub_RDI_mean_{0}_spike_masked.fits'.format(sci_target_name), np.nanmean(new_res_cube, axis=0)*mask, header_sci, overwrite=True )
    # plot_scaling_factor(chi2v_scaling[:, 0],  'scaling_factor_{0}'.format(sci_target_name), header_sci, savepath, r'Scaling factor ($f$ )')

    fits.writeto(savepath + '{0}_residual_cube_after_FM_spike_masked.fits'.format(sci_target_name), residual_cube*mask, header_sci, overwrite=True )
    fits.writeto(savepath + '{0}_residual_cube_after_FM_mean_spike_masked.fits'.format(sci_target_name), np.nanmean(residual_cube, axis=0)*mask, header_sci, overwrite=True )
    fits.writeto(savepath + '{0}_uncertainty_cube.fits'.format(sci_target_name), new_std_cube, header_sci, overwrite=True )
    fits.writeto(savepath + '{0}_uncertainty_cube_2D.fits'.format(sci_target_name), new_std_cube_sigle, header_sci, overwrite=True )



    ##########################################
    ## extracting disk reflectance spectrum ##
    ##########################################


    disk_extract_mask_in_E = fits.getdata(root + '/make_disk_mask/masks/disk_source_mask_onlydisk_inner_E.fits')
    disk_extract_mask_in_W = fits.getdata(root + '/make_disk_mask/masks/disk_source_mask_onlydisk_inner_W.fits')
    disk_extract_mask_in = fits.getdata(root + '/make_disk_mask/masks/disk_source_mask_onlydisk_inner.fits')

    disk_extract_mask_mid_E = fits.getdata(root + '/make_disk_mask/masks/disk_source_mask_onlydisk_mid_E.fits')
    disk_extract_mask_mid_W = fits.getdata(root + '/make_disk_mask/masks/disk_source_mask_onlydisk_mid_W.fits')
    disk_extract_mask_mid = fits.getdata(root + '/make_disk_mask/masks/disk_source_mask_onlydisk_mid.fits')

    disk_extract_mask_out_E = fits.getdata(root + '/make_disk_mask/masks/disk_source_mask_onlydisk_out_E.fits')
    disk_extract_mask_out_W = fits.getdata(root + '/make_disk_mask/masks/disk_source_mask_onlydisk_out_W.fits')
    disk_extract_mask_out = fits.getdata(root + '/make_disk_mask/masks/disk_source_mask_onlydisk_out.fits')

    disk_extract_mask_mid_out = fits.getdata(root + '/make_disk_mask/masks/disk_source_mask_onlydisk_mid_out.fits')
    disk_extract_mask_in_mid = fits.getdata(root + '/make_disk_mask/masks/disk_source_mask_onlydisk_in_mid.fits')

    disk_extract_mask_outx2_E = fits.getdata(root + '/make_disk_mask/masks/disk_source_mask_onlydisk_outx2_E.fits')
    disk_extract_mask_outx2_W = fits.getdata(root + '/make_disk_mask/masks/disk_source_mask_onlydisk_outx2_W.fits')
    disk_extract_mask_outx2 = fits.getdata(root + '/make_disk_mask/masks/disk_source_mask_onlydisk_outx2.fits')



    PSF_convolution_ratio_in_E = fits.getdata(root + '/PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_inner_E.fits'.format(sci_target_name)) 
    PSF_convolution_ratio_in_W = fits.getdata(root + '/PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_inner_W.fits'.format( sci_target_name)) 
    PSF_convolution_ratio_in = fits.getdata(root + '/PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_inner.fits'.format(sci_target_name)) 

    PSF_convolution_ratio_mid_E = fits.getdata(root + '/PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_mid_E.fits'.format(sci_target_name)) 
    PSF_convolution_ratio_mid_W = fits.getdata(root + '/PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_mid_W.fits'.format(sci_target_name)) 
    PSF_convolution_ratio_mid = fits.getdata(root + '/PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_mid.fits'.format(sci_target_name)) 

    PSF_convolution_ratio_out_E = fits.getdata(root + '/PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_out_E.fits'.format(sci_target_name)) 
    PSF_convolution_ratio_out_W = fits.getdata(root + '/PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_out_W.fits'.format(sci_target_name)) 
    PSF_convolution_ratio_out = fits.getdata(root + '/PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_out.fits'.format(sci_target_name)) 

    PSF_convolution_ratio_mid_out = fits.getdata(root + '/PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_mid_out.fits'.format(sci_target_name)) 
    PSF_convolution_ratio_in_mid = fits.getdata(root + '/PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_in_mid.fits'.format(sci_target_name)) 

    PSF_convolution_ratio_outx2_E = fits.getdata(root + '/PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_outx2_E.fits'.format(sci_target_name)) 
    PSF_convolution_ratio_outx2_W = fits.getdata(root + '/PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_outx2_W.fits'.format(sci_target_name)) 
    PSF_convolution_ratio_outx2 = fits.getdata(root + '/PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_outx2.fits'.format(sci_target_name)) 



    disk_residual_cube = new_res_cube


    disk_spectrum_in_E, throughput_factor_1D_in_E = extract_spec(disk_residual_cube, residual_cube, disk_model_raw, disk_extract_mask_in_E,  stellar_spec_1D, weight_map, PSF_convolution_ratio_in_E, z_wave, 'in_E')
    disk_spectrum_in_W, throughput_factor_1D_in_W = extract_spec(disk_residual_cube, residual_cube, disk_model_raw, disk_extract_mask_in_W,  stellar_spec_1D, weight_map, PSF_convolution_ratio_in_W, z_wave, 'in_W')
    disk_spectrum_in, throughput_factor_1D_in = extract_spec(disk_residual_cube, residual_cube, disk_model_raw, disk_extract_mask_in,  stellar_spec_1D, weight_map, PSF_convolution_ratio_in, z_wave, 'in')

    disk_spectrum_mid_E, throughput_factor_1D_mid_E = extract_spec(disk_residual_cube, residual_cube, disk_model_raw, disk_extract_mask_mid_E,  stellar_spec_1D, weight_map, PSF_convolution_ratio_mid_E, z_wave, 'mid_E')
    disk_spectrum_mid_W, throughput_factor_1D_mid_W = extract_spec(disk_residual_cube, residual_cube, disk_model_raw, disk_extract_mask_mid_W,  stellar_spec_1D, weight_map, PSF_convolution_ratio_mid_W, z_wave, 'mid_W')
    disk_spectrum_mid, throughput_factor_1D_mid = extract_spec(disk_residual_cube, residual_cube, disk_model_raw, disk_extract_mask_mid,  stellar_spec_1D, weight_map, PSF_convolution_ratio_mid, z_wave, 'mid')

    disk_spectrum_out_E, throughput_factor_1D_out_E = extract_spec(disk_residual_cube, residual_cube, disk_model_raw, disk_extract_mask_out_E,  stellar_spec_1D, weight_map, PSF_convolution_ratio_out_E, z_wave, 'out_E')
    disk_spectrum_out_W, throughput_factor_1D_out_W = extract_spec(disk_residual_cube, residual_cube, disk_model_raw, disk_extract_mask_out_W,  stellar_spec_1D, weight_map, PSF_convolution_ratio_out_W, z_wave, 'out_W')
    disk_spectrum_out, throughput_factor_1D_out = extract_spec(disk_residual_cube, residual_cube, disk_model_raw, disk_extract_mask_out,  stellar_spec_1D, weight_map, PSF_convolution_ratio_out, z_wave, 'out')

    disk_spectrum_mid_out, throughput_factor_1D_mid_out = extract_spec(disk_residual_cube, residual_cube, disk_model_raw, disk_extract_mask_mid_out,  stellar_spec_1D, weight_map, PSF_convolution_ratio_mid_out, z_wave, 'mid_out')
    disk_spectrum_in_mid, throughput_factor_1D_in_mid = extract_spec(disk_residual_cube, residual_cube, disk_model_raw, disk_extract_mask_in_mid,  stellar_spec_1D, weight_map, PSF_convolution_ratio_in_mid, z_wave, 'in_mid')

    # 2024-06-27
    disk_spectrum_in_mid_no_color_corr, throughput_factor_1D_in_mid_no_color_corr = extract_spec_no_stellar_color(disk_residual_cube, residual_cube, disk_model_raw, disk_extract_mask_in_mid,  stellar_spec_1D, weight_map, PSF_convolution_ratio_in_mid, z_wave, 'in_mid')

    disk_spectrum_outx2_E, throughput_factor_1D_outx2_E = extract_spec(disk_residual_cube, residual_cube, disk_model_raw, disk_extract_mask_outx2_E,  stellar_spec_1D, weight_map, PSF_convolution_ratio_outx2_E, z_wave, 'outx2_E')
    disk_spectrum_outx2_W, throughput_factor_1D_outx2_W = extract_spec(disk_residual_cube, residual_cube, disk_model_raw, disk_extract_mask_outx2_W,  stellar_spec_1D, weight_map, PSF_convolution_ratio_outx2_W, z_wave, 'outx2_W')
    disk_spectrum_outx2, throughput_factor_1D_outx2 = extract_spec(disk_residual_cube, residual_cube, disk_model_raw, disk_extract_mask_outx2,  stellar_spec_1D, weight_map, PSF_convolution_ratio_outx2, z_wave, 'outx2')


    # output disk reflectance spectrum
    fits.writeto(savepath+'{0}_corrected_disk_spectrum_inner.fits'.format(sci_target_name), disk_spectrum_in, overwrite=True)
    fits.writeto(savepath+'{0}_corrected_disk_spectrum_mid.fits'.format(sci_target_name), disk_spectrum_mid, overwrite=True)
    fits.writeto(savepath+'{0}_corrected_disk_spectrum_outer.fits'.format(sci_target_name), disk_spectrum_out, overwrite=True)
    fits.writeto(savepath+'{0}_corrected_disk_spectrum_outerx2.fits'.format(sci_target_name), disk_spectrum_outx2, overwrite=True)
    fits.writeto(savepath+'{0}_corrected_disk_spectrum_mid_out.fits'.format(sci_target_name), disk_spectrum_mid_out, overwrite=True)
    fits.writeto(savepath+'{0}_corrected_disk_spectrum_in_mid.fits'.format(sci_target_name), disk_spectrum_in_mid, overwrite=True)

    fits.writeto(savepath+'{0}_disk_spectrum_in_mid_no_color_corr.fits'.format(sci_target_name), disk_spectrum_in_mid_no_color_corr, overwrite=True)

    # fits.writeto(savepath+'{0}_corrected_disk_spectrum_inner_without_Tcorr.fits'.format(sci_target_name), disk_spectrum_in, overwrite=True)
    # fits.writeto(savepath+'{0}_corrected_disk_spectrum_mid_without_Tcorr.fits'.format(sci_target_name), disk_spectrum_mid, overwrite=True)
    # fits.writeto(savepath+'{0}_corrected_disk_spectrum_outer_without_Tcorr.fits'.format(sci_target_name), disk_spectrum_out, overwrite=True)
    # fits.writeto(savepath+'{0}_corrected_disk_spectrum_outerx2_without_Tcorr.fits'.format(sci_target_name), disk_spectrum_outx2, overwrite=True)
    # fits.writeto(savepath+'{0}_corrected_disk_spectrum_mid_out_without_Tcorr.fits'.format(sci_target_name), disk_spectrum_mid_out, overwrite=True)
    # fits.writeto(savepath+'{0}_corrected_disk_spectrum_in_mid_without_Tcorr.fits'.format(sci_target_name), disk_spectrum_in_mid, overwrite=True)


    fits.writeto(savepath+'{0}_corrected_disk_throughpu_1D_inner.fits'.format(sci_target_name), throughput_factor_1D_in, overwrite=True)
    fits.writeto(savepath+'{0}_corrected_disk_throughpu_1D_mid.fits'.format(sci_target_name), throughput_factor_1D_mid, overwrite=True)
    fits.writeto(savepath+'{0}_corrected_disk_throughpu_1D_outer.fits'.format(sci_target_name), throughput_factor_1D_out, overwrite=True)
    fits.writeto(savepath+'{0}_corrected_disk_throughpu_1D_outerx2.fits'.format(sci_target_name), throughput_factor_1D_outx2, overwrite=True)
    fits.writeto(savepath+'{0}_corrected_disk_throughpu_1D_mid_out.fits'.format(sci_target_name), throughput_factor_1D_mid_out, overwrite=True)
    fits.writeto(savepath+'{0}_corrected_disk_throughpu_1D_in_mid.fits'.format(sci_target_name), throughput_factor_1D_in_mid, overwrite=True)


    plot_throughput(z_wave[:-3], throughput_factor_1D_in[:-3], 'throughput_correction_1D_inner',  savepath, )
    plot_throughput(z_wave[:-3], throughput_factor_1D_mid[:-3], 'throughput_correction_1D_middle',  savepath, )
    plot_throughput(z_wave[:-3], throughput_factor_1D_out[:-3], 'throughput_correction_1D_outer',  savepath, )
    plot_throughput(z_wave[:-3], throughput_factor_1D_outx2[:-3], 'throughput_correction_1D_outer x2',  savepath, )
    plot_throughput(z_wave[:-3], throughput_factor_1D_mid_out[:-3], 'throughput_correction_1D_mid+out',  savepath, )
    plot_throughput(z_wave[:-3], throughput_factor_1D_in_mid[:-3], 'throughput_correction_1D_in+mid',  savepath, )



    stats = ()
    legend = 'disk spectrum'
    plotname = '/HD181327_disk_spectrum'

    plot_multiple_spectrum_errorbar_old(disk_spectrum_in[:,1], disk_spectrum_in[:,2], disk_spectrum_in_E[:,1], disk_spectrum_in_E[:,2], disk_spectrum_in_W[:,1], disk_spectrum_in_W[:,2], z_wave, '{0}_disk_spectrum_comparison_inner'.format(sci_target_name), savepath, 'E+W (inner ~85 au)', 'E side (81-91 au)', 'W side (81-90 au)')
    plot_multiple_spectrum_errorbar_old(disk_spectrum_mid[:,1], disk_spectrum_mid[:,2], disk_spectrum_mid_E[:,1], disk_spectrum_mid_E[:,2], disk_spectrum_mid_W[:,1], disk_spectrum_mid_W[:,2], z_wave, '{0}_disk_spectrum_comparison_mid'.format(sci_target_name), savepath, 'E+W (mid ~97 au)', 'E side (93-105 au)', 'W side (90-103 au)')
    plot_multiple_spectrum_errorbar_old(disk_spectrum_out[:,1], disk_spectrum_out[:,2], disk_spectrum_out_E[:,1], disk_spectrum_out_E[:,2], disk_spectrum_out_W[:,1], disk_spectrum_out_W[:,2], z_wave, '{0}_disk_spectrum_comparison_outer'.format(sci_target_name), savepath, 'E+W (outer ~113 au)', 'E side (102-123 au)', 'W side (101-122 au)')
    plot_multiple_spectrum_errorbar_old(disk_spectrum_outx2[:,1], disk_spectrum_outx2[:,2], disk_spectrum_outx2_E[:,1], disk_spectrum_outx2_E[:,2], disk_spectrum_outx2_W[:,1], disk_spectrum_outx2_W[:,2], z_wave, '{0}_disk_spectrum_comparison_outerx2'.format(sci_target_name), savepath, 'E+W (outer ~134 au)', 'E side (121-147 au)', 'W side (119-147 au)')


    plot_four_spectrum_errorbar_old(disk_spectrum_in[:,1], disk_spectrum_in[:,2], disk_spectrum_mid[:,1], disk_spectrum_mid[:,2], disk_spectrum_out[:,1], disk_spectrum_out[:,2], disk_spectrum_outx2[:,1], disk_spectrum_outx2[:,2], header_sci, '{0}_disk_spectrum_comparison_four_regions'.format(sci_target_name), savepath, 'region 1 (~85 au)', 'region 2 (~97 au)', 'region3 (~113 au)', 'region 4 (~134 au)')
    plot_multiple_spectrum_errorbar_old(disk_spectrum_in[:,1], disk_spectrum_in[:,2], disk_spectrum_mid[:,1], disk_spectrum_mid[:,2], disk_spectrum_out[:,1], disk_spectrum_out[:,2], z_wave[:], '{0}_disk_spectrum_comparison_slides'.format(sci_target_name), savepath, 'inner (~85 au)', 'middle (~97 au)', 'outer (~113 au)')




