import os
import numpy as np
from time import time
from pathlib import Path
import diskmap
from astropy.io import fits
from jwstIFURDI.centering import IFU_centering
from jwstIFURDI.cal_PSF_convolution_effect import calculate_psf_covolution_ratio
from jwstIFURDI.toolbox import shift, plot_scaling_factor, plot_throughput, plot_multiple_spectrum_errorbar_old
from jwstIFURDI.form_cubes import r2_correction_map_2D, single_frame_sub, extract_spec



if __name__ == "__main__":
    timeStamps = [time()]
	# Setup data paths	
    sci_target_name = 'HD181327_IFU_align'
    root = '/Users/sxie/Desktop/example/'

    # Step 1:
    perform_centering = True
    
    # Step 2:
    perform_RDI = True

    # Step 3:
    cal_PSF_convolution_effect = True

    # Step 4:
    creating_r2_scaling_map = True

    # Step 5:
    calculate_uncertainty_cube_from_MCMC_output = True

    # Step 6:
    extract_disk_spectrum = True


    if perform_centering:
        print('########  performing centering ########')
        sci_filename = root + 'data/HD181327_newoutput_prism-clear_s3d.fits'
        ref_filename = root + 'data/iotmic_newoutput_prism-clear_s3d.fits'
        savepath = root +  'centering/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        ########################################
        #####   parameters in centering    ##### 
        ########################################
        # in the case of HD181327
        y_center = 55 
        x_center = 55 
        new_img_size_x = 110 # x_center will be at new_img_size_x//2 
        new_img_size_y = 110 # y_center will be at new_img_size_y//2 
        filter_size = 25
        channel_shortest = 0
        channel_longest = 500
        outer_mask_radius = 40

        v3PA = 131 # IFU_align
        theta_ifu_sci = [v3PA-30, v3PA, v3PA+30, v3PA+90 ] # HD 181327
        theta_ifu_ref = theta_ifu_sci
        aligned_sci_cube, aligned_ref_cube_rotated = IFU_centering(sci_filename, ref_filename, savepath,  sci_target_name, theta_ifu_sci, theta_ifu_ref, x_center, y_center, new_img_size_x, new_img_size_y, filter_size, channel_shortest, channel_longest, )
        print('########  centering completed  ########')


    if perform_RDI:
        print('########  performing RDI PSF subtraction  ########')
        """
        Note: 
        """
        # directory path 
        # path_model = root + '/anadisk_modeling/output_cubes/'
        savepath = root + "/reflectance_measurement/"
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        # input data cubes 
        sci_cube_raw, header_sci = fits.getdata(root + 'centering/sci_cube_expend_{1}_aligned.fits'.format(sci_target_name, sci_target_name), header=True)
        ref_cube_raw = fits.getdata(root + 'centering/ref_cube_expend_{1}_aligned.fits'.format(sci_target_name, sci_target_name))

        nz, ny, nx = sci_cube_raw.shape
        z_wave = np.arange(header_sci['CRVAL3'], header_sci['CRVAL3']+ (nz)*header_sci['CDELT3'], header_sci['CDELT3'], )

        # input spike masks 
        spike_mask = fits.getdata(root + 'data/masks/spike_mask_{0}.fits'.format(sci_target_name))
        mask_cube =  fits.getdata(root+ 'data/masks/{0}_mask_cube.fits'.format(sci_target_name))  # remaning annular regions used in fitting the scaling factor in RDI

        # masking the inner region (<10 pix) 
        x_center = nx//2
        y_center = ny//2
        for y in range(ny):
            for x in range(nx):
                if (abs(x - x_center)) ** 2 + (abs(y - y_center)) ** 2 < 10 ** 2:
                    spike_mask[y,x] = np.nan

        ##  performing RDI PSF subtraction
        new_res_cube = np.zeros(sci_cube_raw.shape)
        new_res_cube, scaling_factor, cost = single_frame_sub(sci_cube_raw, ref_cube_raw, mask_cube)
        plot_scaling_factor(z_wave[20:-21], scaling_factor[20:-21], 'scaling_factor',  savepath, )

        # output residual disk cube
        fits.writeto(savepath + 'residual_single_frame_sub_RDI_{0}.fits'.format(sci_target_name), new_res_cube, header_sci, overwrite=True )
        fits.writeto(savepath + 'residual_single_frame_sub_RDI_{0}_spike_masked.fits'.format(sci_target_name), new_res_cube*spike_mask, header_sci, overwrite=True )
        fits.writeto(savepath + 'residual_single_frame_sub_RDI_mean_combined_{0}_spike_masked.fits'.format(sci_target_name), np.nanmean(new_res_cube, axis=0)*spike_mask, header_sci, overwrite=True )
        print('########  RDI completed  ########')


    if cal_PSF_convolution_effect:
        print('########  calculating PSF convolution effect  ########')
        savepath = root + 'PSF_convolution/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        #######################################
        ##### input disk model parameters ##### 
        #######################################
        # # disk model parameters:
        # g1 = 0.33; alpha_in = 16.2; alpha_out = -4.5;  flux = 2.3e6
        # see MMB_anadisk_model.py for details 
        x_all = [0.33, 16.2, -4.5, 2.3e6]

        #####################################
        ##### input empirical PSF model ##### 
        #####################################
        psf_filename = root + 'data/HD181327_psf_masked.fits'
 
        #######################################
        ##### input scicube and disk mask ##### 
        #######################################
        sci_filename = root +  'centering/sci_cube_expend_HD181327_IFU_align_aligned.fits'
        scicube, sci_header = fits.getdata(sci_filename, header=True) # provide the header

        #############################################################################################
        ######   calculating the PSF convolution effect for all spectral extracting regions    ######
        #############################################################################################
        # path to the files.
        dir_name= root + 'data/masks/'
        #get all the uncal files in a list
        #should only be the nrs_1 files
        paths = Path(dir_name).glob('disk_source_mask_onlydisk_*.fits')
        disk_extracting_region_file_path_list = []
        # iterating over all files
        for path in paths:
            disk_extracting_region_file_path_list.append(str(path))

        for disk_extract_mask in disk_extracting_region_file_path_list:
            extracting_region = disk_extract_mask.split('/')[-1][26:-5]
            PSF_convolution_ratio = calculate_psf_covolution_ratio(x_all, psf_filename, savepath, disk_extract_mask, sci_header, sci_target_name, extracting_region)

        print('########  calculation completed  ########')


    if creating_r2_scaling_map:
        print('########  creating the r2 scaling map for illumination correction  ########')
        """
        the diskmap package was used (https://diskmap.readthedocs.io/en/latest/index.html)
        """

        savepath = root + 'r2_scale/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        ##############################################
        ####### input HD181327 disk parameters #######
        ##############################################
        mapping = diskmap.DiskMap(fitsfile=root + 'centering/sci_cube_expend_{0}_aligned.fits'.format(sci_target_name),
                        pixscale=0.10435,
                        inclination=30.2,
                        # pa = 99.1-111.2294
                        pos_angle=99.1-111.2294, # 111.2294 is the angular offest between the detector frame and the ture north
                        distance=47.72,
                        image_type='total')
        mapping.map_disk(power_law=(0., 0.04, 1.),  # ~4 au scale height at 86 au.
                radius=(1., 500., 500))
        mapping.write_output(filename= savepath + '{0}'.format(sci_target_name))
 
        """
        The output of diskmap has a 1 pix offset compared to the image center, which is the star position. 
        Shifting the image using 'roll' funtion is necessary
        Visual inspection is recommended. 
        """
        radius_img = fits.getdata(savepath + 'HD181327_IFU_align_radius.fits') # not aligned
        scat_img = fits.getdata(savepath + 'HD181327_IFU_align_scat_angle.fits') # not  aligned
        shift_offset = (1, 1)
        shift_value = np.round(shift_offset).astype(int)
        shifted_radius_img = shift(radius_img, shift_value, method='roll')
        shifted_scat_img = shift(scat_img, shift_value, method='roll')

        # output final results: the stellocentric distance map and the scattering angle map
        fits.writeto(savepath + 'HD181327_IFU_align_radius_aligned.fits', shifted_radius_img, overwrite=True) # aligned
        fits.writeto(savepath + 'HD181327_IFU_align_scat_angle_aligned.fits', shifted_scat_img, overwrite=True) # aligned

        ref_radius = 81.656
        radius_map = fits.getdata(root + 'r2_scale/HD181327_IFU_align_radius_aligned.fits')
        weight_map = r2_correction_map_2D(radius_map, ref_radius, function_index=2)
        fits.writeto(savepath + '{0}_r2_correction_weight_map_2D_ref_radius_{1}au.fits'.format(sci_target_name, ref_radius), weight_map, overwrite=True )
        print('########  radius map created  ########')


    if calculate_uncertainty_cube_from_MCMC_output:
        """
        Note: running this step requires performing the throughput estimation using MCMC_Code_fixed_disk.py beforehand.
        Input: MCMC results.

        Output: uncertainty cubes and disk-free residual cubes.
        Note: the output uncertainty cube was then used for MCMC throughput estimation in an iterative way.
        """
        path_model = root + '/anadisk_modeling/output_cubes/'
        if not os.path.exists(path_model):
            os.makedirs(path_model)

        band_width = 100 # number of spectral channels used in calculating std in the spectral direction

        residual_cube = fits.getdata(path_model + '{0}_residual_cube_after_FM.fits'.format(sci_target_name))
        nz, ny, nx = residual_cube.shape
        # replace the outliers in the disk-free residual cubes for the uncertainty estimation
        # outliers that were unremoved by pre-processing
        residual_cube[712:714, :,:] = np.ones((2,ny,nx))* (residual_cube[711]+residual_cube[714])/2

        spike_mask = fits.getdata(root + 'data/masks/spike_mask_{0}.fits'.format(sci_target_name))
        # masking the inner region (<10 pix) 
        x_center = nx//2
        y_center = ny//2
        for y in range(ny):
            for x in range(nx):
                if (abs(x - x_center)) ** 2 + (abs(y - y_center)) ** 2 < 10 ** 2:
                    spike_mask[y,x] = np.nan

        new_std_cube = np.zeros(residual_cube.shape)
        for z in range(nz):
            if z <= band_width//2:
                new_std_cube[z] = np.nanstd(residual_cube[0:band_width], axis=0)
            elif (nz-z) <= band_width//2:
                new_std_cube[z] = np.nanstd(residual_cube[(nz-band_width): nz], axis=0)
            else: 
                new_std_cube[z] = np.nanstd(residual_cube[(z-band_width//2) : (z+band_width//2) ], axis=0)

        # output uncertainty cube for MCMC disk modeling and a disk-free residual cube with spikes masked
        fits.writeto(root + 'anadisk_modeling/{0}_residual_cube_after_FM_spike_masked.fits'.format(sci_target_name), residual_cube*spike_mask, overwrite=True )
        fits.writeto(root + 'anadisk_modeling/{0}_uncertainty_cube.fits'.format(sci_target_name), new_std_cube, overwrite=True )



    if extract_disk_spectrum:
        print('########  extracting disk spectra at different stellocentric distances  ########')
        """
        Once we are satisfied with the disk model we built in Step #5 and MCMC disk modeling, we can start to extract the disk reflectance spectrum. 
        """
        savepath = root + "reflectance_measurement/"
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        # input residual cube and science cube header 
        sci_cube_raw, header_sci = fits.getdata(root + 'centering/sci_cube_expend_{0}_aligned.fits'.format(sci_target_name), header=True)
        nz, ny, nx = sci_cube_raw.shape
        z_wave = np.arange(header_sci['CRVAL3'], header_sci['CRVAL3']+ (nz)*header_sci['CDELT3'], header_sci['CDELT3'], )

        new_res_cube = fits.getdata(root + 'reflectance_measurement/residual_single_frame_sub_RDI_{0}.fits'.format(sci_target_name)) # output from Step 2
        disk_residual_cube = new_res_cube

        # Apparent stellar photosphere model spectrum (stellar color correction)
        stellar_spec_1D = fits.getdata(root + 'data/BT-Settl_6400K_logg4.5_meta_-0.0.fits') # HD181327 host star

        # creating weight map for illumination correction (r2 scaling)
        ref_radius = 81.656
        weight_map = fits.getdata(root + 'r2_scale/{0}_r2_correction_weight_map_2D_ref_radius_{1}au.fits'.format(sci_target_name, ref_radius) ) 

        # input best-fit disk model for performing throughput correction and uncertainty estimation
        path_model = root + 'anadisk_modeling/output_cubes/'
        disk_model_raw = fits.getdata(path_model + '{0}_best_fit_disk_model.fits'.format(sci_target_name))
        residual_cube = fits.getdata(path_model + '{0}_residual_cube_after_FM.fits'.format(sci_target_name))
        # replace the outliers in the disk-free residual cubes for the uncertainty estimation
        # outliers that were unremoved by pre-processing
        residual_cube[712:714, :,:] = np.ones((2,ny,nx))* (residual_cube[711]+residual_cube[714])/2

            
        ##########################################
        ## extracting disk reflectance spectrum ##
        ##########################################
        """
        HD181327:
        Three extracting regions on both sides (i.e., East and West) of the disks 
        Inner region: 80-90 au
        Middle region: 90-105 au
        Outer region: 105-120 au
        Inner + Middle region: 80-105 au
        Middle + Outer region: 90-120 au
        """
        disk_extract_mask_in_E = fits.getdata(root + 'data/masks/disk_source_mask_onlydisk_inner_E.fits')
        disk_extract_mask_in_W = fits.getdata(root + 'data/masks/disk_source_mask_onlydisk_inner_W.fits')
        disk_extract_mask_in = fits.getdata(root + 'data/masks/disk_source_mask_onlydisk_inner.fits')

        disk_extract_mask_mid_E = fits.getdata(root + 'data/masks/disk_source_mask_onlydisk_mid_E.fits')
        disk_extract_mask_mid_W = fits.getdata(root + 'data/masks/disk_source_mask_onlydisk_mid_W.fits')
        disk_extract_mask_mid = fits.getdata(root + 'data/masks/disk_source_mask_onlydisk_mid.fits')

        disk_extract_mask_out_E = fits.getdata(root + 'data/masks/disk_source_mask_onlydisk_out_E.fits')
        disk_extract_mask_out_W = fits.getdata(root + 'data/masks/disk_source_mask_onlydisk_out_W.fits')
        disk_extract_mask_out = fits.getdata(root + 'data/masks/disk_source_mask_onlydisk_out.fits')

        disk_extract_mask_mid_out = fits.getdata(root + 'data/masks/disk_source_mask_onlydisk_mid_out.fits')
        disk_extract_mask_in_mid = fits.getdata(root + 'data/masks/disk_source_mask_onlydisk_in_mid.fits')


        PSF_convolution_ratio_in_E = fits.getdata(root + 'PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_inner_E.fits'.format(sci_target_name)) 
        PSF_convolution_ratio_in_W = fits.getdata(root + 'PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_inner_W.fits'.format( sci_target_name)) 
        PSF_convolution_ratio_in = fits.getdata(root + 'PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_inner.fits'.format(sci_target_name)) 

        PSF_convolution_ratio_mid_E = fits.getdata(root + 'PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_mid_E.fits'.format(sci_target_name)) 
        PSF_convolution_ratio_mid_W = fits.getdata(root + 'PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_mid_W.fits'.format(sci_target_name)) 
        PSF_convolution_ratio_mid = fits.getdata(root + 'PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_mid.fits'.format(sci_target_name)) 

        PSF_convolution_ratio_out_E = fits.getdata(root + 'PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_out_E.fits'.format(sci_target_name)) 
        PSF_convolution_ratio_out_W = fits.getdata(root + 'PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_out_W.fits'.format(sci_target_name)) 
        PSF_convolution_ratio_out = fits.getdata(root + 'PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_out.fits'.format(sci_target_name)) 

        PSF_convolution_ratio_mid_out = fits.getdata(root + 'PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_mid_out.fits'.format(sci_target_name)) 
        PSF_convolution_ratio_in_mid = fits.getdata(root + 'PSF_convolution/PSF_covolution_correction/{0}_PSF_convolution_ratio_in_mid.fits'.format(sci_target_name)) 


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


        # output disk reflectance spectrum
        fits.writeto(savepath+'{0}_corrected_disk_spectrum_inner.fits'.format(sci_target_name), disk_spectrum_in, overwrite=True)
        fits.writeto(savepath+'{0}_corrected_disk_spectrum_mid.fits'.format(sci_target_name), disk_spectrum_mid, overwrite=True)
        fits.writeto(savepath+'{0}_corrected_disk_spectrum_outer.fits'.format(sci_target_name), disk_spectrum_out, overwrite=True)
        fits.writeto(savepath+'{0}_corrected_disk_spectrum_mid_out.fits'.format(sci_target_name), disk_spectrum_mid_out, overwrite=True)
        fits.writeto(savepath+'{0}_corrected_disk_spectrum_in_mid.fits'.format(sci_target_name), disk_spectrum_in_mid, overwrite=True)

        fits.writeto(savepath+'{0}_corrected_disk_throughpu_1D_inner.fits'.format(sci_target_name), throughput_factor_1D_in, overwrite=True)
        fits.writeto(savepath+'{0}_corrected_disk_throughpu_1D_mid.fits'.format(sci_target_name), throughput_factor_1D_mid, overwrite=True)
        fits.writeto(savepath+'{0}_corrected_disk_throughpu_1D_outer.fits'.format(sci_target_name), throughput_factor_1D_out, overwrite=True)
        fits.writeto(savepath+'{0}_corrected_disk_throughpu_1D_mid_out.fits'.format(sci_target_name), throughput_factor_1D_mid_out, overwrite=True)
        fits.writeto(savepath+'{0}_corrected_disk_throughpu_1D_in_mid.fits'.format(sci_target_name), throughput_factor_1D_in_mid, overwrite=True)

        # making plots for inspections
        plot_throughput(z_wave[:-3], throughput_factor_1D_in[:-3], 'throughput_correction_1D_inner',  savepath, )
        plot_throughput(z_wave[:-3], throughput_factor_1D_mid[:-3], 'throughput_correction_1D_middle',  savepath, )
        plot_throughput(z_wave[:-3], throughput_factor_1D_out[:-3], 'throughput_correction_1D_outer',  savepath, )
        plot_throughput(z_wave[:-3], throughput_factor_1D_mid_out[:-3], 'throughput_correction_1D_mid+out',  savepath, )
        plot_throughput(z_wave[:-3], throughput_factor_1D_in_mid[:-3], 'throughput_correction_1D_in+mid',  savepath, )

        stats = ()
        legend = 'disk spectrum'
        plotname = '/HD181327_disk_spectrum'

        plot_multiple_spectrum_errorbar_old(disk_spectrum_in[:,1], disk_spectrum_in[:,2], disk_spectrum_in_E[:,1], disk_spectrum_in_E[:,2], disk_spectrum_in_W[:,1], disk_spectrum_in_W[:,2], z_wave, '{0}_disk_spectrum_comparison_inner'.format(sci_target_name), savepath, 'E+W (inner ~85 au)', 'E side (81-91 au)', 'W side (81-90 au)')
        plot_multiple_spectrum_errorbar_old(disk_spectrum_mid[:,1], disk_spectrum_mid[:,2], disk_spectrum_mid_E[:,1], disk_spectrum_mid_E[:,2], disk_spectrum_mid_W[:,1], disk_spectrum_mid_W[:,2], z_wave, '{0}_disk_spectrum_comparison_mid'.format(sci_target_name), savepath, 'E+W (mid ~97 au)', 'E side (93-105 au)', 'W side (90-103 au)')
        plot_multiple_spectrum_errorbar_old(disk_spectrum_out[:,1], disk_spectrum_out[:,2], disk_spectrum_out_E[:,1], disk_spectrum_out_E[:,2], disk_spectrum_out_W[:,1], disk_spectrum_out_W[:,2], z_wave, '{0}_disk_spectrum_comparison_outer'.format(sci_target_name), savepath, 'E+W (outer ~113 au)', 'E side (102-123 au)', 'W side (101-122 au)')
        plot_multiple_spectrum_errorbar_old(disk_spectrum_in[:,1], disk_spectrum_in[:,2], disk_spectrum_mid[:,1], disk_spectrum_mid[:,2], disk_spectrum_out[:,1], disk_spectrum_out[:,2], z_wave[:], '{0}_disk_spectrum_comparison_slides'.format(sci_target_name), savepath, 'inner (~85 au)', 'middle (~97 au)', 'outer (~113 au)')
        print('########  spectra extraction completed  ########')
 

    ###########################################
    timeStamps.append(time())
    totalTime = timeStamps[-1]-timeStamps[0]
    print('-- Total Processing time: ', totalTime, ' s')
    print('')
            




