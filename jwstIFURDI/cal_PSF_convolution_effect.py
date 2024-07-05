#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from time import time
from astropy.io import fits
from scipy.signal import fftconvolve
from jwstIFURDI.toolbox import imshift, plot_psf_correction_factor
from jwstIFURDI.MMB_anadisk_model import generateModel


# %% FUNCTIONS

def calculate_psf_covolution_ratio(x_all, psf_filename, savepath, disk_extract_mask, sci_header, sci_target_name, extracting_region):

    model = generateModel(x_all) 
    nz = 941  # PRISM model
    convolved_models = np.zeros((nz, model.shape[1], model.shape[0]))
    psf = fits.getdata(psf_filename)
    disk_extract_region_mask = fits.getdata(disk_extract_mask) 


    # psf = psf_shift
    if convolved_models.ndim == 3:
        for z in range(psf.shape[0]):
            convolved_models[z] = fftconvolve(model, psf[z], mode='same')
    elif convolved_models.ndim == 2:
        convolved_models = fftconvolve(model, psf, mode='same')

    fits.writeto(savepath+'disk_model_2D.fits', model, overwrite=True)
    fits.writeto(savepath+'disk_model_convolved_3D_no_aligned.fits', convolved_models, overwrite=True)


    """
    The input empirical NIRSpec PSF has an 0.5 pix offset in the center.
    """
    offset_x_sci = -0.5
    offset_y_sci = -0.5
    convolved_models_shift = np.zeros((psf.shape))
    for z in range(nz):
        convolved_models_shift[z] = imshift(convolved_models[z].astype(float), [offset_x_sci, offset_y_sci], method = 'fourier', )
    fits.writeto(savepath+'disk_model_convolved_3D_aligned.fits', convolved_models_shift, overwrite=True)


    ###### runing a sanity check  ######
    """
    test result indicates that fft can preserve the flux. 
    """
    offset_x_sci = +0.5
    offset_y_sci = +0.5
    convolved_models_shift_back = np.zeros((psf.shape))
    for z in range(nz):
        convolved_models_shift_back[z] = imshift( convolved_models_shift[z].astype(float), [offset_x_sci, offset_y_sci], method = 'fourier', )
    fits.writeto(savepath+'disk_model_convolved_3D_shifted_back_check.fits', convolved_models_shift_back, overwrite=True)
    fits.writeto(savepath+'disk_model_convolved_3D_fft_shiftting_reisudals.fits', convolved_models_shift_back-convolved_models, overwrite=True)
    ####################################    



    disk_regions_model = model  *  disk_extract_region_mask
    disk_regions_convolved = convolved_models_shift  *  disk_extract_region_mask
    print(disk_regions_model.shape)

    disk_regions_model[disk_regions_model==0] = np.nan
    disk_regions_convolved[disk_regions_convolved==0] = np.nan

    if os.path.exists(savepath+'/sanity_check/'):
        pass
    else:
        os.makedirs(savepath+'/sanity_check/')
        os.makedirs(savepath+'/PSF_covolution_correction/')

    PSF_convolution_ratio = np.nansum(disk_regions_convolved, axis=(1,2)) / np.nansum(disk_regions_model, axis=(0,1,2))
    fits.writeto(savepath + '/sanity_check/{0}_extracted_disk_regions_{1}.fits'.format(sci_target_name, extracting_region), disk_regions_convolved, sci_header, overwrite= True)

    #####  making plots  ####
    plot_psf_correction_factor(PSF_convolution_ratio, sci_header, '/PSF_covolution_correction/{0}_PSF_convolution_ratio_{1}'.format(sci_target_name, extracting_region), savepath, '#EA8379', legend = 'PSF convolution ratio')
    ##### saving results ####
    fits.writeto(savepath + '/PSF_covolution_correction/{0}_PSF_convolution_ratio_{1}.fits'.format(sci_target_name, extracting_region), PSF_convolution_ratio, sci_header, overwrite= True)

    return PSF_convolution_ratio 

