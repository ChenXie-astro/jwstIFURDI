
# -*- coding: utf-8 -*-
import numpy as np
import debrisdiskfm
# %% FUNCTIONS



def generateModel(x_all):
    distance = 47.72 #pc
    g_1 = x_all[0] #forward scattering coefficient (from 0 to 1)
    pa = 99.1-111.2294 # position angle of the disk in degrees, counted from +y axis counterclockwise
    # 111.2294 is the angular offset between the detector frame and the true north
    
    inc = 30.2  # inclination of the disk in degrees, counted from face-on (0 degree) to edge-on (90 degree)
    alpha_in = x_all[1] #rising component r^alphaIn (value > 0)

    #critical radius, not far from where the rise and decline joints
    r_c = 78.9766 # 1.655*47.72  from milli+2024

    alpha_out = x_all[2] # decreasing component r^alphaOut (value < 0)
    flux = x_all[3] # factor to scale the overall brightness

    r_in = 0 # cutoff radius from 0 to r_c
    r_out = 300 # cutoff radius from r_c to +infinity
    
    dx = 0
    dy = 0
    #g_2 = x_all[9] #backward scattering coefficient (from -1 to 0)
    #weight1 = x_all[10] #weight factor of the forward scattering coefficient (from 0 to 1) —> backward scattering weight automatically will be 1 minus this
    # g_2 = x_all[9] #backward scattering coefficient (from -1 to 0)
    # w1 = x_all[10] 

    los_factor = 1.             # Keep fixed. Line Of Sight factor: make the 3rd dimension Nx larger to compensate projection effects
    asp_ratio = 0.04            # Keep fixed for now. Aspect ratio for the scale height. 0.04 expected for debris disks (Thebault 2009)

    spf_list = [lambda i: debrisdiskfm.anadisk_sum_mask_MMB.hgg_phase_function(i, [g_1], rayleigh_pol = False)]
    # spf_list = [lambda i: debrisdiskfm.anadisk_sum_mask_MMB.hgg_phase_function2(i, g_1, g_2, w1, rayleigh_pol = False)]
    nspfs=len(spf_list)


    pixscale_JWST_NIRSpec = 0.10435 #arcsec per pixel

    dim = np.copy(110) #dimension of the image size in pixels (i.e., dim * dim), set this value to the number of pixels on each side of your image
    
    psfcenx = int(  dim  / 2 ) #center of x-position in pixels, need to double check if the int function is needed
    psfceny = int(  dim  / 2 ) #center of y-position in pixels, need to double check if the int function is needed

    # The code needs a mask to run, but the mask should rather be added only after convolution by a PSF, so putting it to zero here. 
    mask_diskfm = np.zeros((dim, dim), dtype=bool)

    tst1 = debrisdiskfm.anadisk_sum_mask_MMB.generate_disk( spf_list, inc = inc, pa = pa, R1 = r_in, Rc = r_c, R2 = r_out,
                            aspect_ratio = asp_ratio, beta_out = alpha_out, beta_in = alpha_in,
                            distance = distance, los_factor = los_factor, pixscale = pixscale_JWST_NIRSpec,
                            dim = dim, psfcenx = psfcenx, psfceny = psfceny, mask = mask_diskfm, dx = dx, dy = dy)
    tst1[ psfcenx , psfceny , : ] = np.zeros(nspfs) #only one model, this is Bin copied from other comdes that used this code.

    
    return tst1[:, :, 0]*flux


