import numpy as np
import time
from multiprocessing import Pool, cpu_count
from astropy.io import fits
import os

import matplotlib.pyplot as plt
import corner
import emcee

from scipy.signal import fftconvolve
from scipy import optimize
from jwstIFURDI.toolbox import imshift 
from jwstIFURDI.MMB_anadisk_model import generateModel


def single_frame_sub(sci_cube, ref_cube, mask_cube, ):
    
    """
    Returns the residual images by using the single ref frame RDI approach.
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
        Returns the value of the cost function used for the single ref frame RDI approach.

        Args:
            nu: scaling factor 
        Returns:
            cost

        Written: Chen Xie, 2023-10.

        Note that: 'sci_image' (sci image), 'ref_img' (ref), 'mask_img' (mask) are global variables in this nested function that will be updated in each iteration.
        """
        return np.log(np.nansum( ((nu * ref_img  - sci_img) * mask_img)**2  , axis=(0,1)))

    if sci_cube.ndim == 3:
        nz, ny, nx = sci_cube.shape
        res_cube = np.zeros(sci_cube.shape)
        scaling_factor = np.zeros((nz))
        # cost = np.zeros((nz))
        if mask_cube is None:
            mask_cube = np.ones(sci_cube.shape)
            print('*******   Note: using the entire FoV in scaling the reference image in RDI   *******')

        for z in range(nz):
            nu_0 = np.nansum(sci_cube[z] * mask_cube, axis=(0,1)) / np.nansum(ref_cube[z] * mask_cube, axis=(0,1)) 
            ref_img = ref_cube[z]
            sci_img = sci_cube[z]
            mask_img = mask_cube

            minimum = optimize.fmin(cost_function_subtraction, nu_0, disp=False)
            scaling_factor[z] = minimum[0]
            res_cube[z] = sci_cube[z] - minimum[0] * ref_cube[z]
            # cost[z] = cost_function_subtraction(minimum[0]) 
    elif sci_cube.ndim == 2:
        # ny, nx = sci_cube.shape
        nz = 1
        res_cube = np.zeros(sci_cube.shape)
        # scaling_factor = np.zeros((nz))
        # cost = np.zeros((nz))
        if mask_cube is None:
            mask_cube = np.ones(sci_cube.shape)
            print('*******   Note: using the entire FoV in scaling the reference image in RDI   *******')

        nu_0 = np.nansum(sci_cube * mask_cube, axis=(0,1)) / np.nansum(ref_cube * mask_cube, axis=(0,1)) 
        ref_img = ref_cube
        sci_img = sci_cube
        mask_img = mask_cube

        minimum = optimize.fmin(cost_function_subtraction, nu_0, disp=False)
        scaling_factor = minimum[0]
        res_cube = sci_cube - minimum[0] * ref_cube
 
    return res_cube, scaling_factor



def res(model):
    """
    The input empirical NIRSpec PSF has a 0.5 pix offset in the center.
    Added by Chen Xie: 2024-05-31
    """
    offset_x_sci = -0.5
    offset_y_sci = -0.5

    if sci_cube.ndim == 3:
        convolved_models = np.zeros((sci_cube.shape))
        for z in range(sci_cube.shape[0]):
            # convolved_models[z] = fftconvolve(model, psfs[z], mode='same')
            convolved_models[z] = imshift(fftconvolve(model, psfs[z], mode='same').astype(float), [offset_x_sci, offset_y_sci], method = 'fourier', ) 
    elif sci_cube.ndim == 2:
        # convolved_models = fftconvolve(model, psfs, mode='same')
        convolved_models = imshift(fftconvolve(model, psfs, mode='same').astype(float), [offset_x_sci, offset_y_sci], method = 'fourier', )


        # Multiply by transmission map
        # convolved_models[i] = np.multiply(convolved_models[i], transmission)
    # Subtract from data

    sci_cube_neg_injected = sci_cube - convolved_models
    res_cube, scaling_factor = single_frame_sub(sci_cube_neg_injected, ref_cube, mask_cube )

    return res_cube,  scaling_factor


# Set Initial ranges
def lnprior(x_var):

    bounds = np.array([[0,  0,  -15, 100000, ], 
                       [0.5, 60, 0, 4000000,]])

    for i in range(len(x_var)):
        if x_var[i] <= bounds[0, i] or x_var[i] >= bounds[1, i]:
            return -np.inf
    return 0


def lnpost(x_all):
    if lnprior(x_all) == -np.inf:
        return -np.inf
    model0 = generateModel(x_all)
    
    data,  scaling_factor = res(model0)
    model = np.zeros(model0.shape)
    
    lnlike = chi2(data*disk_mask, unc*disk_mask, model)

    return lnlike


def chi2(data, data_unc, model, lnlike = True):
    """Calculate the chi-squared value or log-likelihood for given data and model. 
    Note: if data_unc has values <= 0, they will be ignored and replaced by NaN.
    Input:  data: 2D array, observed data.
            data_unc: 2D array, uncertainty/noise map of the observed data.
            lnlike: boolean, if True, then the log-likelihood is returned.
    Output: chi2: float, chi-squared or log-likelihood value."""
    data_unc[np.where(data_unc <= 0)] = np.nan
    chi2 = np.nansum(((data-model)/data_unc)**2)

    print('reduced chi2: ', chi2/(np.count_nonzero(~np.isnan(data_unc)) - len(x_all)))

    if lnlike:
        loglikelihood = -0.5*np.log(2*np.pi)*np.count_nonzero(~np.isnan(data_unc)) - 0.5*chi2 - np.nansum(np.log(data_unc))
        # -n/2*log(2pi) - 1/2 * chi2 - sum_i(log sigma_i) 
        return loglikelihood
    return chi2



######################################################



if __name__ == '__main__':
    sci_target_name = 'HD181327_IFU_align'
    root = '/Users/sxie/Desktop/example/'
    path_masks = root + 'data/masks'              

    sci_cube_raw = fits.getdata(root+ '/centering/sci_cube_expend_{0}_aligned.fits'.format(sci_target_name))
    ref_cube_raw = fits.getdata(root+ '/centering/ref_cube_expend_{0}_aligned.fits'.format(sci_target_name))
    psfs_raw = fits.getdata(root+ 'data/HD181327_psf_masked.fits')
    sigma = fits.getdata(root+ 'anadisk_modeling/{0}_uncertainty_cube.fits'.format(sci_target_name))

    nz, ny, nx = sci_cube_raw.shape
    """
    input masks in post-processing
    """
    spike_mask = fits.getdata(path_masks+ '/spike_mask_{0}.fits'.format(sci_target_name))
    disk_region = fits.getdata(path_masks+ '/disk_mask_0_1_2D_for_PSF_convolution_Test.fits'.format(sci_target_name))
    mask_cube =  fits.getdata(path_masks+ '/{0}_mask_cube.fits'.format(sci_target_name))[0]
    FoV = fits.getdata(path_masks+ '/{0}_FoV_extra_spike.fits'.format(sci_target_name))
 

    #######################################
    ##### input disk model parameters ##### 
    #######################################
    # # disk model parameters:
    # g1 = 0.33; alpha_in = 16.2; alpha_out = -4.5;  flux = 2.3e6
    # see MMB_anadisk_model.py for details 
    x_all = [0.33,  16,  -4., 2000000., ]
    var_values_init = x_all

    n_dim = len(x_all)    # number of variables
    n_walkers = int(10*n_dim)              # an even number (>= 2n_dim)
    step =  5                      # how many steps are expected for MCMC to runx0
    trunc = 1                                        # A step number you'd like to truncate at (aim: get rid of the burrning stage)
    
    making_corner_plot = True

    # Channel range from 0 to 941 
    # NIRSpec IFU has 941 spectral channels in the PRISM/CLEAR configuration
    channel_range = np.arange(0,nz,1)

    # group_number = 1   # divide the work in to 10 parallel jobs. 
    # if  group_number == 10:
    #     channel_range = np.arange((group_number-1)*100, 941, 1 ) 
    # else:   
    #     channel_range = np.arange((group_number-1)*100, group_number*100, 1 )    

    for channel_number in channel_range:
        print('#################### {0} ###################'.format(channel_number))

        sci_cube = sci_cube_raw[channel_number]
        ref_cube = ref_cube_raw[channel_number]
        psfs = psfs_raw[channel_number]
        unc = sigma[channel_number]
        disk_mask = disk_region * spike_mask * FoV

        savepath = root + "anadisk_modeling/mcmc_result/channel_{0}/".format(channel_number)
        filename = savepath+ "state_HD181327_Model_{0}.h5".format(channel_number)
        if os.path.exists(savepath):
            filename = savepath+ "state_HD181327_Model_{0}.h5".format(channel_number)
        else:
            os.makedirs(savepath)
            filename = savepath+ "state_HD181327_Model_{0}.h5".format(channel_number)

        print('######## var_values_init:', var_values_init) 
        with Pool() as pool:
            start = time.time()

            if not os.path.exists(filename):  #initial run, no backend file existed
                backend = emcee.backends.HDFBackend(filename)   # the backend file is used to store the status
                sampler = emcee.EnsembleSampler(nwalkers = n_walkers, ndim = n_dim, log_prob_fn=lnpost, pool = pool, backend=backend)

                n_counts=0
                values_ball =[]
                for i in range(n_walkers*1000):
                    tmp = np.array(var_values_init) +  np.array(var_values_init)*np.random.randn(n_dim) 

                    if  lnprior(tmp) == -np.inf: 
                        pass
                    else:
                        values_ball.append(tmp)
                        n_counts += 1
                        if n_counts == n_walkers:
                            break

                check_bounds = np.zeros((len(values_ball)))
                for i in range(len(values_ball)):
                    if  lnprior(tmp) == -np.inf:
                        check_bounds[i] = 1 
                if np.count_nonzero(check_bounds) > 0:
                    print('Initial inputs outside bounds')
                    print('Stoping MCMC')
                elif np.array(values_ball).shape[0] != n_walkers:
                    print('Insufficient inputs')
                    print('Stoping MCMC')
                else:
                    print('Initial inputs are within the valid range')
                    print('Starting MCMC')
                    print('Shape of inputs (n_walkers, n_vars):', np.array(values_ball).shape )
                    sampler.run_mcmc(values_ball, step, progress=True)

            else:    #load the data directly from a backend file, this is used when you want to pick up from a previous MCMC run
                backend = emcee.backends.HDFBackend(filename)   # the backend file is used to store the status
                sampler = emcee.EnsembleSampler(nwalkers = n_walkers, ndim = n_dim, log_prob_fn=lnpost, pool = pool, backend=backend)
                sampler.run_mcmc(None, nsteps = step, progress=True)

            end = time.time()
            serial_time = end - start

            print("1 nodes * {0} cores {1} steps with multiprocess took {2:.1f} seconds".format(cpu_count(), step, serial_time))
        ######################################################
        ################### making outputs ###################
        ######################################################


        samples = sampler.chain[:, trunc:, :].reshape((-1, n_dim))

        g_1, alf_in, alf_out, flux,   = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))    
        mcmc_values = np.array([g_1, alf_in, alf_out, flux,  ])
        best_fit_para = [g_1[0], alf_in[0], alf_out[0], flux[0],]  


        model0 = generateModel(best_fit_para)
        convolved_model0 = fftconvolve(model0, psfs, mode='same').astype(float)   # runing a test 
        convolved_model0_shifted = imshift(fftconvolve(model0, psfs, mode='same').astype(float), [-0.5, -0.5], method = 'fourier', ) # runing a test 
        res_img,  scaling_factor = res(model0)
        model = np.zeros(model0.shape)
        chi2vcube = np.nansum((((res_img-model)*disk_mask)/(unc*disk_mask))**2)/(np.count_nonzero(~np.isnan(unc*disk_mask)) - len(x_all))

        best_fit_values = np.zeros((2))
        best_fit_values[0] = scaling_factor
        best_fit_values[1] = chi2vcube
        print('best_fit_values: ', best_fit_values)
        fits.writeto(savepath + 'HD181327_residual_channel_{0}.fits'.format(channel_number), res_img, overwrite=True)
        fits.writeto(savepath + 'HD181327_disk_model_best_fit_{0}.fits'.format(channel_number), model0, overwrite=True)
        fits.writeto(savepath + 'HD181327_mcmc_values_{0}.fits'.format(channel_number), mcmc_values, overwrite=True)
        fits.writeto(savepath + 'HD181327_best_fit_values_addtion_{0}.fits'.format(channel_number),  best_fit_values, overwrite=True)
        fits.writeto(savepath + 'sigma_{0}.fits'.format(channel_number), unc, overwrite=True)
        fits.writeto(savepath + 'HD181327_disk_model_best_fit_{0}_convolved_aligned.fits'.format(channel_number), convolved_model0_shifted, overwrite=True)
 

        if making_corner_plot:
            plt.figure()
            labels = [r'$g_1$', r'$\alpha_{in}$', r'$\alpha_{out}$', 'flux',   ]
            fig = corner.corner(samples, labels=labels, quantiles =[.16, .50, .84])         # corner plots with the quantiles  (-1sigma, median, +1sigma)
            axes = np.array(fig.axes).reshape((n_dim, n_dim))
            for yi in range(n_dim):
                for xi in range(n_dim):
                    if xi == yi:
                        ax = axes[yi, xi]
                        title  = labels[xi] + '= {0}'.format(round(mcmc_values[xi, 0], 3)) + r'$^{{+{0}}}_{{-{1}}}$'.format(round(mcmc_values[xi, 1], 3), round(mcmc_values[xi, 2], 3))
                        ax.set_title(title, color="k")

            plt.savefig(savepath + 'corner.pdf')
            plt.close()
            #####################################################

            plt.figure()
            fig, axes = plt.subplots(n_dim, figsize=(7, 18), sharex=True)
            samples = sampler.get_chain()
            # labels = ["m", "b", "log(f)"]
            labels.append('loglikelihood')
            for i in range(n_dim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(labels[i])
                # ax.yaxis.set_label_coords(-0.1, 0.5)

            axes[-1].set_xlabel("step number")
            plt.savefig(savepath + 'time_series.pdf')
            plt.close()
            ######################################################

        fits.writeto(savepath + 'HD181327_residual_channel_{0}_disk.fits'.format(channel_number), sci_cube - scaling_factor*ref_cube, overwrite=True)



    mcmc_papa = np.zeros((nz, n_dim, 3))
    chi2v_scaling =  np.zeros((nz, 2))
    residual_cube = np.zeros(sci_cube_raw.shape)
    disk_model = np.zeros(sci_cube_raw.shape)

    for z in range(nz):
        channel_number = z 

        inputpath = root + "anadisk_modeling/mcmc_result/channel_{0}/".format(channel_number)
        filename_res_img = inputpath + 'HD181327_residual_channel_{0}.fits'.format(channel_number) 
        filename_disk = inputpath + 'HD181327_disk_model_best_fit_{0}_convolved_aligned.fits'.format(channel_number)   # use this line on 2024-06-22
        filename_mcmc_result = inputpath + 'HD181327_mcmc_values_{0}.fits'.format(channel_number) 
        filename_chi2v = inputpath + 'HD181327_best_fit_values_addtion_{0}.fits'.format(channel_number) 
        filename_model_h5 = inputpath+ "state_HD181327_Model_{0}.h5".format(channel_number) 

        if os.path.exists(filename_res_img)  is False or os.path.exists(filename_res_img)  is False or os.path.exists(filename_res_img)  is False or os.path.exists(filename_res_img)  is False:
            print('Incompleted reducation of channel images #{0}'.format(channel_number)) 
        else:
            res_img = fits.getdata(filename_res_img)
            disk_img = fits.getdata(filename_disk)
            mcmc_result = fits.getdata(filename_mcmc_result)
            chi2v_single = fits.getdata(filename_chi2v)

        mcmc_papa[z] = mcmc_result 
        chi2v_scaling[z] = chi2v_single 
        residual_cube[z] = res_img  
        disk_model[z] = disk_img

    #########################
    ####### output cube #####
    #########################
    # best fit disk parameters obtained with MCMC (per wavelength)
    fits.writeto(savepath + '{0}_mcmc_para.fits'.format(sci_target_name), mcmc_papa, overwrite=True)
    # reduced chi-square and flux scaling factor (per wavelength)
    fits.writeto(savepath + '{0}_chi2v_scaling.fits'.format(sci_target_name), chi2v_scaling, overwrite=True)
    # residual images of the disk-free data cube
    fits.writeto(savepath + '{0}_residual_cube_after_FM.fits'.format(sci_target_name), residual_cube, overwrite=True)
    # best fit convolved disk model 
    fits.writeto(savepath + '{0}_best_fit_disk_model.fits'.format(sci_target_name), disk_model, overwrite=True)
