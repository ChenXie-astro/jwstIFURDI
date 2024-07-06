# from matplotlib.pyplot import *
import numpy as np
from skimage.transform import rotate
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift as spline_shift

"""
pip install scikit-image==0.16.2
"""

def imshift(image,
            shift,
            pad=True,
            cval=0.,
            method='fourier',
            kwargs={}):
    """
    Shift an image.
    
    Parameters
    ----------
    image : 2D-array
        Input image to be shifted.
    shift : 1D-array
        X- and y-shift to be applied.
    pad : bool, optional
        Pad the image before shifting it? Otherwise, it will wrap around
        the edges. The default is True.
    cval : float, optional
        Fill value for the padded pixels. The default is 0.
    method : 'fourier' or 'spline' (not recommended), optional
        Method for shifting the frames. The default is 'fourier'.
    kwargs : dict, optional
        Keyword arguments for the scipy.ndimage.shift routine. The default
        is {}.
    
    Returns
    -------
    imsft : 2D-array
        The shifted image.
    
    """
    
    if pad:
        
        # Pad image.
        sy, sx = image.shape
        xshift, yshift = shift
        padx = np.abs(int(xshift)) + 5
        pady = np.abs(int(yshift)) + 5
        impad = np.pad(image, ((pady, pady), (padx, padx)), mode='constant', constant_values=cval)
        
        # Shift image.
        if method == 'fourier':
            imsft = np.fft.ifftn(fourier_shift(np.fft.fftn(impad), shift[::-1])).real
        elif method == 'spline':
            imsft = spline_shift(impad, shift[::-1], order=5, **kwargs)
        else:
            raise UserWarning('Image shift method "' + method + '" is not known')
        
        # Crop image to original size.
        return imsft[pady:pady + sy, padx:padx + sx]
    else:
        if method == 'fourier':
            return np.fft.ifftn(fourier_shift(np.fft.fftn(image), shift[::-1])).real
        elif method == 'spline':
            return spline_shift(image, shift[::-1],order=5, **kwargs)
        else:
            raise UserWarning('Image shift method "' + method + '" is not known')



def shift(array, shift_value, method='roll', ):
    '''
    Shift a 1D or 2D input array.
    
    The array can be shift either using roll.

    Note that if the shifting value is an integer, the function uses
    numpy roll procedure to shift the array. The user can force to use
    np.roll, and in that case the function will round the shift value
    to the nearest integer values.
    
    Parameters
    ----------
    array : array
        The array to be shifted
    
    shift_value : float or sequence
        The shift along the axes. If a float, shift_value is the same for each axis. 
        If a sequence, shift_value should contain one value for each axis.
    

    Returns
    -------
    shift : array
        The shifted array

    '''

    method = method.lower()
    
    # array dimensions
    Ndim = array.ndim
    dims = array.shape
    if (Ndim != 1) and (Ndim != 2):
        raise ValueError('This function can shift only 1D or 2D arrays')

    # check if shift values are int and automatically change method in case they are
    if (shift_value.dtype.kind == 'i'):
        method = 'roll'
    else:
        # force integer values
        if method is 'roll':
            shift_value = np.round(shift_value)
        

    # detects NaN and replace them with real values
    mask = None
    nan_mask = np.isnan(array)
    if np.any(nan_mask):
        medval = np.nanmedian(array)
        array[nan_mask] = medval

        mask = np.zeros_like(array)
        mask[nan_mask] = 1
        
        mask = _shift_interp_builtin(mask, shift_value, mode='constant', cval=1)

    # shift with appropriate function                
    if (method == 'roll'):
        shift_value = np.round(shift_value).astype(int)
        shifted = _shift_roll(array, shift_value)
    else:
        raise ValueError('Unknown shift method \'{0}\''.format(method))

    # puts back NaN
    if mask is not None:
        shifted[mask >= 0.5] = np.nan
    
    return shifted

def _shift_interp_builtin(array, shift_value, mode='constant', cval=0):
    shifted = ndimage.shift(array, np.flipud(shift_value), order=3, mode=mode, cval=cval)

    return shifted


def _shift_roll(array, shift_value):
    Ndim  = array.ndim

    if (Ndim == 1):
        shifted = np.roll(array, shift_value[0])
    elif (Ndim == 2):
        shifted = np.roll(np.roll(array, shift_value[0], axis=1), shift_value[1], axis=0)
    else:
        raise ValueError('This function can shift only 1D or 2D arrays')
        
    return shifted



def frame_rotate(array, angle, rot_center=None, interp_order=4, border_mode='constant'):
    """ Rotates a frame or 2D array.   
    Parameters
    ----------
    array : Input image, 2d array.
    angle : Rotation angle.
    rot_center : Coordinates X,Y  of the point with respect to which the rotation will be 
                performed. By default the rotation is done with respect to the center 
                of the frame; central pixel if frame has odd size.
    interp_order: Interpolation order for the rotation. See skimage rotate function.
    border_mode : Pixel extrapolation method for handling the borders. 
                See skimage rotate function.
        
    Returns
    -------
    array_out : Resulting frame.      
    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array')

    ny,nx = array.shape 
    x_center = nx//2  # There is a systematic offset from our frame registration.
    y_center = ny//2 
    rot_center = (x_center, y_center) 

    min_val = np.nanmin(array)
    im_temp = array - min_val
    max_val = np.nanmax(im_temp)
    im_temp /= max_val

    array_out = rotate(im_temp, angle, order=interp_order, center=rot_center, cval=np.nan,
                       mode=border_mode)

    array_out *= max_val
    array_out += min_val
    array_out = np.nan_to_num(array_out)
             
    return array_out


def derotate_and_combine(cube, angles, method):
    """ Derotates a cube of images then mean-combine them.   
    Parameters
    ----------
    cube : the input cube, 3D array.
    angles : the list of parallactic angles corresponding to the cube.
        
    Returns
    -------
    image_out : the mean-combined image
    cube_out : the cube of derotated frames.   
    """
    if cube.ndim != 3:
        raise TypeError('Input cube is not a cube or 3d array')
    if angles.ndim != 1:
        raise TypeError('Input angles must be a 1D array')
    if len(cube) != len(angles):
        raise TypeError('Input cube and input angle list must have the same length')
        
    shape = cube.shape
    cube_out = np.zeros(shape)
    for im in range(shape[0]):
        cube_out[im] = frame_rotate(cube[im], -angles[im])
    
    if method == 'mean':
        image_out = np.nanmean(cube_out, axis=0)
    elif method == 'median':
        image_out = np.nanmedian(cube_out, axis=0)
 
    return image_out, cube_out






def plot_psf_correction_factor(input_1D_spec, header, plotname, savepath, color, legend):
    """
    plotting the obtained psf correction factors
    """
    pdfname = savepath + '{0}.pdf'.format(plotname)
    pdfname = savepath + '{0}.png'.format(plotname)

    fig = plt.figure(figsize=(9,6), dpi=300)
    fig.subplots_adjust(hspace=0.0, wspace=0.000, bottom=0.08, top=0.96, left=0.1, right=0.96)
    gs = gridspec.GridSpec(1, 1,)
    plt.rcParams['xtick.direction'] = 'in' 
    plt.rcParams['ytick.direction'] = 'in' 
    # plt.rcParams['xtick.top'] = True
    # plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

    topplot = plt.subplot(gs[0], )
    #######  add line indicators ######
    # arr = mpatches.FancyArrowPatch((2.7, 2.1), (3.35 , 2.1),
    #                             arrowstyle='|-|', mutation_scale=10, lw=2, color = 'gray')
    # topplot.add_patch(arr)
    # topplot.annotate(r"$\rm{H_2O}$", (.5, 1.2), xycoords=arr, ha='center', va='bottom', size = 25,  color = 'gray', weight='bold')
    # # plt.rcParams['font.serif'] =
    # arr = mpatches.FancyArrowPatch((3.1, 1.8), (3.1, 2),
    #                             arrowstyle='-', mutation_scale=10, lw=2, color = 'gray')
    # topplot.add_patch(arr)
 
    # arr = mpatches.FancyArrowPatch((3.8, 2.5), (4.8 , 2.5),
    #                             arrowstyle='|-|', mutation_scale=10, lw=2, color = 'gray')
    # topplot.add_patch(arr)
    # topplot.annotate(r"$\rm{H_2O}$", (.5, 1.2), xycoords=arr, ha='center', va='bottom', size = 25,  color = 'gray', weight='bold')


    nz = input_1D_spec.shape[0]
    print(nz)
    z_wave = np.arange(header['CRVAL3'], header['CRVAL3']+ (nz)*header['CDELT3'], header['CDELT3'])
    # topplot = plt.subplot(gs[0])
    plt.plot(z_wave[20:-21], input_1D_spec[20:-21], lw=2, alpha=1, color = color, label=legend)


    ylim_min = 0.8
    ylim_max = 1.2
    # plt.ylim([ylim_min, ylim_max])
    xlim_min = 0.5    
    xlim_max = 5.5
    plt.xlim([xlim_min, xlim_max])

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.xlabel(r'Wavelength ($\rm \mu m$)', fontsize=16)
    plt.ylabel(r' PSF convolution ratio', fontsize=16)

    topplot.tick_params(axis='both', which='major', labelsize=16, length=8, width=1.6, pad=10)
    for axis in ['top','bottom','left','right']:
        topplot.spines[axis].set_linewidth(1.6)
    legend = plt.legend(loc='best', fontsize=14,frameon=False)

    fig.align_ylabels()
    fig.tight_layout()
    plt.savefig(pdfname, transparent= True)
    plt.clf()



def plot_scaling_factor(z_wave, input_y, plotname, savepath, ):
    # pdfname = 'jet_sepctrum.pdf'# % (output_name)
    pdfname = savepath + '{0}.png'.format(plotname)
    fig = plt.figure(figsize=(10,5), dpi=300)
    ax = fig.add_subplot(111)
    # plt.fig = plt.figure(figsize=(8,6), dpi=300)
    # fig.subplots_adjust(hspace=0.0, wspace=0.0001, bottom=0.08, top=0.96, left=0.1, right=0.96)
    gs = gridspec.GridSpec(1, 1,)
    plt.rcParams['xtick.direction'] = 'in' 
    plt.rcParams['ytick.direction'] = 'in' 
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position("right")

    plt.axvline(x=4.26, color ='grey', ls='--', lw=1,  alpha=0.6)
    plt.axvline(x=3.1, color ='grey', ls='--', lw=1,  alpha=0.6)

    # z_wave = np.arange(header['CRVAL3'], header['CRVAL3']+ (nz)*header['CDELT3'], header['CDELT3'])
    plt.plot(z_wave, input_y, '-', lw=3.5, alpha=1, color ='#E07B54', zorder=10)

    ylim_min = 0.5  
    ylim_max = 1.0
    # plt.ylim([ylim_min, ylim_max])
    xlim_min = 0.5    
    xlim_max = 5.5
    plt.xlim([xlim_min, xlim_max])
    plt.tick_params(axis='both', which='major', labelsize=16)

    # plt.ylabel(r'Throughput correction factor ($T_{\lambda}$)',  fontsize=16)
    plt.ylabel(r'Scaling factor ($f_{\rm RDI}$)',  fontsize=16)
    plt.xlabel(r'Wavelength ($\rm \mu m$)', fontsize=16)
    # plt.axvline(x=3.1, color ='gray', ls='--', lw=1,  alpha=0.6)
    # plt.axvline(x=4.268, color ='gray', ls='--', lw=1,  alpha=0.6)

    ax.tick_params(axis='both', which='major', labelsize=16, length=8, width=1.6, pad=10)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.6)

    fig.tight_layout()
    plt.savefig(pdfname, transparent= True)
    plt.clf()


def plot_throughput(z_wave, input_y, plotname, savepath, ):
    # pdfname = 'jet_sepctrum.pdf'# % (output_name)
    pdfname = savepath + '{0}.png'.format(plotname)
    fig = plt.figure(figsize=(10,5), dpi=300)
    ax = fig.add_subplot(111)
    # plt.fig = plt.figure(figsize=(8,6), dpi=300)
    # fig.subplots_adjust(hspace=0.0, wspace=0.0001, bottom=0.08, top=0.96, left=0.1, right=0.96)
    gs = gridspec.GridSpec(1, 1,)
    plt.rcParams['xtick.direction'] = 'in' 
    plt.rcParams['ytick.direction'] = 'in' 
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position("right")

    # plt.axvline(x=4.26, color ='grey', ls='--', lw=1,  alpha=0.6)
    # plt.axvline(x=3.1, color ='grey', ls='--', lw=1,  alpha=0.6)
    # plt.axvline(x=1.65, color ='grey', ls='--', lw=1,  alpha=0.6)
    
    # z_wave = np.arange(header['CRVAL3'], header['CRVAL3']+ (nz)*header['CDELT3'], header['CDELT3'])
    plt.plot(z_wave[20:-21], input_y[20:-21], '-', lw=3.5, alpha=1, color ='#ECC97F',   zorder=10)

    ylim_min = 0.6 
    ylim_max = 1.05
    # plt.ylim([ylim_min, ylim_max])
    xlim_min = 0.5    
    xlim_max = 5.5
    plt.xlim([xlim_min, xlim_max])
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.ylabel(r'Throughput correction factor ($T_{\lambda}$)',  fontsize=16)
    plt.xlabel(r'Wavelength ($\rm \mu m$)', fontsize=16)
    # plt.axvline(x=3.1, color ='gray', ls='--', lw=1,  alpha=0.6)
    # plt.axvline(x=4.268, color ='gray', ls='--', lw=1,  alpha=0.6)

    ax.tick_params(axis='both', which='major', labelsize=16, length=8, width=1.6, pad=10)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.6)


    plt.tick_params(axis='both', which='major', labelsize=16)
    fig.tight_layout()
    plt.savefig(pdfname, transparent= True)
    plt.clf()


def plot_spectrum_errorbar(input_1D_spec, input_1D_spec_error, model, z_wave, z_wave_model, stats, plotname, savepath,):
    pdfname = savepath + '{0}.pdf'.format(plotname)
    fig = plt.figure(figsize=(10,7.8), dpi=300)
    fig.subplots_adjust(hspace=0.0, wspace=0.000, bottom=0.08, top=0.96, left=0.1, right=0.96)
    gs = gridspec.GridSpec(2, 1,height_ratios=[2,1])
    plt.rcParams['xtick.direction'] = 'in' 
    plt.rcParams['ytick.direction'] = 'in' 
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True


    topplot = plt.subplot(gs[0], )

    plt.errorbar(z_wave, input_1D_spec, input_1D_spec_error, 0,  color = '#8FC9E2', label='disk spectrum of HD181327', fmt='o', markersize=4, alpha=0.4, ls='none', zorder=6)
    plt.plot(z_wave_model, model, '-', lw=3.5, alpha=0.8, color = '#ECC97F',  label='model spectrum', zorder=10)


    ylim_min = 0.6   
    ylim_max = 2.1
    plt.ylim([ylim_min, ylim_max])
    xlim_min = 0.5    
    xlim_max = 5.5
    plt.xlim([xlim_min, xlim_max])
    plt.tick_params(axis='both', which='major', labelsize=16)
    legend = plt.legend(loc='lower right', fontsize=15,frameon=True, )
    plt.ylabel(r'F$_{\rm{disk}}$/F$_{\rm{star}}$ (normalized)',  fontsize=16)


    topplot.tick_params(axis='both', which='major', labelsize=16, length=8, width=1.6, pad=10)
    for axis in ['top','bottom','left','right']:
        topplot.spines[axis].set_linewidth(1.6)
    plt.setp(topplot.get_xticklabels(), visible=False)


    ##### fitting parameters ######
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    plt.text(1.9, 1.4, stats, fontsize=16, bbox=bbox,  horizontalalignment='right')


    #######  add line indicators ######
    arr = mpatches.FancyArrowPatch((2.7, 1.6), (3.35 , 1.6),
                                arrowstyle='|-|', mutation_scale=10, lw=2, color = 'gray')
    topplot.add_patch(arr)
    topplot.annotate(r"$\rm{H_2O}$", (.5, 1.2), xycoords=arr, ha='center', va='bottom', size = 25,  color = 'gray', weight='bold')
    # plt.rcParams['font.serif'] =
    arr = mpatches.FancyArrowPatch((4.268, 1.6), (4.268, 1.8),
                                arrowstyle='-', mutation_scale=10, lw=2, color = 'gray')
    topplot.add_patch(arr)
    topplot.annotate(r"$\rm{CO_2}$", (.5, 1.2), xycoords=arr, ha='center', va='bottom', size = 25, color = 'gray', weight='bold')

    arr = mpatches.FancyArrowPatch((3.1, 1.30), (3.1, 1.5),
                                arrowstyle='-', mutation_scale=10, lw=2, color = 'gray')
    topplot.add_patch(arr)
    # topplot.annotate("$Fresnel Peak$", (.5, 1), xycoords=arr, ha='center', va='bottom', size = 20, color = 'gray')




    ################ chi2 per channel  ########
    # very useful to indicate the goodness of the model fitting as the function of wavelength and as the change of unsertainty
    #

    bottomplot = plt.subplot(gs[1], sharex = topplot )
    input_1D_spec_error[np.where(input_1D_spec_error <= 0)] = np.nan
    chi2_i = np.zeros((z_wave_model.shape[0]))
    for z in range(z_wave_model.shape[0]):
        chi2_i[z] = ((input_1D_spec[z+115]-model[z])/input_1D_spec_error[z+115])**2

    plt.plot(z_wave_model, chi2_i, lw=1.5, alpha=1, color = '#ECC97F', ls='-', label=r'$\chi$$^{2}_{\lambda}$', zorder=10)

    ylim_min = 0.25   
    ylim_max = 2.5
    # plt.ylim([ylim_min, ylim_max])
    xlim_min = 0.5    
    xlim_max = 5.5
    plt.xlim([xlim_min, xlim_max])
    plt.tick_params(axis='both', which='major', labelsize=16)
    # plt.axhline(y = 1, color = 'r', linestyle = '-') 

    plt.xlabel(r'Wavelength ($\rm \mu m$)', fontsize=16)
    plt.ylabel(r'$\chi$$^{2}_{\lambda}$',  fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)

    bottomplot.tick_params(axis='both', which='major', labelsize=16, length=8, width=1.6, pad=10)
    for axis in ['top','bottom','left','right']:
        bottomplot.spines[axis].set_linewidth(1.6)
    legend = plt.legend(loc='best', fontsize=15,frameon=True)
    # legend.get_frame().set_alpha(None)
    # legend.get_frame().set_facecolor((0, 0, 1, 0.1))

    fig.align_ylabels()
    # fig.tight_layout()
    plt.savefig(pdfname, transparent= True)
    plt.clf()



###########################################################

def plot_multiple_spectrum_errorbar_old(input_1D_spec_1, input_1D_spec_error_1, input_1D_spec_2, input_1D_spec_error_2, input_1D_spec_3, input_1D_spec_error_3, z_wave, plotname, savepath, label_1, label_2, label_3):
    blue = '#0099ff'
    # pdfname = 'jet_sepctrum.pdf'# % (output_name)
    # pdfname = savepath + '{0}.pdf'.format(plotname)
    pdfname = savepath + '{0}.png'.format(plotname)
    fig = plt.figure(figsize=(10,5.2), dpi=300)

    fig.subplots_adjust(hspace=0.0, wspace=0.000, bottom=0.08, top=0.96, left=0.1, right=0.96)
    gs = gridspec.GridSpec(1, 1,)
    plt.rcParams['xtick.direction'] = 'in' 
    plt.rcParams['ytick.direction'] = 'in' 
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True

    # z_wave = np.arange(header['CRVAL3'], header['CRVAL3']+ (nz)*header['CDELT3'], header['CDELT3'], )

    topplot = plt.subplot(gs[0], )
    # z_wave=z_wave[15:-15]
    # plt.errorbar(z_wave, input_1D_spec_1, input_1D_spec_error_1, 0,  color = '#8FC9E2', label=label_1, fmt='o', markersize=4, alpha=0.4, ls='none', zorder=6)
    # plt.errorbar(z_wave, input_1D_spec_1/input_1D_spec_1[381], input_1D_spec_error_1/input_1D_spec_1[381], 0,  color = '#E07B54', label=label_1, fmt='o', markersize=4, alpha=0.4, ls='none', zorder=7)
    # plt.errorbar(z_wave, input_1D_spec_2/input_1D_spec_2[381], input_1D_spec_error_2/input_1D_spec_2[381], 0,  color = '#E1C855', label=label_2, fmt='o', markersize=4, alpha=0.4, ls='none', zorder=6)
    # plt.errorbar(z_wave, input_1D_spec_3/input_1D_spec_3[381], input_1D_spec_error_3/input_1D_spec_3[381], 0,  color = '#51B1B7', label=label_3, fmt='o', markersize=4, alpha=0.4, ls='none', zorder=6)
    markers, caps, bars = plt.errorbar(z_wave[20:-21], input_1D_spec_1[20:-21]/input_1D_spec_1[380], input_1D_spec_error_1[20:-21]/input_1D_spec_1[380], 0,  color ='#E07B54',  label=label_1, fmt='o', markersize=1, alpha=0.8, ls='-', zorder=10)
    [cap.set_alpha(0.3) for cap in caps]
    [bar.set_alpha(0.3) for bar in bars]
    markers, caps, bars = plt.errorbar(z_wave[20:-21], input_1D_spec_2[20:-21]/input_1D_spec_2[380], input_1D_spec_error_2[20:-21]/input_1D_spec_2[380], 0,  color = '#E1C855', label=label_2, fmt='o', markersize=1, alpha=0.8, ls='-', zorder=8)
    [cap.set_alpha(0.3) for cap in caps]
    [bar.set_alpha(0.3) for bar in bars]
    markers, caps, bars = plt.errorbar(z_wave[20:-21], input_1D_spec_3[20:-21]/input_1D_spec_3[380], input_1D_spec_error_3[20:-21]/input_1D_spec_3[380], 0,  color = '#51B1B7', label=label_3, fmt='o', markersize=1, alpha=0.8, ls='-', zorder=6)
    [cap.set_alpha(0.3) for cap in caps]
    [bar.set_alpha(0.3) for bar in bars]
    # markers, caps, bars = plt.errorbar(z_wave, input_1D_spec_4/input_1D_spec_4[381], input_1D_spec_error_4/input_1D_spec_4[381], 0,  color = '#818181', label=label_4, fmt='o', markersize=1, alpha=0.8, ls='-', zorder=6)
    # [cap.set_alpha(0.2) for cap in caps]
    # [bar.set_alpha(0.2) for bar in bars]


    #######  add line indicators ######
    arr = mpatches.FancyArrowPatch((2.7, 1.8), (3.35 , 1.8),
                                arrowstyle='|-|', mutation_scale=10, lw=2, color = 'gray')
    topplot.add_patch(arr)
    topplot.annotate(r"$\rm{H_2O}$", (.5, 1.2), xycoords=arr, ha='center', va='bottom', size = 25,  color = 'gray', weight='bold')
    # plt.rcParams['font.serif'] =
    arr = mpatches.FancyArrowPatch((4.268, 1.7), (4.268, 1.9),
                                arrowstyle='-', mutation_scale=10, lw=2, color = 'gray')
    topplot.add_patch(arr)
    topplot.annotate(r"$\rm{CO_2}$", (.5, 1.2), xycoords=arr, ha='center', va='bottom', size = 25, color = 'gray', weight='bold')

    arr = mpatches.FancyArrowPatch((3.1, 1.30), (3.1, 1.5),
                                arrowstyle='-', mutation_scale=10, lw=2, color = 'gray')
    topplot.add_patch(arr)
    # topplot.annotate("$Fresnel Peak$", (.5, 1), xycoords=arr, ha='center', va='bottom', size = 20, color = 'gray')


    ylim_min = 0.5  
    ylim_max = 2.2
    plt.ylim([ylim_min, ylim_max])
    xlim_min = 0.5    
    xlim_max = 5.5
    plt.xlim([xlim_min, xlim_max])
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.ylabel(r'$F_{\rm{disk}}$/$F_{\rm{star}}$ (normalized)',  fontsize=16)
    plt.xlabel(r'Wavelength ($\rm \mu m$)', fontsize=16)
    # plt.axvline(x=3.1, color ='gray', ls='--', lw=1,  alpha=0.6)
    # plt.axvline(x=4.268, color ='gray', ls='--', lw=1,  alpha=0.6)

    topplot.tick_params(axis='both', which='major', labelsize=16, length=8, width=1.6, pad=10)
    for axis in ['top','bottom','left','right']:
        topplot.spines[axis].set_linewidth(1.6)

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(loc='upper left', fontsize=15,frameon=True)
    plt.tick_params(axis='both', which='major', labelsize=16)
    fig.tight_layout()
    plt.savefig(pdfname, transparent= True)
    plt.clf()


def plot_four_spectrum_errorbar_old(input_1D_spec_1, input_1D_spec_error_1, input_1D_spec_2, input_1D_spec_error_2, input_1D_spec_3, input_1D_spec_error_3, input_1D_spec_4, input_1D_spec_error_4, header, plotname, savepath, label_1, label_2, label_3, label_4):
    blue = '#0099ff'
    # pdfname = 'jet_sepctrum.pdf'# % (output_name)
    pdfname = savepath + '{0}.pdf'.format(plotname)
    fig = plt.figure(figsize=(10,5.2), dpi=300)

    fig.subplots_adjust(hspace=0.0, wspace=0.000, bottom=0.08, top=0.96, left=0.1, right=0.96)
    gs = gridspec.GridSpec(1, 1,)
    plt.rcParams['xtick.direction'] = 'in' 
    plt.rcParams['ytick.direction'] = 'in' 
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True

    z_wave = np.arange(header['CRVAL3'], header['CRVAL3']+ (nz)*header['CDELT3'], header['CDELT3'], )

    topplot = plt.subplot(gs[0], )

    # plt.errorbar(z_wave, input_1D_spec_1/input_1D_spec_1[381], input_1D_spec_error_1/input_1D_spec_1[381], 0,  color = '#E07B54', label=label_1, fmt='o', markersize=1, alpha=0.4, ls='none', zorder=7)
    # plt.errorbar(z_wave, input_1D_spec_2/input_1D_spec_2[381], input_1D_spec_error_2/input_1D_spec_2[381], 0,  color = '#E1C855', label=label_2, fmt='o', markersize=1, alpha=0.4, ls='none', zorder=6)
    # plt.errorbar(z_wave, input_1D_spec_3/input_1D_spec_3[381], input_1D_spec_error_3/input_1D_spec_3[381], 0,  color = '#51B1B7', label=label_3, fmt='o', markersize=1, alpha=0.4, ls='none', zorder=6)
    # plt.errorbar(z_wave, input_1D_spec_4/input_1D_spec_4[381], input_1D_spec_error_4/input_1D_spec_4[381], 0,  color = '#818181', label=label_4, fmt='o', markersize=1, alpha=0.4, ls='none', zorder=6)

    markers, caps, bars = plt.errorbar(z_wave, input_1D_spec_1/input_1D_spec_1[380], input_1D_spec_error_1/input_1D_spec_1[380], 0,  color = '#E07B54', label=label_1, fmt='o', markersize=1, alpha=0.8, ls='-', zorder=7)
    [cap.set_alpha(0.2) for cap in caps]
    [bar.set_alpha(0.2) for bar in bars]
    markers, caps, bars = plt.errorbar(z_wave, input_1D_spec_2/input_1D_spec_2[380], input_1D_spec_error_2/input_1D_spec_2[380], 0,  color = '#E1C855', label=label_2, fmt='o', markersize=1, alpha=0.8, ls='-', zorder=6)
    [cap.set_alpha(0.2) for cap in caps]
    [bar.set_alpha(0.2) for bar in bars]
    markers, caps, bars = plt.errorbar(z_wave, input_1D_spec_3/input_1D_spec_3[380], input_1D_spec_error_3/input_1D_spec_3[380], 0,  color = '#51B1B7', label=label_3, fmt='o', markersize=1, alpha=0.8, ls='-', zorder=6)
    [cap.set_alpha(0.2) for cap in caps]
    [bar.set_alpha(0.2) for bar in bars]
    markers, caps, bars = plt.errorbar(z_wave, input_1D_spec_4/input_1D_spec_4[380], input_1D_spec_error_4/input_1D_spec_4[380], 0,  color = '#818181', label=label_4, fmt='o', markersize=1, alpha=0.8, ls='-', zorder=6)
    [cap.set_alpha(0.2) for cap in caps]
    [bar.set_alpha(0.2) for bar in bars]

    ylim_min = 0.4   
    ylim_max = 2.3
    plt.ylim([ylim_min, ylim_max])
    xlim_min = 0.5    
    xlim_max = 5.5
    plt.xlim([xlim_min, xlim_max])
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.ylabel(r'$F_{\rm{disk}}$/$F_{\rm{star}}$ (normalized)',  fontsize=16)
    plt.xlabel(r'Wavelength ($\rm \mu m$)', fontsize=16)
    plt.axvline(x=3.1, color ='gray', ls='--', lw=1,  alpha=0.6)
    plt.axvline(x=4.268, color ='gray', ls='--', lw=1,  alpha=0.6)

    topplot.tick_params(axis='both', which='major', labelsize=16, length=8, width=1.6, pad=10)
    for axis in ['top','bottom','left','right']:
        topplot.spines[axis].set_linewidth(1.6)

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(loc='upper left', fontsize=15,frameon=True)
    plt.tick_params(axis='both', which='major', labelsize=16)
    fig.tight_layout()
    plt.savefig(pdfname, transparent= True)
    plt.clf()
