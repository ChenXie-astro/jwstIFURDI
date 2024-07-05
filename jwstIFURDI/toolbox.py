# from matplotlib.pyplot import *
import numpy as np
from astropy.io import fits

from skimage.transform import rotate
from time import time
import os
import numpy as np
from scipy.interpolate import interp2d
from scipy import ndimage
from scipy import optimize

# from scipy.integrate import simps
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


