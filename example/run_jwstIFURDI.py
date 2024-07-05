
from time import time
import os
from jwstIFURDI import IFU_centering


if __name__ == "__main__":
    timeStamps = [time()]
	# Setup data paths	

    new_data_HD181327_IFU_align = True
    if new_data_HD181327_IFU_align:
        sci_target_name = 'HD181327_IFU_align'
        root = '/Users/sxie/Desktop/JWST'
        path = os.path.join(root, 'ms_pre_processing')
        sci_filename = os.path.join(path + '/HD181327_newoutput_prism-clear_s3d.fits')
        ref_filename = os.path.join(path + '/iotmic_newoutput_prism-clear_s3d.fits')
        # disk_mask_filename = '/Users/sxie/Desktop/JWST/ms_post_processing/make_disk_mask/disk_mask_0_1_2D.fits'
        savepath = os.path.join(root,  'ms_post_processing/centering/')

        print(path)
        print(savepath)
        print(sci_filename)
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
        aligned_sci_cube, aligned_ref_cube_rotated = IFU_centering(sci_filename, ref_filename, savepath, sci_target_name, theta_ifu_sci, theta_ifu_ref, x_center, y_center, new_img_size_x, new_img_size_y, filter_size, channel_shortest, channel_longest, )


    timeStamps.append(time())
    totalTime = timeStamps[-1]-timeStamps[0]
    print('-- Total Processing time: ', totalTime, ' s')
    print('')





