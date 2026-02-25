import sys, os
import re
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from quality_classification.quality_classification import QualityClassification
from image.image_restoration import ImageRestoration
from image.masking import Masking
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy import stats
import tensorflow as tf
import warnings
from astropy.utils.exceptions import AstropyWarning

import multiprocessing as mp
from astropy.stats import sigma_clip


warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


SCI_RE = re.compile('sci')
DIFF_RE = re.compile('diff')
REF_RE = re.compile('ref')
OBJ_RE = re.compile('ZTF')



def most_common(lst):
    return max(set(lst), key=lst.count)


class ImagePreprocessingNeedleLasair:

    """
    Streaming data for Needle-lasair version.
    This class is used to preprocess the image of the transient object.
    - img_path: str, path to the image data
    - host_data_path: str, path to the host data
    - mag_path: str, path to the magnitude data
    - output_path: str, path to save the output data
    - label_dict: dict, dictionary mapping labels
    """

    #--------------------------------
    # Initialization methods
    #--------------------------------
    def __init__(self, 
                 masking = True,
                 objectInfo = None,
                 image_urls = None, 
                 obj_path = None):

        if objectInfo is None:
            raise ValueError("objectInfo must be provided")
        
        self.masking = masking  
        self.image_urls = image_urls
        self.obj_path = obj_path
        self.objectInfo = objectInfo
        self.ztf_object = self.objectInfo['objectId']
        self.target_ra, self.target_dec = objectInfo['objectData']['ramean'], objectInfo['objectData']['decmean']
        self.host_ra, self.host_dec = objectInfo['sherlock']['raDeg'], objectInfo['sherlock']['decDeg']
        self.host_ra = None if self.host_ra == 0.0 else self.host_ra
        self.host_dec = None if self.host_dec == 0.0 else self.host_dec
        self.__quality_check_model = QualityClassification(verbose=False)
     

        print('image preprocessing ztf_object: ', self.ztf_object)
        self.processed_array = None
        self.processed_array = self.get_needle_imgdata

        # try: 
        #     self.processed_array = self.get_needle_imgdata
        #     if self.processed_array is None:    
        #         print('No image data available')
        #     else:
        #         print('Image data loaded successfully')
        # except:
        #     print('Excution fault, no image data available')
                
    #--------------------------------
    # Image processing methods
    #--------------------------------
    def _zscale(self, img, log_img=False):
        """
        Apply Z-scale normalization to an image.

        Parameters:
        - img: ndarray, input image
        - log_img: bool, whether to apply logarithmic scaling

        Returns:
        - ndarray, normalized image
        """
        # vmin = visualization.ZScaleInterval().get_limits(img)[0]
        _, median, _ = stats.sigma_clipped_stats(img, mask=None, sigma=3.0, cenfunc='median')
        img = np.nan_to_num(img, nan=median)
        return np.log(img) if log_img else img

    def _image_normal(self, imgs):
        """
        Normalize the image data to the range [0, 1].

        Parameters:
        - imgs: list or numpy array, input images

        Returns:
        - list, normalized images
        """
        if not isinstance(imgs, (list, np.ndarray)):
            raise TypeError("imgs must be a list or numpy array")
            
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]
            
        imgs_array = np.array(imgs)
        vmin = np.nanmin(imgs_array.flatten())
        vmax = np.nanmax(imgs_array.flatten())
        
        return [(img - vmin) / (vmax - vmin + 1e-6) for img in imgs]

    def _check_shape(self, img):
        """
        Check if the image has the shape (60, 60) and does not contain only NaN values.
        """
        return img is not None and img.shape == (60, 60) and not np.all(np.isnan(img))

    def _cutout_img_wcs(self, data, header, size=60):
        """
        Cutout the image to the proposed size.
        """
        pixels = WCS(header=header).all_world2pix(self.target_ra, self.target_dec, 1)
        pixels = [int(x) for x in pixels]
        cutout = Cutout2D(data, position=pixels, size=size)
        return cutout.data

    def plot_imgs(self, imgs):
        """
        show the test image
        """
        if len(imgs.shape) == 3:
            plt.figure(figsize=(imgs.shape[0]*3, 3))
            for i in range(imgs.shape[0]):
                plt.subplot(1, imgs.shape[0], i+1)
                plt.imshow(imgs[i])
                plt.title('score: {:.2f}'.format(self.quality_check(imgs[i].copy())))
                plt.gca().set_axis_off()
            plt.show()
        else:
            plt.figure(figsize=(3, 3))
            plt.imshow(imgs)
            plt.title('score: {:.2f}'.format(self.quality_check(imgs.copy())))
            plt.gca().set_axis_off()
            plt.show()

    def _cutout_img_pixel(self, img, size=60):
        """
        cut the image to the desired size
        """
        for i in range(2):
            if img.shape[i] > size:
                img = img[:size, :] if i == 0 else img[:, :size]
        return img

    #--------------------------------
    # Image quality check methods
    #--------------------------------
    def _quality_check(self, image, threshold = 0.5):
        """
        check the quality of the image
        """
        image_copy = image.copy()
        if np.isnan(image_copy).any():
            return False
        if self._check_shape(image_copy):
            result = self.__quality_check_model.run(image_copy)
            if result >= threshold:
                return True
            else:
                return False
        else: 
            return False
            

    def _get_header_data(self, filename):
        """
        Cutout image with 60x60 size and handle FITS file with different extensions.
        """
        try: 
            with fits.open(filename, ignore_missing_end=True) as f:
                # print('-------------------test header data: ', f[0].header)
                if filename.endswith('fz'):
                    f.verify('fix')
                    
                    # hdr = f[1].header
                    # data = f[1].data
                    hdr = f[0].header
                    data = f[0].data
                else:
                    hdr = f[0].header
                    data = f[0].data
                self.fits_header = hdr
                if hdr['NAXIS1'] >= 60 and hdr['NAXIS2'] >= 60:
                    data = self._cutout_img_pixel(data)
                if np.all(np.isnan(data)):
                    return None, None
                else:
                    return data, hdr
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            return None, None


    #--------------------------------
    # Image coordinate mapping methods
    #--------------------------------

    def _check_sci_ref_alignment(self, sci_hdr, ref_hdr): 
        '''
        check if the science and reference images are aligned
        '''
        try: 
            sci_ra, sci_dec = sci_hdr['CRVAL1'], sci_hdr['CRVAL2']
            ref_ra, ref_dec = ref_hdr['CRVAL1'], ref_hdr['CRVAL2']
            if abs(sci_ra - ref_ra) < 2 and abs(sci_dec - ref_dec) < 2:
                return True
            else:
                print(f"sci_ra: {sci_ra}, ref_ra: {ref_ra}, \n sci_dec: {sci_dec}, ref_dec: {ref_dec}")
                return False
        except:
            return True
        

    #--------------------------------
    # Image data loading methods
    #--------------------------------


    # def _get_detection_with_mag(self, meta_path):
    #     m = open(meta_path, 'r')
    #     jfile = json.loads(m.read())
    #     return jfile["f2"]["withMag"]

    # @property
    # def _get_image_meta(self):
    #     with open(os.path.join(self.__img_dir, 'image_meta.json'), 'r') as mj:
    #         meta = json.load(mj)
    #     return meta
    
    # @property
    # def _get_image_with_mag(self):
    #     with open(os.path.join(self.__img_dir, 'mag_with_img.json'), 'r') as j:
    #         mag_wg = json.load(j)
    #     return mag_wg
    
    @property
    def _get_image_info(self): 
        '''
        get all the image information for g and r band.
        sort the image by the magnitude, from low to high. To make sure the peak data is the top one.
        '''

        band_data = {}
        for band in self.__band_dict:
            candids = self.mag_wg["candidates_with_image"][f'f{self.__band_dict[band]}']
            if not candids: #or self.meta[f'f{self.__band_dict[band]}']["obj_with_no_ref"]
                band_data[band] = None
            elif len(candids) == 0:
                band_data[band] = None
            else: # make sure there are mag data to process, otherwise return None
                mags = [[m['magpsf'], m["filefracday"]] for m in candids]
                mags.sort(key=lambda x: x[0])
                band_data[band] = np.array(mags)

        return band_data

    def _check_raw_images(self):
        '''
        check if the raw science and reference images are good quality
        '''
        if self.raw_sci_image is None or self.raw_ref_image is None:
            return False, False
        else:
            return self.quality_check(self.raw_sci_image), self.quality_check(self.raw_ref_image)
    

    def _get_reference_image(self):
        """
        if one obj has multiple ref_imgs, and top one is bad, the other is good, we could still use it.
        """

        ref_filename = os.path.join(self.obj_path, 'ref_peak.fits')
        good_ref, good_hdr = None, None
        bad_ref, bad_hdr = None, None
        i = 0
        while i < len(self.image_urls):
            ref_url = self.image_urls[i]['Template']
            os.system(f'curl -o {ref_filename} {ref_url}')
            data, hdr = self._get_header_data(ref_filename)
            if data is not None:
                quality_score = self._quality_check(data)
                if quality_score:
                    good_ref, good_hdr = data, hdr
                    break
                else:
                    bad_ref, bad_hdr = data, hdr
                    i += 1
                    continue
            else:
                i += 1
                continue
        return good_ref, good_hdr, bad_ref, bad_hdr

    def _get_science_image(self):
        '''
        get the science image and the header data
        '''
        good_sci, good_hdr = None, None
        bad_sci, bad_hdr = None, None
        i = 0
        sci_filename = os.path.join(self.obj_path, 'sci_peak.fits')
        while i < len(self.image_urls):
            sci_url = self.image_urls[i]['Science']
            os.system(f'curl -o {sci_filename} {sci_url}')
            sci_data, sci_hdr = self._get_header_data(sci_filename)
            if sci_data is not None:
                quality_score = self._quality_check(sci_data)
                if quality_score:
                    good_sci, good_hdr = sci_data, sci_hdr
                    break
                else:
                    bad_sci, bad_hdr = sci_data, sci_hdr
                    i += 1
                    continue
            else:
                i += 1
                continue
            i += 1
        return good_sci, good_hdr, bad_sci, bad_hdr


    
           
    def _load_image_data(self):
        """
        Load image data from Lasair db for preprocessing.

        Returns:
            tuple: (sci_data, sci_hdr, ref_data, ref_hdr, image_flags)
                Contains image arrays, headers, and quality flags:
                1 = good, 0 = none, -1 = bad
        """

        sci_filename = os.path.join(self.obj_path, 'sci_peak.fits')
        ref_filename = os.path.join(self.obj_path, 'ref_peak.fits')


        if self.image_urls is None:
            print("No image urls available")
            return None, None, None, None, {'science': 0, 'reference': 0}

        cache = None
        image_flags = {'science': 0, 'reference': 0}

        has_sci, has_ref = False, False
        sci_good, sci_hdr_good, sci_bad, sci_hdr_bad = self._get_science_image()
        ref_good, ref_hdr_good, ref_bad, ref_hdr_bad = self._get_reference_image()


        # If good pair found, return immediately
        if sci_good is not None and ref_good is not None:
            # print("Returning GOOD quality science and reference image data")
            return sci_good, sci_hdr_good, ref_good, ref_hdr_good, {'science': 1, 'reference': 1}
        else:
            # Cache this band's best data if no good match found
            has_sci = sci_good is not None or sci_bad is not None
            has_ref = ref_good is not None or ref_bad is not None

            if has_sci and has_ref:
                sci_data = sci_good if sci_good is not None else sci_bad
                sci_hdr = sci_hdr_good if sci_hdr_good is not None else sci_hdr_bad
                ref_data = ref_good if ref_good is not None else ref_bad
                ref_hdr = ref_hdr_good if ref_hdr_good is not None else ref_hdr_bad
                image_flags['science'] = 1 if sci_good is not None else (-1 if sci_data is not None else 0)
                image_flags['reference'] = 1 if ref_good is not None else (-1 if ref_data is not None else 0)
                return sci_data, sci_hdr, ref_data, ref_hdr, image_flags
            else:
                return None, None, None, None, image_flags

    #--------------------------------
    # Image restoration methods
    #--------------------------------

    def _run_obj(self):

        """
        Runs image preprocessing pipeline for a given object.
        - Loads science and reference images
        - Restores low-quality images (if possible)
        - Applies masking and random augmentation (flip/rotate)
        - Returns 1 processed sample as a numpy array

        Returns:
            np.ndarray: array of processed image samples with shape 
                        (1, 2, 60, 60) [science, reference]
        """

        
        processed_img_array = None

        best_sci_data, best_sci_hdr, best_ref_data, best_ref_hdr, image_flags = self._load_image_data() 

        if image_flags['science'] != 0 and image_flags['reference'] != 0:
            alignment = self._check_sci_ref_alignment(best_sci_hdr, best_ref_hdr)
            if alignment:
                # print(best_sci_data.shape, best_ref_data.shape)
                img_restoration = ImageRestoration(obj_id = self.ztf_object, 
                                                    sci_data = best_sci_data, sci_hdr = best_sci_hdr, 
                                                    ref_data = best_ref_data, ref_hdr = best_ref_hdr, 
                                                    target_ra = self.target_ra, target_dec = self.target_dec, 
                                                    host_ra = self.host_ra, host_dec = self.host_dec)
                
                # Initialize restoration score
                restore_score = 0.0

                if image_flags['science'] < 1 or image_flags['reference'] < 1:
                    
                    if image_flags['science'] == 1 and image_flags['reference'] == -1:
                        restore_score = img_restoration._SSIM_restore(is_sci = False , threshold = 0.2)
                        # if display:
                        #     if img_restoration.sci_data is not None and img_restoration.ref_data is not None: 
                        #         display_image_pair(img_restoration.sci_data, img_restoration.ref_data, titles=None)
                            
                    
                    elif image_flags['science'] == -1 and image_flags['reference'] == 1:
                        restore_score = img_restoration._SSIM_restore(is_sci = True , threshold = 0.2)
                        # if display:
                        #     if img_restoration.sci_data is not None and img_restoration.ref_data is not None: 
                        #         display_image_pair(img_restoration.sci_data, img_restoration.ref_data, titles=None)
            
                    else:
                        # print('The science and reference images are both bad quality.')
                        restore_score = 0.0
                    
                else:
                    restore_score = 1.0

                if restore_score >= 0.8:
                    if self.masking:
                        img_masking = Masking(sci_data = img_restoration.sci_data, ref_data = img_restoration.ref_data, 
                                            pixel_target = img_restoration.pixel_coords_target, 
                                            pixel_host = img_restoration.pixel_coords_host,
                                            display = False)
                        
                        img_masking._get_masked_img(sigma = 2)
          
                        final_sci, final_ref = img_masking.masked_sci_data, img_masking.masked_ref_data
                    else: # no masking, raw data
                        final_sci, final_ref = img_restoration.sci_data, img_restoration.ref_data

                    processed_img_array = np.array(self._image_normal([final_sci, final_ref]))

        return processed_img_array


    
    @property
    def get_needle_imgdata(self): 
        '''
        get the science and reference image data for NEEDLE inputs.
        '''
        image_array = self._run_obj()

        if image_array is not None:
            # If images are missing the last channel dimension, add it
            sci = image_array[0]
            ref = image_array[1]
            if sci.ndim == 2:
                sci = sci[..., np.newaxis]
            if ref.ndim == 2:
                ref = ref[..., np.newaxis]
            return np.concatenate((sci, ref), axis=-1)   # (h, w, 2)
        else:
            return None



        
    
    #--------------------------------
    # Plotting methods
    #--------------------------------


        
    def plot_mask(self, image, mask1, mask2, fname):
        plt.clf()
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')#, vmin=np.min(sci_data))
        plt.subplot(1, 3, 2)
        plt.imshow(mask1, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.imshow(mask2, cmap='gray')
        plt.savefig(f'{self.__output_dir}/{fname}.png')

    def plot_restore(self, image, image1):
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')#, vmin=np.min(sci_data))
        plt.subplot(1, 2, 2)
        plt.imshow(image1, cmap='gray')
        plt.savefig(f'{self.__output_dir}/restored.png')

    def plot_coordinates(self, image, target_coords, host_coords):
        plt.imshow(image)
        plt.plot(target_coords[0], target_coords[1], marker='o', color='red', markersize=5, label= 'target RA/DEC')
        plt.plot(host_coords[0], host_coords[1], marker='o', color='blue', markersize=5, label= 'host RA/DEC')
        plt.legend()
        plt.savefig(f'{self.__output_dir}/coords.png')

    def plot_img(self, image, fname):
        plt.clf()
        plt.imshow(image, cmap='gray')#, vmin=np.min(sci_data))
        plt.savefig(f'{self.__output_dir}/{fname}.png')


# def process_obj(obj, mag_path, host_path, img_path, output_path):
#     if not os.path.exists(output_path + obj):
#         img_demo = ImagePreprocessing(ztf_object = obj, 
#                                         mag_path = mag_path,
#                                         host_data_path = host_path,
#                                         img_path = img_path,
#                                         output_path = output_path,
#                                         ztf_obj_info_path = None,
#                                         display = False,
#                                         augment = False,
#                                         train_mode = False,
#                                         masking = False) 
#         img_demo.save_obj_imgdata()
#         print(f'Successfully processed {obj}')
#     else:
#         print(f'object {obj} already exists.')
        

