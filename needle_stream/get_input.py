'''
This script is used to collect data from LASAIR and preprocess the data for the needle stream model.

Input:
    objectId: list of object IDs
    objectInfo: list of object information
    BClassifier: classifier model
    NUM_CORES: number of cores to use
Output:
    input data for the needle stream model
    {
        "img_data": img_data,
        "meta" : meta_data,
        "find_host": find_host
    }
'''

import os
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
import logs
from datetime import datetime
import multiprocessing as mp
sys.path.append("./source") 
from settings import *
from source.light_curve.light_curve_upsampling import NeedleMetaPipeline
from source.light_curve.cal_extinction import ext
from source.image.image_preprocessing import ImagePreprocessingNeedleLasair
import astropy.coordinates as coords
from astropy import units as u

import requests
import warnings
warnings.filterwarnings("ignore") 

NUM_CORES = mp.cpu_count()



def get_PS_host(ra, dec, radius = 0.00139):
    
    wdata = None
    if ra is None or dec is None:
        return None

    try:
        queryurl = 'https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/stack.json?'
        queryurl += 'ra='+str(ra)
        queryurl += '&dec='+str(dec)
        queryurl += '&radius='+str(radius) #0.003, 10 arcsec
        queryurl += '&columns=[raStack,decStack,gPSFMag,gPSFMagErr,rPSFMag,rPSFMagErr,iPSFMag,iPSFMagErr,zPSFMag,zPSFMagErr,yPSFMag,yPSFMagErr, gApMag,gApMagErr,rApMag,rApMagErr,iApMag,iApMagErr,zApMag,zApMagErr,yApMag,yApMagErr,yKronMag]'
        queryurl += '&nDetections.gte=6&pagesize=10000' # APMag
        print(queryurl)
        # print('\nQuerying PS1 for reference stars via MAST...\n')
        query = requests.get(queryurl, timeout=20)
        results = query.json()
    except:
        return None

    if len(results['data']) >= 1:
        data = np.array(results['data'])

        data = data[:,:-1]
        
        # remove unvalid coordinates
        data = data[(data[:,0]> -999) & (data[:,1]>-999)]
        # Below is a bit of a hack to remove duplicates

        if len(data) < 1:
            print('no data after filtering out -999 and failing star-galaxy separation.')
            return None

        data[data == -999] = np.nan
        
        catalog = coords.SkyCoord(ra=data[:,0]*u.degree, dec=data[:,1]*u.degree)
        data2 = []
        indices = np.arange(len(data))
        used = []
        for i in data:
            source = coords.SkyCoord(ra=i[0]*u.degree, dec=i[1]*u.deg)
            d2d = source.separation(catalog)
            catalogmsk = d2d < 2.5*u.arcsec
            indexmatch = indices[catalogmsk]
            for j in indexmatch:
                if j not in used:
                    data2.append(data[j])
                    for k in indexmatch:
                        used.append(k)

        # print(data2)
        if len(data2)>=1:
            # add g-r and r-i columns
            data2 = np.array(data2)
            wdata = pd.DataFrame(data2, columns = ['ra', 'dec', 'gPSF', 'gPSFerr', 'rPSF', 'rPSFerr', 'iPSF', 'iPSFerr', 'zPSF', 'zPSFerr', 'yPSF', 'yPSFerr', 'gAp', 'gAperr', 'rAp', 'rAperr', 'iAp', 'iAperr', 'zAp', 'zAperr', 'yAp', 'yAperr'])
            exts = ext(ra, dec)
            for i in ['g', 'r', 'i', 'z', 'y']:
                wdata[f'{i}PSF'] = wdata[f'{i}PSF'] - exts[f'PS_{i}']
                wdata[f'{i}Ap'] = wdata[f'{i}Ap'] - exts[f'PS_{i}']

            wdata['g-r_PSF'] = wdata['gPSF'] - wdata['rPSF']
            wdata['r-i_PSF'] = wdata['rPSF'] - wdata['iPSF']
            wdata['g-r_PSFerr'] = np.sqrt(wdata['gPSFerr']**2 + wdata['rPSFerr']**2)
            wdata['r-i_PSFerr'] = np.sqrt(wdata['rPSFerr']**2 + wdata['iPSFerr']**2)
            wdata['g-r_Ap'] = wdata['gAp'] - wdata['rAp']
            wdata['r-i_Ap'] = wdata['rAp'] - wdata['iAp']
            wdata['g-r_Aperr'] = np.sqrt(wdata['gAperr']**2 + wdata['rAperr']**2)
            wdata['r-i_Aperr'] = np.sqrt(wdata['rAperr']**2 + wdata['iAperr']**2)

    return wdata


def get_host_meta(host_ra, host_dec, only_complete=True):
    """
    Add host galaxy metadata from a CSV file.

    Parameters:
    - host_ra: float, host right ascension
    - host_dec: float, host declination
    - only_complete: bool, whether to include only complete metadata

    Returns:
    - list, host metadata for the object or None if not found
    """
    def add_mag(line, band):
        return line.get(f'{band}Ap') or line.get(f'{band}PSF') or None

    wdata = get_PS_host(ra = host_ra, dec = host_dec) 

    if wdata is not None:
        h_row = [add_mag(wdata.iloc[0], band) for band in ['g', 'r', 'i', 'z', 'y', 'g-r_', 'r-i_']]
        h_dict = dict(zip(['g', 'r', 'i', 'z', 'y', 'g-r_', 'r-i_'], h_row))
        return h_dict if all(h_row) or not only_complete else None
    else:
        return None 

def get_obj_meta(candidates, candi_idx, disc_mjd, host_mag, for_mixed = False):

    def _get_ratio(delta_mag_recent, delta_t_recent, delta_mag_disc, delta_t_disc):
        """
        Calculate the ratios of delta magnitude to delta time for recent and discovery values.

        Parameters:
        - delta_mag_recent: float, recent delta magnitude
        - delta_t_recent: float, recent delta time
        - delta_mag_disc: float, discovery delta magnitude
        - delta_t_disc: float, discovery delta time

        Returns:
        - tuple of floats, (ratio_recent, ratio_disc)
        """
        if isinstance(delta_mag_disc, np.ndarray):
            return (
                np.divide(delta_mag_recent, delta_t_recent, out=np.zeros_like(delta_mag_recent), where=delta_t_recent != 0),
                np.divide(delta_mag_disc, delta_t_disc, out=np.zeros_like(delta_mag_disc), where=delta_t_disc != 0)
            )
        else:
            if delta_t_disc == 0.0 or delta_t_recent == 0.0:
                return 0.0, 0.0
            return delta_mag_recent / delta_t_recent, delta_mag_disc / delta_t_disc


    candi_mag = candidates[candi_idx]['magpsf']
    disc_band_mag = candidates[-1]['magpsf']
    delta_t_discovery_band = round(candidates[candi_idx]['mjd'] - candidates[-1]['mjd'], 5)
    delta_t_discovery = round(candidates[candi_idx]['mjd'] - disc_mjd, 5)
    delta_mag_discovery = round(candi_mag - disc_band_mag, 5)

    if candi_idx < len(candidates) - 1: 
        delta_t_recent = round(candidates[candi_idx]['mjd'] - candidates[candi_idx + 1]['mjd'], 5)
        delta_mag_recent = round(candi_mag - candidates[candi_idx + 1]['magpsf'], 5)
    else:
        delta_t_recent = delta_t_discovery
        delta_mag_recent = delta_mag_discovery

    ratio_recent, ratio_disc = _get_ratio(delta_mag_recent, delta_t_recent, delta_mag_discovery, delta_t_discovery_band)
    if for_mixed:
        row = [candi_mag, disc_band_mag, delta_mag_discovery, delta_t_discovery_band, delta_t_discovery, ratio_recent, ratio_disc]
    else: # for meta_r
        row = [candi_mag, disc_band_mag, delta_mag_discovery, delta_t_discovery, ratio_recent, ratio_disc]

    if host_mag is not None:
        delta_host_mag = round(candi_mag - host_mag, 5)
        row += [delta_host_mag]

    return row


def collect_data_from_lasair(objectId=None, objectInfo=list, **log):


    def find_earliest_discovery_mjd(objectInfo):
        # some new objects have forced photometry, which the discovery dates could be earlier.
        # this functions check the earliest date that the event is rising from the forced information.
        # for g and r band, find the lastest detection date that the diff > 0, and choose the earliest among two dates as the final pre-discovery date
        def find_earliest_disc_each_band(forced_info, fid, unforced_disdate):
            b_info = np.array([(x['mjd'], x['forcediffimflux']) for x in forced_info if x['fid'] == fid and x['ranr'] > -99999.0 and x['mjd'] <= unforced_disdate], 
                            dtype=[('mjd', 'f8'), ('forcediffimflux', 'f8')])
            
            if b_info.shape[0] != 0:
                b_info_sort = np.sort(b_info, order='mjd')
                pre_disc_mjd = b_info_sort[0][0]
                n = 0
                while n < b_info_sort.shape[0] - 1:
                    if b_info_sort[n][1] < 0 and b_info_sort[n+1][1] > 0:
                        pre_disc_mjd =  b_info_sort[n+1][0]
                    n += 1
                return pre_disc_mjd
            else:
                return unforced_disdate

        unforced_disdate = objectInfo['objectData']['discMjd']
    
        if 'forcedphot' in objectInfo.keys():
            forced_info = objectInfo['forcedphot']
            max_mjd_1 = find_earliest_disc_each_band(forced_info, 1, unforced_disdate)
            max_mjd_2 = find_earliest_disc_each_band(forced_info, 2, unforced_disdate)
            print(f'g band pre_disc is {max_mjd_1}, r band pre_disc is {max_mjd_2}.')
            return max_mjd_1 if max_mjd_1 <= max_mjd_2 else max_mjd_2 
        else:
            return unforced_disdate

    def find_peak_mag_images(candids):
        if len(candids) > 0:
            info = np.array([[m["magpsf"], m["mjd"], m['image_urls']] for m in candids])
            idx = np.argmin(info[:,0])
            image_urls = []
            i = idx
            while i < info.shape[0]:
                image_urls.append(info[i][2])
                i += 1
            return float(info[idx][1]), float(info[idx][0]), image_urls, idx
        else:
            return None, None, None, None



    candidates = objectInfo['candidates']

    candids = [x for x in candidates if 'image_urls' in x.keys()] # TEST ALL 
    candids_g = [x for x in candidates if 'image_urls' in x.keys() and x['fid'] == 1]
    candids_r = [x for x in candidates if 'image_urls' in x.keys() and x['fid'] == 2]
    disdate = find_earliest_discovery_mjd(objectInfo)


    flag_r = False

    img_data, meta_r, meta_mixed, find_host = None, None, None, False
    
    # get the images 
    peak_mjd_r, peak_mag_r, image_urls_r, idx_r = find_peak_mag_images(candids_r)
    peak_mjd_g, peak_mag_g, image_urls_g, idx_g = find_peak_mag_images(candids_g)

    obj_path = os.path.join(NEEDLE_OBJ_PATH, objectId)
    if os.path.isdir(obj_path) == False:
        os.makedirs(obj_path)

    # get restored and masked images
    img_data = ImagePreprocessingNeedleLasair(masking = True,
                 objectInfo = objectInfo,
                 image_urls = image_urls_r, 
                 obj_path = obj_path).processed_array
    if img_data is None:
        img_data = ImagePreprocessingNeedleLasair(masking = True,
                 objectInfo = objectInfo,
                 image_urls = image_urls_g, 
                 obj_path = obj_path).processed_array
    if img_data is None:
        if logs.log:
            logs.log.write('object %s images in g and r bands do not pass criteria or not found.\n' % objectId)
        return None, None, None, False


    # get host meta 
    if 'sherlock' in objectInfo.keys() and objectInfo['sherlock']['raDeg'] is not None and objectInfo['sherlock']['decDeg'] is not None:
        host_ra, host_dec = objectInfo['sherlock']['raDeg'], objectInfo['sherlock']['decDeg']
        host_ra = None if host_ra == 0.0 else host_ra
        host_dec = None if host_dec == 0.0 else host_dec
        print('host_ra, host_dec: ', host_ra, host_dec)
        host_meta = get_host_meta(host_ra, host_dec)

        if host_meta is not None:
            host_g, host_r = host_meta['g'], host_meta['r']
            # sherlock_meta = objectInfo['sherlock']['separationArcsec']
            if 'separationArcsec' in objectInfo['sherlock']:
                host_meta['offset'] = float(objectInfo['sherlock']['separationArcsec'])
                find_host = True
            else:
                host_meta['offset'] = None   
                find_host = False
        else:
            host_g, host_r = None, None
            host_meta = None
            find_host = False
    else:
        host_g, host_r = None, None
        host_meta = None
        find_host = False


    # get meta_mixed
    meta_process = NeedleMetaPipeline(objectInfo = objectInfo, img_host_data = host_meta)

    meta_r, meta_mixed, find_host = meta_process.get_needle_meta(meta_process.lc_data)

    return img_data, meta_r, meta_mixed, find_host


