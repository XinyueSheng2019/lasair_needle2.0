import os
import numpy as np
from astropy.io import fits
from needle_stream.preprocessing import single_transient_preprocessing
from needle_stream.preprocessing import feature_reduction_for_mixed_band
from needle_stream.preprocessing import apply_data_scaling
from needle_stream.preprocessing import feature_reduction_for_mixed_band_no_host
from needle_train.transient_model import *
from tensorflow.keras import models
from settings import MODEL_PATH_TH, MODEL_PATH_T

custom_objects = {
    'F1PerClassMetrics': F1PerClassMetrics,
    'CustomLearningRateSchedule': CustomLearningRateSchedule,
    'PrecisionPerClassMetrics': PrecisionPerClassMetrics,
    'RecallPerClassMetrics': RecallPerClassMetrics,
    'focal_loss_fixed_modified': focal_loss_modified()
}


def needle_th_prediction(img_data, meta_mixed):
    # for lasair-needle2.0, we only use the mixed meta. If one object only has r-band data, then g-band features will be padded with zeros.

    if meta_mixed is not None:
        result_mixed = [] 
        _img_data, meta_mixed = single_transient_preprocessing(img_data, meta_mixed)
        meta_mixed =  np.nan_to_num(meta_mixed)
        meta_mixed, _ = feature_reduction_for_mixed_band(meta_mixed)
        mixed_classifier = models.load_model(MODEL_PATH_TH, custom_objects=custom_objects) 
        _meta_mixed = apply_data_scaling(meta_mixed, 'models/hosted_model/global_scaling_data_hosted_new.json')
        result_mixed = mixed_classifier.predict({'image_input': _img_data, 'meta_input': _meta_mixed})
    else:
        result_mixed = None

    return result_mixed


def needle_t_prediction(img_data, meta_mixed):
    # binary classifier to selec t SLSN-I or SN, as TDE should be filtered out by previous steps.

    if meta_mixed is not None:
        result_mixed = [] 
        _img_data, meta_mixed = single_transient_preprocessing(img_data, meta_mixed)
        meta_mixed =  np.nan_to_num(meta_mixed)
        meta_mixed, _ = feature_reduction_for_mixed_band_no_host(meta_mixed)
        mixed_classifier = models.load_model(MODEL_PATH_T, custom_objects=custom_objects) 
        _meta_mixed = apply_data_scaling(meta_mixed, 'models/hostless_model/global_scaling_data_hostless_new.json')
        result_mixed = mixed_classifier.predict({'image_input': _img_data, 'meta_input': _meta_mixed})
    else:
        result_mixed = None
  
    return result_mixed

