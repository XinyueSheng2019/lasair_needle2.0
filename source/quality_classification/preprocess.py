import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from astropy import visualization


def open_with_h5py(filepath):
    imageset = np.array(h5py.File(filepath, mode='r')['imageset'])
    labels = np.array(h5py.File(filepath, mode='r')['label'])
    return imageset, labels


def zscale(img):
    # where to set up pencentages.
   
    vmin = visualization.ZScaleInterval().get_limits(img)[0]
    vmax = visualization.ZScaleInterval().get_limits(img)[1]
    img[img > vmax] = vmax
    img[img < vmin] = vmin 
    img = np.nan_to_num(img, nan = vmin)
    return img


def image_normal(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def preprocessing(filepath, train_set_ratio=0.8, random_seed=None):
    imageset, labels = open_with_h5py(filepath)
    # scaling
    new_imageset = []
    for img in imageset:
        new_imageset.append(image_normal(zscale(img)).reshape(60, 60, 1))
    imageset = np.array(new_imageset)

    # Stratified Sampling
    if train_set_ratio > 0:
        train_imageset, test_imageset, train_labels, test_labels = train_test_split(imageset, labels,
                                                                                    train_size=train_set_ratio,
                                                                                    random_state=random_seed,
                                                                                    shuffle=True, stratify=labels)
    else:
        train_imageset = None
        test_imageset = imageset
        train_labels = None
        test_labels = labels
    return train_imageset, train_labels, test_imageset, test_labels
