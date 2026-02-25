import h5py
import numpy as np
from sklearn.model_selection import train_test_split


def open_with_h5py(filepath):
    imageset = np.array(h5py.File(filepath, mode='r')['imageset'])
    labels = np.array(h5py.File(filepath, mode='r')['label'])
    return imageset, labels


def save_to_h5py(dataset, labels, filepath):
    print('image shape: ', dataset.shape)
    print('label shape: ', labels.shape)

    f1 = h5py.File(filepath, "w")
    f1.create_dataset("imageset", dataset.shape, dtype='f', data=dataset)
    f1.create_dataset("label", labels.shape, dtype='i', data=labels)
    f1.close()


def gen_new_data(imageset, labels, factor=10):
    """
    factor: good/bad samples ratio
    """
    num_bad = len(labels[labels == 0])
    bad_imageset = imageset[labels == 0]
    good_imageset = imageset[labels == 1][:factor*num_bad]
    bad_labels = labels[labels == 0]
    good_labels = labels[labels == 1][:factor*num_bad]
    new_imageset = np.concatenate((good_imageset, bad_imageset), axis=0)
    new_labels = np.concatenate((good_labels, bad_labels), axis=0)
    return new_imageset, new_labels


r_data_path = 'dataset/r_image_set_fixed.hdf5'
g_data_path = 'dataset/g_image_set_fixed.hdf5'

r_imageset, r_labels = open_with_h5py(r_data_path)
g_imageset, g_labels = open_with_h5py(g_data_path)

new_r_imageset, new_r_labels = gen_new_data(r_imageset, r_labels, factor=6)
new_g_imageset, new_g_labels = gen_new_data(g_imageset, g_labels, factor=6)

all_imageset = np.concatenate((new_r_imageset, new_g_imageset), axis=0)
all_labels = np.concatenate((new_r_labels, new_g_labels), axis=0)

train_imageset, test_imageset, train_labels, test_labels = train_test_split(all_imageset, all_labels,
                                                                            train_size=0.8,
                                                                            random_state=42,
                                                                            shuffle=True, stratify=all_labels)
print('number of bad in test set:', len(test_labels[test_labels == 0]))

train_data_path = 'dataset/train_image_set_fixed.hdf5'
test_data_path = 'dataset/test_image_set_fixed.hdf5'
save_to_h5py(train_imageset, train_labels, train_data_path)
save_to_h5py(test_imageset, test_labels, test_data_path)
