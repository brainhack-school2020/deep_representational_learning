import scipy.io as sio
import scipy
import os
import os.path as op
from os.path import isfile, join
import glob
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

def average_subjects_rdms(subject_array,group,session):
    general_dir= '/home/adf/faghelss/CharestLab/brainhackschool/deep_representational_learning/'
    rdm_dir=join(general_dir,'derivatives/data_rdms_matlab/')
    
    # group_dir=join(general_dir,f'/{group}/')
    if group: 
        grp_str='sub' 
    else: 
        grp_str='ctrl-sub'
    
    rdm_all=[]
    for sub in subject_array:
        print(f'loading eeg rdms for sub {sub}')
        file_name=join(rdm_dir,f'rsa_decoding_{grp_str}{sub}_sess{session}.mat')
        rdm=sio.loadmat(file_name)
        rdm=np.asarray(rdm['rdm_time'])

        rdm_all.append(rdm)

    return  rdm_all

def normalise_dist(distances):

    normalised_dist = (distances - np.min(distances.ravel())) / np.max(distances.ravel())
    
    return normalised_dist

def reorder_rdm(utv, newOrder):
    ax, bx = np.ix_(newOrder, newOrder)
    newOrderRDM = scipy.spatial.distance.squareform(utv)[ax, bx]
    return scipy.spatial.distance.squareform(newOrderRDM, 'tovector', 0)

def get_stimuli(image_dir,nimg):
    from PIL import Image
    images = glob.glob(os.path.join(image_dir, '*.png'))

    image_order = np.asarray(list(range(1,nimg+1)))-1

    images.sort()
    size = 128, 128
    image_data=[]
    for x in image_order:
        this_im = images[x]
        im = Image.open(this_im)
        im.thumbnail(size)
        pix = np.array(im)
        image_data.append(pix)
    return image_data, image_order

def define_labels_colors(cmap, image_order):
    
    cmap=cmap.colors
    n_categories = 6
    cmap_is = np.linspace(0, 255, n_categories).astype(int)

    category_colors = {}
    category_colors['face fear'] = cmap[cmap_is[0], :]
    category_colors['face happy'] = cmap[cmap_is[1], :]
    category_colors['face neutral'] = cmap[cmap_is[2], :]
    category_colors['animals'] = cmap[cmap_is[3], :]
    category_colors['scenes'] = cmap[cmap_is[4], :]
    category_colors['objects'] = cmap[cmap_is[4], :]
    labels_rdm = {}
    for j, _ in enumerate(image_order):
        if j<8:
            labels_rdm[j] = 'face fear'
        elif j<16:
            labels_rdm[j]= 'face happy'
        elif j<24:
            labels_rdm[j]= 'face neutral' 
        elif j<32:
            labels_rdm[j]= 'animals'
        elif j<41:
            labels_rdm[j]= 'scenes'
        elif j<49:
            labels_rdm[j]= 'objects'

    return category_colors, labels_rdm


def get_mds(utv, pos=None):
    
    rdm = scipy.spatial.distance.squareform(utv)
    seed = np.random.RandomState(seed=3)
    mds = MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit_transform(rdm, init=pos)

    return pos

def compute_rdm(measurements):
    """compute_rdm computes a correlation distance based rdm from
    a measurements array
    Args:
    measurements (2D array): conditions by features
    Returns:
    correlation distance vector
    """
    row, col = np.triu_indices(measurements.shape[0], 1)
    measurements = measurements - measurements.mean(axis=1, keepdims=True)
    measurements /= np.sqrt(np.einsum('ij,ij->i', measurements, measurements))[:, None]
    return 1 - np.einsum('ik,jk', measurements, measurements)[row, col]

