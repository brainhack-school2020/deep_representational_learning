import scipy.io as sio
import scipy
import os
import os.path as op
from os.path import isfile, join
import glob
import numpy as np

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


