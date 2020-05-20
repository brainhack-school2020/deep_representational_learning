import scipy.io as sio
from os import listdir
from os.path import isfile, join
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
        
        file_name=join(rdm_dir,f'rsa_decoding_{grp_str}{sub}_sess{session}.mat')
        rdm=np.array(sio.loadmat(file_name)['rdm_time'])

        rdm_all=np.concatenate((rdm_all,rdm),axis=0)

        return  rdm_all

