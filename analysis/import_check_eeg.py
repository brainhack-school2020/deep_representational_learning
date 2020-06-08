import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
import scipy
import scipy.spatial.distance
from helpful_functions import average_subjects_rdms, normalise_dist, get_stimuli, reorder_rdm

general_dir     = '/home/adf/faghelss/CharestLab/brainhackschool/deep_representational_learning/'
image_dir       = op.join(general_dir,'stimuli/')
output_rdm_file = op.join(general_dir,'derivatives/output_rdms.npy')

# # # # # # load data from all participants, append data from both groups, average them, and save as .npy for future use # # # # # # #

# create list refering to subjects : super-recognizer group: sub 1-16, neurotypical controls (sub 1, 3-8)
subs_sr  = np.asarray(list(range(1,17)))
subs_ctl = np.asarray(list(range(1,9)))
subs_ctl = np.delete(subs_ctl,[1,5])

# this loads all rdms, group = 0 is for ctrl
rdms_ctrl = np.asarray(average_subjects_rdms(subject_array=subs_ctl,group=0,session=1))

# this load all rdms, group = 1 is for SRs
rdms_SRs  = np.asarray(average_subjects_rdms(subject_array=subs_sr,group=1,session=1))

rdms_all=np.concatenate((rdms_SRs,rdms_ctrl))
rdms_avg  = (np.asarray(rdms_all).mean(axis=0))

# save
np.save(output_rdm_file, rdms_avg)

_min, _max = np.amin(rdms_avg), np.amax(rdms_avg)

# # # # # # show the group-average RDMs # # # # # # #

# load time vector (previously obtained from preprocessing in fieldrip (matlab))
times = scipy.io.loadmat(op.join('time_vector.mat'))['time'][0]

#  time windows of interest
time_range= dict()
time_range['0'] = np.logical_and(times<.0, times>-.2)
time_range['1'] = np.logical_and(times<.09, times>.00)
time_range['2'] = np.logical_and(times<.200, times>.120)
time_range['3'] = np.logical_and(times<.400, times>.22)

# average in each time window, show as a square matrix (original RDM format)
rdms_norm = dict()
for count in list(range(0,4)):
    rdms_norm[f'{count}'] = (rdms_avg[:,time_range[f'{count}']].mean(axis=1).squeeze())
    rdms_norm[f'{count}'] = scipy.spatial.distance.squareform(rdms_norm[f'{count}'])

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)

ax1.title.set_text(' baseline ')
ax2.title.set_text('~40 ms')
ax3.title.set_text('~150 ms')
ax4.title.set_text('~300 ms')

ax1.imshow(rdms_norm['0'], cmap='plasma', vmin = _min, vmax = _max)
plt.xticks([], [])
plt.yticks([], [])
ax2.imshow(rdms_norm['1'], cmap='plasma', vmin = _min, vmax = _max)
plt.xticks([], [])
plt.yticks([], [])
ax3.imshow(rdms_norm['2'], cmap='plasma', vmin = _min, vmax = _max)
plt.xticks([], [])
plt.yticks([], [])
ax4.imshow(rdms_norm['3'], cmap='plasma', vmin = _min, vmax = _max)
plt.xticks([], [])
plt.yticks([], [])
plt.show()