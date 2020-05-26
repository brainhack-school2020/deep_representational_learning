import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
import scipy
import scipy.spatial.distance
from helpful_functions import average_subjects_rdms, normalise_dist, get_stimuli, reorder_rdm

# create list refering to subjects 
# Super-recognizer group: sub 1-16 
# neurotypical controls (sub 1, 3-8)
subs_sr  = np.asarray(list(range(1,17)))
subs_ctl = np.asarray(list(range(1,9)))
subs_ctl = np.delete(subs_ctl,[1,5])


# load all rdms, group = 0 is for ctrl
rdms_ctrl = np.asarray(average_subjects_rdms(subject_array=subs_ctl,group=0,session=1))

# load all rdms, group = 1 is for SRs
rdms_SRs = np.asarray(average_subjects_rdms(subject_array=subs_sr,group=1,session=1))

# append data
rdms_all=np.concatenate((rdms_SRs,rdms_ctrl))


general_dir= '/home/adf/faghelss/CharestLab/brainhackschool/deep_representational_learning/'

# load stimuli
nimg = 49
image_dir = op.join(general_dir,'stimuli/')
output_rdm_file = op.join(general_dir,'derivatives/output_rdms.npy')
image_data, image_order = get_stimuli(image_dir,nimg)

# define category colors
# categories, category_colors = define_category_mds(cmap, image_order)

times = scipy.io.loadmat(op.join('time_vector.mat'))['time'][0]

# the time windows of interest : need to make in a loop for the future.
time_range_1 = np.logical_and(times<.0, times>-.2)
time_range_2 = np.logical_and(times<.09, times>.00)
time_range_3 = np.logical_and(times<.200, times>.120)
time_range_4 = np.logical_and(times<.400, times>.22)

# time_range = np.asarray(list(range(70,150)))

rdms_avg  = (np.asarray(rdms_all).mean(axis=0))


np.save(output_rdm_file, rdms_avg)

_min, _max = np.amin(rdms_avg), np.amax(rdms_avg)

rdm_norm_1 = (rdms_avg[:,time_range_1].mean(axis=1).squeeze())
rdm_norm_2 = (rdms_avg[:,time_range_2].mean(axis=1).squeeze())
rdm_norm_3 = (rdms_avg[:,time_range_3].mean(axis=1).squeeze())
rdm_norm_4 = (rdms_avg[:,time_range_4].mean(axis=1).squeeze())


# show as a square matrix (original RDM format),
rdm_1 = scipy.spatial.distance.squareform(rdm_norm_1)
rdm_2 = scipy.spatial.distance.squareform(rdm_norm_2)
rdm_3 = scipy.spatial.distance.squareform(rdm_norm_3)
rdm_4 = scipy.spatial.distance.squareform(rdm_norm_4)

# rdm = reorder_rdm(rdm, image_order)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)

ax1.title.set_text(' baseline ')
ax2.title.set_text('~40 ms')
ax3.title.set_text('~150 ms')
ax4.title.set_text('~300 ms')

ax1.imshow(rdm_1, cmap='magma', vmin = _min, vmax = _max)
#ax1.autoscale(False)
plt.xticks([], [])
plt.yticks([], [])
ax2.imshow(rdm_2, cmap='magma', vmin = _min, vmax = _max)
#ax2.autoscale(False)
plt.xticks([], [])
plt.yticks([], [])
ax3.imshow(rdm_3, cmap='magma', vmin = _min, vmax = _max)
#ax3.autoscale(False)
plt.xticks([], [])
plt.yticks([], [])
ax4.imshow(rdm_4, cmap='magma', vmin = _min, vmax = _max)
#ax4.autoscale(False)
plt.xticks([], [])
plt.yticks([], [])
# fig.tight_layout()
plt.show()



# group='SRs'
# # plot_mds(rdms_avg, image_data, categories, category_colors, scaler=0.02)
# figname = f'MDS_{group}_{time_label}.png'
# figfull = op.join(figure_dir, 'MDS',figname)
# plt.gcf().savefig(figfull)