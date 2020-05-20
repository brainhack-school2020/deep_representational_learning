from helpful_functions import average_subjects_rdms, normalise_dist, get_stimuli, reorder_rdm
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
import scipy
import scipy.spatial.distance

# create list refering to subjects 
# Super-recognizer group: sub 1-16 
# neurotypical controls (sub 1, 3-8)
subs_sr  = np.asarray(list(range(1,17)))
subs_ctl = np.asarray(list(range(1,9)))
subs_ctl = np.delete(subs_ctl,1)


# load all rdms, group = 1 is for SRs
rdms_all = average_subjects_rdms(subject_array=subs_sr,group=1,session=1)

general_dir= '/home/adf/faghelss/CharestLab/brainhackschool/deep_representational_learning/'

# load stimuli
nimg = 49
image_dir = op.join(general_dir,'stimuli/')
image_data, image_order = get_stimuli(image_dir,nimg)

# define category colors
# categories, category_colors = define_category_mds(cmap, image_order)

# times = scipy.io.loadmat(op.join( 'time_vector.mat'))['time'][0]

# time_label='500ms'
# if time_label == '250ms':
    # time_range = np.logical_and(times<.310, times>.230)
# else:
    # time_range = np.logical_and(times<.600, times>.580)

time_range = np.asarray(list(range(70,150)))

rdms_avg  = np.asarray(rdms_all).mean(axis=0)
rdm_avg_timewindow = rdms_avg[:,time_range].mean(axis=1).squeeze()
rdm_norm_utv = normalise_dist(rdm_avg_timewindow)

# show as a square matrix (original RDM format)
rdm = scipy.spatial.distance.squareform(rdm_norm_utv)

# rdm = reorder_rdm(rdm, image_order)

plt.imshow(rdm, cmap='inferno')
plt.show()


# group='SRs'
# # plot_mds(rdms_avg, image_data, categories, category_colors, scaler=0.02)
# figname = f'MDS_{group}_{time_label}.png'
# figfull = op.join(figure_dir, 'MDS',figname)
# plt.gcf().savefig(figfull)