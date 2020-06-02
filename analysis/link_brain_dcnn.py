from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from helpful_functions import average_subjects_rdms, normalise_dist, get_stimuli, reorder_rdm, define_labels_colors
import os.path as op
import scipy
import scipy.spatial.distance
import scipy.io
from show_VGG16_RDMs import show_activation_layer_rdm
import numpy as np
import pandas as pd
import os
import os.path as op
import glob
import seaborn as sns


#   # # # # # # # INITIALIZE  PATHS, LOAD STIMULI & AVERAGED (RDM) DATA, DEFINE TIME AXIS
general_dir= '/home/adf/faghelss/CharestLab/brainhackschool/deep_representational_learning/'
figure_dir = op.join(general_dir,'analysis/models/models_rdm_visualisation')
output_layer_rdms_dir= op.join(general_dir,'analysis/models/models_layer_rdms')
image_dir = op.join(general_dir,'stimuli/')

# experimental stimuli directory
nimg = 49
images_files = glob.glob(os.path.join(image_dir, '*.png'))
images_files.sort()
image_order = np.asarray(list(range(1,nimg+1)))-1

rdm_file = op.join(general_dir,'derivatives/output_rdms.npy')

# the time windows of interest
times = scipy.io.loadmat(op.join('time_vector.mat'))['time'][0] # load time vector
time_range_1 = np.logical_and(times<.0, times>-.2)

# load group-averaged representational dissimilarity matrix (RDM), clear out baseline
human_rdms_timecourse = np.load(rdm_file)
m_baseline = human_rdms_timecourse[:,time_range_1].mean(axis=1).squeeze()
for t in list(range(0,len(times))):
    human_rdms_timecourse[:,t] = np.subtract(human_rdms_timecourse[:,t],m_baseline) +.5

# load model
model = VGG16(weights='imagenet', include_top=True)
nb_layers=len(model.layers)
layers_interest=list(range(1,nb_layers))
# layers_interest = [2,5,9,13,17,10,21,22]

# initialize activation layer variable as directory : layer_name
count=0
layers_names = []
activation_layers = dict()
for lay in layers_interest:
    count=+1
    layers_names.append(model.layers[lay].name)
    activation_layers[model.layers[lay].name] = {}

sns.set()

kendall_timec_df = pd.DataFrame()
palette = sns.color_palette("cubehelix", len(layers_interest)) # sns.palplot(sns.cubehelix_palette(8))8))
corr_timecourse = dict()
# correlate brain and dcnn rdms
for lay in layers_interest:
    layer_name = model.layers[lay].name # e.g. block1_conv1

    # save layer's RDM to correlate with human brain RDMs
    corr_rdms_df = pd.read_pickle(op.join(output_layer_rdms_dir,f'rdm_{layer_name}.pkl'))
    corr_rdms_array = normalise_dist(1-np.asarray(corr_rdms_df))

    pearson_timecourse=[]
    pvals_timecourse=[]
    corr_timecourse[layer_name]   = {}
    for this_slice in range(len(times)):
        human_rdm = normalise_dist(scipy.spatial.distance.squareform(human_rdms_timecourse[:,this_slice].squeeze()))
        human_rdm_df = pd.DataFrame(data=human_rdm)
        x = np.asarray(human_rdm.squeeze())
        y = np.asarray(corr_rdms_array.squeeze())


        r, p = scipy.stats.kendalltau(x.flatten(), y.flatten()) # corr_rdms_df.corrwith(human_rdm_df)
        pearson_timecourse.append(r)
        pvals_timecourse.append(p)

    temp_df = pd.DataFrame({f'{layer_name}': pearson_timecourse})
    temp_df = pd.DataFrame({f'{layer_name}': pearson_timecourse})
    kendall_timec_df = pd.concat([kendall_timec_df,temp_df],axis=1)

sns.set_style("dark")
plt.style.use("dark_background")

ax, fig  = plt.subplots(figsize=(22,6))

sns.set_palette(sns.color_palette("Blues", nb_layers)) # rocket, "muted purple"
fig = sns.lineplot(data=kendall_timec_df,dashes=False)
plt.legend(layers_names, ncol=2, loc='upper left')
fig.set(xticks=[20,60,100,140, 180,217])
fig.set(xticklabels=times[list([20,60,100,140,180,217])].round(2))#round(times[list([20,60,100,140,180,217])],2)
plt.title(f'timecourse of brain & {model.name} representation similarity \n (darker tones --> deeper layers)',fontsize=20)
plt.xlabel('time (s)',fontsize=15)
plt.ylabel('similarity to brain representation (kendall''s tau)',fontsize=15)
# show the time course association of each layer's RDM and brain RDM and save as figure
figure_name=f'brain_x_{model.name}_timecourse.png'
print(figure_name)
figfull = op.join(figure_dir,figure_name)
plt.gcf().savefig(figfull)
plt.show()