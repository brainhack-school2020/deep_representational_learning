from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from helpful_functions import average_subjects_rdms, normalise_dist, get_stimuli, reorder_rdm, define_labels_colors
import os.path as op
import scipy
import scipy.spatial.distance
import scipy.io
import pingouin as pg
from show_VGG16_RDMs import show_activation_layer_rdm
import numpy as np
import pandas as pd
import os
import os.path as op
import glob
import seaborn as sns


#   # # # # # # # INITIALIZE  PATHS, LOAD STIMULI & AVERAGED (RDM) DATA, DEFINE TIME AXIS
general_dir           = '/home/adf/faghelss/CharestLab/brainhackschool/deep_representational_learning/'
figure_dir            = op.join(general_dir,'analysis/models/models_rdm_visualisation')
output_layer_rdms_dir = op.join(general_dir,'analysis/models/models_layer_rdms')
image_dir             = op.join(general_dir,'stimuli/')

# experimental stimuli list
nimg         = 49
images_files = glob.glob(os.path.join(image_dir, '*.png')).sort()
image_order  = np.asarray(list(range(1,nimg+1)))-1


# file for a group-averaged representational dissimilarity matrix (RDM): this was averaged from N=23 participants in "check_rdms.py" script
rdm_file     = op.join(general_dir,'derivatives/output_rdms.npy')

# time vector from matlab (filedtrip preprocessing)
times        = scipy.io.loadmat(op.join('time_vector.mat'))['time'][0] # load time vector
time_range_1 = np.logical_and(times<.0, times>-.2) # this is the baseline

# load group-averaged representational dissimilarity matrix (RDM), clear out baseline
human_rdms_timecourse = np.load(rdm_file)
m_baseline            = human_rdms_timecourse[:,time_range_1].mean(axis=1).squeeze()
for time in list(range(0,len(times))): human_rdms_timecourse[:,time] = np.subtract(human_rdms_timecourse[:,time],m_baseline) +.5

# load model, we start with vgg16 because it's handy and appropriate for vision models 
model           = VGG16(weights='imagenet', include_top=True)
nb_layers       = len(model.layers)
layers_interest = list(range(1,nb_layers)) # 1 cause we skip the input layer

# initialize activation layer variable as a dict, create layers_names dict (useful for plotting later)
count        = 0
layers_names = []
activation_layers = dict()
for lay in layers_interest:
    count =+ 1
    layers_names.append(model.layers[lay].name)
    activation_layers[model.layers[lay].name] = {}


# initialize dataframe for correlations
kendall_timec_df = pd.DataFrame()
pval_df          = pd.DataFrame()
corr_timecourse = dict()

# # # # # # # CORRELATE DCNN & BRAIN RDMs THROUGH TIME  (EEG from -.2 to .65 after image onset) # # #  

# # #  COMPUTE KENDALL's TAU CORRELATION COEFFICIENT # # # 
for lay in layers_interest:
    layer_name = model.layers[lay].name # e.g. block1_conv1

    # load layer's RDM to correlate with human brain RDMs
    corr_rdms_df = pd.read_pickle(op.join(output_layer_rdms_dir,f'rdm_{layer_name}.pkl')) 
    corr_rdms_array = normalise_dist(scipy.spatial.distance.squareform(1-(np.asarray(corr_rdms_df))))

    pearson_timecourse=[]
    pvals_timecourse=[]
    corr_timecourse[layer_name]   = {}
    for this_slice in range(len(times)):
        human_rdm = normalise_dist((human_rdms_timecourse[:,this_slice].squeeze()))
        human_rdm_df = pd.DataFrame(data=human_rdm)
        y = (human_rdm.squeeze())
        x = (corr_rdms_array.squeeze())


        r, p = scipy.stats.kendalltau(x.flatten(), y.flatten())
        pearson_timecourse.append(r)
        pvals_timecourse.append(p)

    temp_df          = pd.DataFrame({f'{layer_name}': pearson_timecourse})
    temp_val_df      = pd.DataFrame({f'{layer_name}': pvals_timecourse})
    kendall_timec_df = pd.concat([kendall_timec_df,temp_df],axis=1)
    pval_df    = pd.concat([pval_df ,temp_val_df],axis=1)


# # #  COMPUTE PARTIAL CORRELATION HERE USING PENGUIN AND PANDAS # # # 

# create panda data frames for every rdms anc concatenate in big_df for partial correlation with penguin
vgg_rdms_all = pd.DataFrame()
big_df = pd.DataFrame()
for lay in layers_interest:
    layer_name = model.layers[lay].name # e.g. block1_conv1

    # save layer's RDM to correlate with human brain RDMs
    temp_df = pd.read_pickle(op.join(output_layer_rdms_dir,f'rdm_{layer_name}.pkl'))
    temp_df = normalise_dist(scipy.spatial.distance.squareform(1-(np.asarray(temp_df))))
    temp_df = pd.DataFrame(temp_df.flatten(), index=range(0,1176),columns=[layers_names[lay-1]])
    vgg_rdms_all = pd.concat([vgg_rdms_all, temp_df],axis=1)

human_rdms_all = pd.DataFrame()
for this_slice in range(len(times)):
    temp=pd.DataFrame(human_rdms_timecourse[:,this_slice].squeeze(), index=range(0,1176),columns=[f'{times[this_slice].round(3)}'])
    human_rdms_all = pd.concat([human_rdms_all,temp],axis=1)

big_df= pd.concat([vgg_rdms_all,human_rdms_all],axis=1) # big data frame containing everything

# # # here we compute the partial correlation between humand and dcnn rdms, with layer "conv_out" thrown out of covariance # # # # # # ## # #
def partial_corr_brainxdcnn(conv_out=0):

    layers_interest          = list(range(1,nb_layers))
    layers_interest          = np.delete(layers_interest,[conv_out])

    partialKendall_timec_df  = pd.DataFrame()
    partial_pval_df          = pd.DataFrame()
    partial_corr_timecourse  = dict()
    # correlate brain and dcnn rdms
    for lay in layers_interest:
        
        layer_name = model.layers[lay].name # e.g. block1_conv1

        print(f"{layer_name}")
        # save layer's RDM to correlate with human brain RDMs
        corr_rdms_df = pd.read_pickle(op.join(output_layer_rdms_dir,f'rdm_{layer_name}.pkl'))
        corr_rdms_array = normalise_dist(scipy.spatial.distance.squareform(1-(np.asarray(corr_rdms_df))))

        partialK_timecourse=[]
        partialK_pvals_timecourse=[]
        partial_corr_timecourse[layer_name]   = {}
        for this_slice in range(len(times)):
            coeffs_temp = pg.partial_corr(data=big_df ,x=f"{layer_name}", y=f"{times[this_slice].round(3)}",
            covar=[layers_names[conv_out]],method = 'kendall')
            partialK_timecourse.append(coeffs_temp.r.kendall)
            partialK_pvals_timecourse.append(coeffs_temp['p-val'].kendall)
        
        temp_df          = pd.DataFrame({f'{layer_name}': partialK_timecourse})
        temp_val_df      = pd.DataFrame({f'{layer_name}': partialK_pvals_timecourse})
        partialKendall_timec_df = pd.concat([partialKendall_timec_df,temp_df],axis=1)
        partial_pval_df     = pd.concat([partial_pval_df  ,temp_val_df],axis=1)
    return partialKendall_timec_df, partial_pval_df

def plot_brain_dnn_similiarity(corr_df,pval_df,nb_layers):
    # set seaborn style, or put black background
    sns.set()
    sns.set(font_scale=2) 
    plt.style.use("dark_background")

    # set critical p-values to assess significance, here we Bonferonni-corrected 
    crit_pval = (1E-40)/(218*nb_layers)
    list_pointx   = np.asarray(list(range(0,218)))
    signif_points = np.asarray(pval_df.block5_conv3<crit_pval)
    list_pointx[signif_points]

    ax, fig  = plt.subplots(figsize=(22,6))

    sns.set_palette(sns.color_palette("Blues", nb_layers)) # rocket, "muted purple"
    fig = sns.lineplot(data=corr_df,dashes=False,linewidth=2)
    # plt.legend(layers_names, ncol=2, loc='upper left')
    # plt.legend(layers_names, ncol=2, loc='upper left')
    fig.get_legend().remove()
    fig.axvline(51, color='w', linestyle='--')
    fig.xaxis.grid(False)
    fig.yaxis.grid(False)
    fig.set(xticks=[51,91,131,171, 211])
    fig.set(xticklabels=times[list([51,91,131,171, 211])].round(2))#round(times[list([20,60,100,140,180,217])],2)
    plt.title(f'timecourse of similarity between brain & DCNN ({model.name}) representations  \n (darker tones --> deeper layers)',fontsize=20)
    plt.xlabel('time from onset(s)',fontsize=15)
    plt.ylabel('similarity to brain representation (kendall''s tau)',fontsize=19)
    # plt.plot(list_pointx[signif_points], np.zeros(times[signif_points].shape), linewidth=2, color='gray')
    # show the time course association of each layer's RDM and brain RDM and save as figure
    figure_name=f'brain_x_{model.name}_timecourse.png'
    print(figure_name)
    figfull = op.join(figure_dir,figure_name)
    plt.gcf().savefig(figfull)
    plt.show()

conv_out=1

plot_brain_dnn_similiarity(corr_df=kendall_timec_df,pval_df=pval_df,nb_layers=22)

max_layers=np.asarray((normalise_dist(kendall_timec_df.max())+.1)*50)

layers_names[conv_out]
partialKendall_timec_df, partial_pval_df = partial_corr_brainxdcnn(conv_out=conv_out)



sns.set(font_scale=2)
# here we plot when each layer peaks with brain representations
sns.set_style("darkgrid")
plt.style.use("dark_background")
sns.set_palette(sns.color_palette("Blues",nb_layers)) # rocket,

ax, fig  = plt.subplots(figsize=(18,18))

ax = sns.stripplot(x=times[list(kendall_timec_df.idxmax())], y=layers_names,size=30,
                   edgecolor="gray",linewidth=.8, alpha=1)
ax.invert_yaxis()
# Make the grid horizontal instead of vertical
ax.xaxis.grid(True)
ax.yaxis.grid(False)
ax.set_xlim(-.2,.65)
plt.title(f'occurence of peak brain x DCNN similarity as a function of model hierarchy',fontsize=29)
plt.ylabel(f'{model.name} hierarchy \n ----------------------------> deeper layers ',fontsize=20)
plt.xlabel('occurence of peak similarity  (s)',fontsize=29)
figure_name=f'brain_x_{model.name}_timepeaks.png'
print(figure_name)
figfull = op.join(figure_dir,figure_name)
plt.gcf().savefig(figfull)
plt.show()