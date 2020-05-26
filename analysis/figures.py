import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
import scipy
import scipy.spatial.distance
import scipy.io
from helpful_functions import average_subjects_rdms, normalise_dist, get_stimuli, reorder_rdm, define_labels_colors

general_dir= '/home/adf/faghelss/CharestLab/brainhackschool/deep_representational_learning/'

# load stimuli
nimg = 49
image_dir = op.join(general_dir,'stimuli/')
rdm_file = op.join(general_dir,'derivatives/output_rdms.npy')

rdms_timecourse = np.load(rdm_file)

# define category colors
# categories, category_colors = define_category_mds(cmap, image_order)
times = scipy.io.loadmat(op.join('time_vector.mat'))['time'][0]

# the time windows of interest : need to make in a loop for the future.
time_range_1 = np.logical_and(times<-.05, times>-.2)
time_range_2 = np.logical_and(times<.09, times>.00)
time_range_3 = np.logical_and(times<.200, times>.120)
time_range_4 = np.logical_and(times<.400, times>.22)


rdm_norm_1 = normalise_dist(rdms_timecourse[:,time_range_3].mean(axis=1).squeeze())

# show as a square matrix (original RDM format),
rdm_1 = (scipy.spatial.distance.squareform(rdm_norm_1))



import plotly.express as px
import matplotlib.colors as colors
from matplotlib import cm

# now we load labels, images, and category colors
image_data, image_order = get_stimuli(image_dir,nimg)
# define category colors
cmap = cm.get_cmap('viridis', 255)
categories, labels_rdm = define_labels_colors(cmap, image_order)

# Use directly Columns as argument. You can use tab completion for this!
# fig = px.scatter(rdm_1, x=xs, y=ys, color=xs, size=2)
fig = px.imshow(rdm_1,labels=dict(x="stimuli images", y="stimuli images", color="pair-wise decoding accuracy"))
                # x=labels_rdm,
                # y=labels_rdm)
fig.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import pylab as pl

# A = np.random.random(size=(5,5))
# fig, ax = plt.subplots(1, 1)

# xl, yl, xh, yh=np.array(ax.get_position()).ravel()
# w=xh-xl
# h=yh-yl
# xp=xl+w*0.1 #if replace '0' label, can also be calculated systematically using xlim()
# size=0.05

# img=mpimg.imread('microblog.png')
# ax.matshow(A)
# ax1=fig.add_axes([xp-size*0.5, yh, size, size])
# ax1.axison = False
# imgplot = ax1.imshow(img,transform=ax.transAxes)

# now try to update this RDMs with interactive button
from ipywidgets import interactive, HBox, VBox
import plotly.offline as py
import numpy as np

py.init_notebook_mode()

rdm_file = op.join(general_dir,'derivatives/output_rdms.npy')

rdms_timecourse = np.load(rdm_file)

def update_time(timems):
    times = scipy.io.loadmat(op.join('time_vector.mat'))['time'][0]
    margins = .05
    # the time window of interest, with a margin. 
    time_range_1 = np.logical_and(times<timems+margins, times>timems-margins)

    # average over time window, and normalise the distances
    rdm_norm_1 = normalise_dist(rdms_timecourse[:,time_range_1].mean(axis=1).squeeze())

    # show as a square matrix (original RDM format),
    rdm_1 = (scipy.spatial.distance.squareform(rdm_norm_1))

    fig = px.imshow(rdm_1,labels=dict(x="stimuli images", y="stimuli images", color="pair-wise decoding accuracy"))
    fig.show()
    # f.data[0].time = times(timems)

time_slider = interactive(update_time, timems=(0, 215, 1))
time_slider

vb = VBox((f, freq_slider))
vb.layout.align_items = 'center'
vb

# f = go.FigureWidget(
#     data=[
#         go.heatmap(
#                     z=matrix_df.transpose().values.tolist(),
#         x=matrix_df.columns[::-1],
#         y=matrix_df.columns[::-1],
#         colorscale='Viridis'
#     ))
# )




# x = y = np.arange(-5, 5, 0.1)
# yt = x[:, np.newaxis]
# z = np.cos(x * yt) + np.sin(x * yt) * 2

# f = go.FigureWidget(
#     data=[
#         go.Surface(z=z, x=x, y=y,
#                    colorscale='Viridis')],
#     layout=go.Layout(scene=go.layout.Scene(
#         camera=go.layout.scene.Camera(
#             up=dict(x=0, y=0, z=1),
#             center=dict(x=0, y=0, z=0),
#             eye=dict(x=1.25, y=1.25, z=1.25))
#     ))
# )


def update_time(timems):
    # times=blablabla
    f.data[0].time = times(timems)


time_slider = interactive(update_time, timems=(1, 50, 0.1))
vb = VBox((f, freq_slider))
vb.layout.align_items = 'center'
vb