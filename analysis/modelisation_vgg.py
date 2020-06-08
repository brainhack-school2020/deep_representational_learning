from keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from helpful_functions import normalise_dist
from show_VGG16_RDMs import show_activation_layer_rdm
import seaborn as sns
import scipy 
import numpy as np
import pandas as pd
import os
import os.path as op
import glob

# general path of the repository
general_dir           = '/home/adf/faghelss/CharestLab/brainhackschool/deep_representational_learning/'
figure_dir            = op.join(general_dir,'analysis/models/models_rdm_visualisation')
predicted_imgs_dir    = op.join(general_dir,'analysis/models/models_prediction_examples')
output_layer_rdms_dir = op.join(general_dir,'analysis/models/models_layer_rdms')
image_dir             = op.join(general_dir,'stimuli/')

# experimental stimuli directory
nimg = 49
images_files = glob.glob(os.path.join(image_dir, '*.png'))
images_files.sort()
image_order = np.asarray(list(range(1,nimg+1)))-1

# image description of the model used : here VGG16
model=VGG16()
plot_model(model, to_file=f'{model.name}.png')

# get model and weights (trained from imagenet)
model = VGG16(weights='imagenet', include_top=True)
nb_layers=len(model.layers)
layers_interest=list(range(1,nb_layers))

# try out some predictions of our stimuli, save as image in predicted_imgs_dir
prediction_imgs = [16, 29,32, 42, 47] # some examples. 32, 42, 47, faces [1:24] won't work (absent from imagenet..)
img_number      = prediction_imgs[1]
one_img_path    = op.join(image_dir,f'stim_0{img_number}.png')
img             = image.load_img(one_img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features         = model.predict(x)
prob             = "{:.2f}".format(decode_predictions(features,top=1)[0][0][2])
classified_as    = decode_predictions(features,top=1)[0][0][1]

# show and save figure
fig, ax = plt.subplots(figsize=(6,6))
fig = plt.imshow(img)
ax.title.set_text(f'{model.name} predicted image as a <<{classified_as}>> at {prob} probability')
plt.xticks([], [])
plt.yticks([], [])
figure_name=f'predicted_{img_number}.png'
print(figure_name)
figfull = op.join(predicted_imgs_dir,figure_name)
plt.gcf().savefig(figfull)
plt.close('all')
plt.show()


# # # # # here we feed all our stimuli to the dcnn, and extract each layers' activations from this input (stored in "activation_layers") # # # # # #
activation_layers = dict()
for lay in layers_interest:
    activation_layers[model.layers[lay].name] = {}

model.summary() 

for that_img in image_order:
    one_img_path= images_files[that_img]
    print(f'modelling img {that_img} activation through vgg16')

    img = image.load_img(one_img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    for lay in layers_interest:
        

        layer_name = model.layers[lay].name # e.g. block1_conv1

        get_layer_activation = tf.keras.backend.function(
        [model.input],
        [model.get_layer(layer_name).output]
        )

        act = get_layer_activation(x)
        activ = np.asarray(act).squeeze()#.mean(axis=0)

        activation_layers[layer_name][f'img_{that_img}'] = activ.flatten()

# # # # # # we now compupte the dissimilarity between the layers activation values for each pairs of images, 
# # # # # # creating a dissimilarity matrix per layer of the dcnn which we show as a figure

def show_activation_layer_rdm(activation_layers,layer_name):
    df = pd.DataFrame(activation_layers[layer_name])

    # Compute the correlation matrix
    corr = df.corr()
    rdm  = 1 - corr
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    sns.set()
    # plt.style.use("white_background")
    # sns.set(style="dark")

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    fig = sns.heatmap(rdm, cmap='rocket', center=0, #vmin=.1, vmax=1 , mask=mask,
                square=True, linewidths=.25, cbar_kws={"shrink": .5})

    fig.set_yticklabels(fig.get_yticklabels(), rotation=45)
    plt.ylabel('visual stimuli')
    plt.xlabel('visual stimuli')
    figure_name=f'RDM_{layer_name}.png'
    print(figure_name)
    figfull = op.join(figure_dir,figure_name)
    plt.gcf().savefig(figfull)
    plt.close('all')
    return corr            

for lay in layers_interest:
    layer_name = model.layers[lay].name # e.g. block1_conv1
    corr_rdm = show_activation_layer_rdm(activation_layers,layer_name)
    # save layer's RDM to correlate with human brain RDMs
    corr_rdm.to_pickle(op.join(output_layer_rdms_dir,f'rdm_{layer_name}.pkl'))
    # show the layer's RDM figures and save them
