from keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from helpful_functions import normalise_dist
from show_VGG16_RDMs import show_activation_layer_rdm
import numpy as np
import pandas as pd
import os
import os.path as op
import glob

# image description of the model
model=VGG16()
plot_model(model, to_file='vgg.png')
plt.show()


# general path of the repository
general_dir= '/home/adf/faghelss/CharestLab/brainhackschool/deep_representational_learning/'
figure_dir = op.join(general_dir,'analysis/models/models_rdm_visualisation')
output_layer_rdms_dir= op.join(general_dir,'analysis/models/models_layer_rdms')
image_dir = op.join(general_dir,'stimuli/')

# experimental stimuli directory
nimg = 49
images_files = glob.glob(os.path.join(image_dir, '*.png'))
images_files.sort()
image_order = np.asarray(list(range(1,nimg+1)))-1

model = VGG16(weights='imagenet', include_top=True)
nb_layers=len(model.layers)
layers_interest=list(range(1,nb_layers))

# try out a prediction of some of our images. Not working so much at the moment..
img_number = 1
one_img_path =op.join(image_dir,f'stim_00{img_number}.png')
img = image.load_img(one_img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

print('Predicted:', decode_predictions(features,top=2)[0])


# initialize activation layer variable as directory : layer_name
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



def show_activation_layer_rdm(activation_layers,layer_name):
    df = pd.DataFrame(activation_layers[layer_name])

    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    sns.set(style="white")

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    fig = sns.heatmap(1-corr, mask=mask, cmap='viridis', center=0, #vmin=.1, vmax=1
                square=True, linewidths=.5, cbar_kws={"shrink": .5})#annot=True)

    fig.set_yticklabels(fig.get_yticklabels(), rotation=45)
    plt.ylabel('visual stimuli')
    plt.xlabel('visual stimuli')
    return corr            


for lay in layers_interest:
    layer_name = model.layers[lay].name # e.g. block1_conv1
    corr_rdm = show_activation_layer_rdm(activation_layers,layer_name)
    # save layer's RDM to correlate with human brain RDMs
    corr_rdm.to_pickle(op.join(output_layer_rdms_dir,f'rdm_{layer_name}.pkl'))
    # show the layer's RDM figures and save them
    figure_name=f'RDM_{layer_name}.png'
    print(figure_name)
    figfull = op.join(figure_dir,figure_name)
    plt.gcf().savefig(figfull)
    plt.close('all')
