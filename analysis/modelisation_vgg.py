from keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from helpful_functions import normalise_dist
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

# experimental stimuli directory
nimg = 49
image_dir = op.join(general_dir,'stimuli/')
images_files = glob.glob(os.path.join(image_dir, '*.png'))
images_files.sort()
image_order = np.asarray(list(range(1,nimg+1)))-1

model = VGG16(weights='imagenet', include_top=True)
nb_layers=len(model.layers)
layers_interest=list(range(1,nb_layers))


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
    
    print(model.layers[lay].name)
    activation_layers[model.layers[lay].name] = {}

for that_img in image_order:
    one_img_path= images_files[that_img]
    print(that_img)

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

        #print('Running image in layer', layer_name)

        act = get_layer_activation(x)
        activ = np.asarray(act).squeeze()#.mean(axis=0)

        activation_layers[layer_name][f'img_{that_img}'] = activ.flatten()


# layer_name = 'block5_conv1' #model.layers[lay].name # e.g. block1_conv1

# sns.scatterplot(x=activation_layers[layer_name][f'img_{1}'], y=activation_layers[layer_name][f'img_{35}'])
# plt.show()

df = pd.DataFrame(activation_layers['block1_conv1'])

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
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()

