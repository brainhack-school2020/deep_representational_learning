from keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from helpful_functions import normalise_dist
import numpy as np
import os
import os.path as op

# image description of the model
model=VGG16()
plot_model(model, to_file='vgg.png')
plt.show()


# general path of the repository
general_dir= '/home/adf/faghelss/CharestLab/brainhackschool/deep_representational_learning/'

# experimental stimuli directory
nimg = 49
image_dir = op.join(general_dir,'stimuli/')

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

activation_layers = dict()
for lay in layers_interest:
    
    print(model.layers[lay].name)
    activation_layers[model.layers[lay].name] = {}


get_layer_activation = tf.keras.backend.function(
[model.input],
[model.get_layer(layer_name).output]
    )


for that_img in list(range(0,nimg)):

one_img_path =op.join(image_dir,f'stim_00{that_img}.png')
img = image.load_img(one_img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

    for lay in layers_interest:

        layer_name = model.layers[lay].name # block1_conv1

        print('Running image in layer', layer_name)

        act = get_layer_activation(x)
        activ = np.asarray(act).squeeze()

        activation_layers[layer_name][f'img_{img_number}'] = activ.flatten()
