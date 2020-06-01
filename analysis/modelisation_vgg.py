from keras.applications.vgg16 import VGG16
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

model = VGG16()
plot_model(model, to_file='vgg.png')
plt.show()