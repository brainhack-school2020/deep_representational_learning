import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# figure_dir = '~/CharestLab/brainhackschool/deep_representational_learning/analysis/models/models_rdm_visualisation'


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
    sns.heatmap(1-corr, mask=mask, cmap='plasma',vmin=.1, vmax=1.1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

# for lay in layers_interest:
#     layer_name = model.layers[lay].name # e.g. block1_conv1
#     show_activation_layer_rdm(activation_layers,layer_name)
#     figure_name=f'RDM_{layer_name}.png'
#     print(figure_name)
#     figfull = op.join(figure_dir,figure_name)
#     plt.gcf().savefig(figfull)
#     plt.close('all')



