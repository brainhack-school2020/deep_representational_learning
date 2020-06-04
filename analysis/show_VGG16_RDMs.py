import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def plot_brain_dnn_similiarity(corr_df,pval_df):
    # set seaborn style, or put black background
    sns.set()
    plt.style.use("dark_background")

    # set critical p-values to assess significance, here we Bonferonni-corrected 
    crit_pval = (1E-40)/(218*nb_layers)
    list_pointx   = np.asarray(list(range(0,218)))
    signif_points = np.asarray(pval_df.block1_conv1<crit_pval)
    list_pointx[signif_points]

    ax, fig  = plt.subplots(figsize=(22,6))

    sns.set_palette(sns.color_palette("Purples", nb_layers)) # rocket, "muted purple"
    fig = sns.lineplot(data=kendall_timec_df,dashes=False,linewidth=2)
    plt.legend(layers_names, ncol=2, loc='upper left')
    fig.xaxis.grid(True)
    fig.yaxis.grid(False)
    fig.set(xticks=[20,60,100,140, 180,217])
    fig.set(xticklabels=times[list([20,60,100,140,180,217])].round(2))#round(times[list([20,60,100,140,180,217])],2)
    plt.title(f'timecourse of similarity between brain & DCNN ({model.name}) representations  \n (darker tones --> deeper layers)',fontsize=20)
    plt.xlabel('time from onset(s)',fontsize=15)
    plt.ylabel('similarity to brain representation (kendall''s tau)',fontsize=15)
    plt.plot(list_pointx[signif_points], np.zeros(times[signif_points].shape), linewidth=2, color='gray')
    # show the time course association of each layer's RDM and brain RDM and save as figure
    figure_name=f'brain_x_{model.name}_timecourse.png'
    print(figure_name)
    figfull = op.join(figure_dir,figure_name)
    plt.gcf().savefig(figfull)
    plt.show()


# for lay in layers_interest:
#     layer_name = model.layers[lay].name # e.g. block1_conv1
#     show_activation_layer_rdm(activation_layers,layer_name)
#     figure_name=f'RDM_{layer_name}.png'
#     print(figure_name)
#     figfull = op.join(figure_dir,figure_name)
#     plt.gcf().savefig(figfull)
#     plt.close('all')



