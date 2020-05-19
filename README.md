# the deep_representational_learning repository : Summary

The aim of this project will (most likely) be to make a deep neural network that is close(r) to human brain representations. This will be in the visual modality.

 
The project will unfold in three steps. 

1) create a custom deep convolutional neural network (DCNN; using  TensorFlow & Keras)
2) create, from high-density EEG time series, group-averaged representational dissimialrity matrices (RDMs; we will be feeding eeg topographies to linear classifers to do so).
3) bias the weights of each of the DCNNs' layers so that it best captures human representations. Roughly, this will be done using the RDMs computed in step 2. 

The first two steps will be developped in parallel. The last step will integrate their output.

# Project definition
# Background

The idea of restraining a DCNN  weight representations is highly inspired by work from Cichy et al., (2014;PNAS) and Kietzmann et al., (2019;PNAS).


# Tools

The structure of the analyses  will rely on :

- Python: TensorFlow & Keras
- Matlab : MVPA-light & Fieldtrip toolboxes.
- We will try to make the repository BIDS friendly as much as possible.

# Data

Description : N = 21 neurotypical human participants; EEG recordings (128 electrodes BioSemi); 
~3200 trials per particpant (total trials > 63 000).


# 1) Deep neural networks
# concepts and tools




# 2) Using linear classifiers to generate EEG Representational Dissimialrity Matrices.




# 3) 

