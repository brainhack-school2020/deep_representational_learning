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

Description : N = 21 neurotypical human participants; preprocessed EEG recordings (128 electrodes BioSemi); 
~3200 trials per particpant (total trials > 63 000).

Task: 

Participants completed a simple one-back task over a stream of images containing faces of different emotions/gender, human-made/natural objects, animals and scenes.
They had to answer (key press) whenever two identical images were shown in a row (e.g. imgA-imgZ-imgR-*imgR*).

![alt text](./img/methods_eeg_oneback.png=200x375)


# Deliverables

The plan is to have : 


- Python scripts for the steps 2 and 3
- A binder repo that enables to reproduce these analyses


# Results
# Progress overview
 This project was first initiated the 19th of May 2020 by Simon Faghel-Soubeyrand as a part of the Brain Hack SChool.


# Tools I learned during this project
- deep learning
- Python
- using calcul Quebec clusters


# Conclusion and acknowledgements
 
