# Summary

I'm a PhD student at Université de Montréal. I use an aggregate of psychophysics, EEG, and computational techniques to understand individual differences in vision, specifically in object/face recognition.


The aim of my project will (most likely) be to make a deep neural network that is closer to human brain representations. This will be in the visual modality.

 
The project will unfold in three steps. 

1) create a custom deep convolutional neural network (DCNN; using  TensorFlow & Keras).
2) create, from high-density EEG time series, group-averaged representational dissimialrity matrices (RDMs; we will be feeding eeg topographies to linear classifers to do so).
3) bias the weights of each of the DCNNs' layers so that it best captures human representations. Roughly, this will be done using the RDMs computed in step 2. 

The first two steps will be developped in parallel. The last step will integrate their output.

# Project definition
# Background

The idea of restraining a DCNN  weight representations is highly inspired by work from Cichy et al., (2014;PNAS) and Kietzmann et al., (2019;PNAS).


# Tools

The structure of the analyses will rely on :

- Python: pyrsa, scikit-learn, tensorFlow & keras, and other visualisation tools.
- We will try to make the repository BIDS friendly as much as possible.

# Data

Description : N = 23 neurotypical human participants; preprocessed EEG recordings (128 electrodes BioSemi); 
~3200 trials per particpant (total trials ~= 73,000).

Task: participants completed a simple one-back task over a stream of images containing faces of different emotions/gender, human-made/natural objects, animals and scenes.
They had to answer (key press) whenever two identical images were shown in a row (e.g. imgA-imgZ-imgR-**imgR**).

![alt text](methods_eeg_oneback.png)


# Deliverables

The plan is to have at least: 


- Python scripts for steps 1 and 3
- Markdown README.md explaining the whole pipeline.

And perhaps :

- A contained that enables to reproduce these analyses


# Results

**Representational Dissimiarlity matrices (RDMs)** have been derived at each time step, and averaged across participants. 
Each of this *stimulus* x *stimulus* matrix indicates the brain's representational model for various visual stimui.

![alt text](rdms_avg_timecourse.png)


2D coordinates of the representational distances from the RDMs of 3 subjects groups were derived with Multi-Dimensional Scaling (MDS).


![alt text](eeg-rsa-mds.gif)





# Progress overview
 This project was first initiated the 19th of May 2020 by Simon Faghel-Soubeyrand as a part of the Brain Hack SChool.


# Tools I learned during this project
 TBD
# Conclusion and acknowledgements
 TBD
