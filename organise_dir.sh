#!/bin/bash

#### Defining paths
topdir=~/CharestLab/brainhackschool/deep_representational_learning
data_rdm=~/CharestLab/brainhackschool/deep_representational_learning/data_rdms_matlab



# Create dir rdms
mkdir ${topdir}/rdms
rdms_dir=${topdir}/rdms


###Create dataset_description.json
jo -p "Name"="SuperRecogniser-U.Birmingham CharestLab Sample Dataset" "BIDSVersion"="1.0.2" >> ${topdir}/dataset_description.json


#### Organization ####



