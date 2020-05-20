#!/bin/bash

#### Defining paths
topdir=~/CharestLab/brainhackschool/deep_representational_learning
data_rdm=~/CharestLab/brainhackschool/deep_representational_learning/data_rdms_matlab
# dcm2niidir=/Users/franklinfeingold/Desktop/dcm2niix_3-Jan-2018_mac


# Create dir rdms
mkdir ${topdir}/rdms
rdms_dir=${topdir}/rdms


###Create dataset_description.json
jo -p "Name"="SuperRecogniser-U.Birmingham CharestLab Sample Dataset" "BIDSVersion"="1.0.2" >> ${topdir}/dataset_description.json


####Anatomical Organization####
#for subj in 3; do
#	echo "Processing subject $subj"

###Create structure
#mkdir -p ${niidir}/sub-${subj}/ses-1/

###Convert dcm to nii
#Only convert the Dicom folder anat
#for direcs in anat; do
#${dcm2niidir}/dcm2niix -o ${niidir}/sub-${subj} -f ${subj}_%f_%p ${dcmdir}/${subj}/${direcs}
#done


