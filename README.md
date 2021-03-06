# Generation-of-SAR-Images-Using-Deep-Learning

## Paper Link: https://link.springer.com/article/10.1007%2Fs42979-020-00364-z
## Dataset Link: https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=targets

## Overview: 
This work aims to generate new Synthetic Aperture Radar (SAR) Images using Generative Adversarial Networks (GANs). A robust Densely Connected Convolutional Neural Network (DenseNet) model capable of classifying six distinct SAR target classes has also been proposed in this paper. The aim is to use the classifier in order to evaluate the GAN images quantitatively as well as establish an overall proof of concept. (More details can be found in the paper!)

![alt text](https://github.com/MukundSai7907/Generation-of-SAR-Images-Using-Deep-Learning/blob/main/Readme_Images/Overview.png?raw=true)

## DenseNet Structure: 
The end to end DenseNet structure is show below
![alt text](https://github.com/MukundSai7907/Generation-of-SAR-Images-Using-Deep-Learning/blob/main/Readme_Images/DENSE_NET.png?raw=true)


## GAN Structure: 
The discriminator is modelled as
![alt text](https://github.com/MukundSai7907/Generation-of-SAR-Images-Using-Deep-Learning/blob/main/Readme_Images/DIS.png?raw=true)
The generator is modelled as
![alt text](https://github.com/MukundSai7907/Generation-of-SAR-Images-Using-Deep-Learning/blob/main/Readme_Images/GEN.png?raw=true)


## Results:
Some GAN generated images (right) and dataset images (left) are placed next to each other to demonstrate the GAN's effectiveness

![alt text](https://github.com/MukundSai7907/Generation-of-SAR-Images-Using-Deep-Learning/blob/main/Readme_Images/TUR.png?raw=true)
