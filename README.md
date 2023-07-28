# Detection of COVID-19 using chest X-Ray

This project aims to leverage chest X-ray images to develop a predictive model for COVID-19
diagnosis. By analyzing the characteristics and patterns observed in these images, the model
seeks to accurately predict whether a patient is diagnosed with COVID-19. The proposed
approach offers the potential for a faster and more accessible diagnostic method, especially in
areas where RT-PCR testing resources may be limited.
  
## Project Workflow
   
<img src="https://github.com/ACM40960/project-SidBagwe98/assets/134402582/bc2ca92f-cde5-4e02-9ffa-8b46b48beb3c" width="800" height="400" >

The proposed system works in following steps:

1) Identifying and splitting the data by labels which can be used to train our model.
2) Designing a supervised learning model(CNN) which will learn from the different features of the image and provide predictions with the highest accuracy possible.
3) Finally, our model will be evaluated using the test dataset and the results will be noted down which will indicate if the proposed model can be used
   to detect Covid-19 cases. The focus here will be on the false-negatives as the goal is to predict the positive cases of Covid-19 correctly.

## Dataset 

For the purpose of the model we use the following dataset which was taken from kaggle.

Link: https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia

The Dataset consists of 6536 x-ray images and contains three sub-folders
(COVID19, PNEUMONIA, NORMAL). The data is then divided into train, validation and test sets where the test set is 20% of the total data.

<img src="https://github.com/ACM40960/project-SidBagwe98/assets/134402582/56bc077e-4a71-4969-8433-b8c27b65a1e9" width="400" height="300">   <img src="https://github.com/ACM40960/project-SidBagwe98/assets/134402582/9aacff4f-b02d-4dcf-8a74-5f4530ea0a8a" width="400" height="300">

The images above show the X-ray images of a normal person and of a person diagnosed with Covid-19.

## Model

The model designed here uses three convolutional layers with an input shape of (100, 100, 3) and Rectified Linear Unit (ReLU) activation function to introduce
non-linearity in the model. The first layer has 32 filters to convolve over the input image. It also employs dropout and L2 regularization techniques to improve generalization and prevent overfitting. In the final dense layer, the softmax activation function is used to obtain the class probabilities for multi-class classification

<img src="https://github.com/ACM40960/project-SidBagwe98/assets/134402582/5218e838-c526-4c42-aa91-5b5177711971" width="800" height="400">
