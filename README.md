## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive) 

Overview
---
In this project, I used my learned knowledge about deep neural networks and convolutional neural networks to classify traffic signs ([German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)). Solving this project required training of a deep neural network. I outline the steps I used below. I achieved an Validation accuracy of 0.993 and a Test accuracy of 0.951.



The Dataset
---
The traffic dataset are divided into 43 classes. Here is an overview over the different classes and their corresponding number of images. The size of each image is just 32x32 pixels and some of them are very dark or blurry and the distribution of each class is very unevenly.
![sample_dataset](https://github.com/JiashengYan/CarND-Term1-P2/blob/master/dataset_overview.png)


Preprocessing
---
The following two preprocessing are chosen as they both improved the accuracy of prediction.
1. Convert to grayscale
2. Normalize data into (0,1)

Class Balancing
---
Because the images in dataset are not evenly distributed among all 43 classes and this reduces the accuracy of network for predicting signs with fewer samples. Therefore, I upsampled them to achieve a uniform calss distribution. Instead of adding same images over, I used warpperspective to transformed images to right or left in a random small angle.

Model Architecture
---
I used a modified version of the LeNet architecture with following layers:
* Layer 1 : 5x5 Filter with depth 12
* Layer 2 : 5x5 Filter with depth 32
* Fully Connected Layer : n = 512
* Dropout Layer : Dropout Value = 0.6
* Fully Connected Layer : n = 256
* Dropout Layer : Dropout Value = 0.6
The architecture is deeper than the standard LeNet as the Traffic Sign Classification Problem is more complex than Numbers Classification. Dropout is also implemented in order to reduce overfitting.

Model Training
---
The loss funtion imcludes L2 Regularization besides errors to prevent overfitting. The AdamOptimizer is applied with learnig-rate as 0.001 and coefficient for Regularization as 0.0001. The network are trained in 20 epochs with a batch size of 128. The Validation Accuracy increased quickly in the first 5 epoch to 9.90 and approach 9.95 slowly.

Solution Approach
---
To solve the problem, firstly i used the default LeNet with image preprocessed, yet the result stay around 9.90, then I set the weightmatrix of each layer deeper and the result was improved but still not very satisfying. Then I learned that the balance of classes played a big role and through upsampling under-sampled classes, the validation accuracy improved to 9.95 and test accuracy to 9.53.

Test a Model on New Images
---
It is shown that overall the network can classify signs correctly, yet the probabilities of correct results are still not always very certain
