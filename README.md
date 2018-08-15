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

Histogram equalization is also applied to the dataset and the signs are visually much sharper, especially for images that are originally very dark. Yet the result didn't show any improvement or deterioration.

Class Balancing
---
Because the images in dataset are not evenly distributed among all 43 classes and this reduces the accuracy of network for predicting signs with fewer samples. Therefore, I upsampled them to achieve a uniform calss distribution. Instead of adding same images over, I used warpperspective to transformed images to right or left in a random small angle.

Model Architecture
---
I modified the standard LeNet architecture with deeper layers and removed one of the fully connected layer of LeNet:
* Convolutional Layer 1 : 5x5 Filter with depth 12, activation function:relu, Maximalpooling
* Convolutional Layer 2 : 5x5 Filter with depth 32, activation function:relu, Maximalpooling
* Fully Connected Layer : n = 256, activation function:relu
* Dropout Layer : Dropout Value = 0.5
* Output Layer : n = 43
The architecture is deeper than the standard LeNet as the Traffic Sign Classification Problem is more complex than Numbers Classification. Trying to remove one of the convolutional layers reduced the accuracy of prediction, therefore all tow conbolutional layers are remained. Dropout is also implemented in order to reduce overfitting and the result was best with value 0.5.

Model Training
---
The loss funtion imcludes L2 Regularization of weightmatrix to prevent overfitting. The Nadam optimizer was applied with learnig-rate as 0.001 and coefficient for Regularization as 0.0001 as it converges faster than AdamOptimizer. Different batch-sizes such as 32, 64, 128, 256 are experimented. while 32 ,64 ,128 show similar peformance, larger batch like 256 slows the convergence. Therefore the batchsize was set as 128. The network was trained in 10, 20, 50 and 100 epochs. The best result appear between the 30th and 50th epoch ,the validation accuracy stops improving after that and begin to overfit. ![Learn curve](https://github.com/JiashengYan/CarND-Term1-P2/blob/master/learn_curve.png)

Solution Approach
---
To solve the problem, firstly i used the default LeNet with image preprocessed, yet the result stay around 9.90, then I set the weightmatrix of each layer deeper and the result was improved but still not very satisfying. Then I learned that the balance of classes played a big role and through upsampling under-sampled classes, the validation accuracy improved to 9.95 and test accuracy to 9.53.

* Data Preprocess: First of all, the images were converted to grayscale and values were normalized into (0,1) as both measures improved the accuracy. Histogram equalization was also applied to the dataset and the signs are visually much sharper, especially for images that are originally very dark. Yet the result didn't show any improvement or deterioration. As the data is very unbalanced, I upsampled them to achieve a uniform calss distribution. Instead of adding same images over, I used warpperspective to transformed images to right or left in a random small angle. The training result from the balanced data improved from around 0.91 to 0.94.

* Nadam Optimizer: Both AdamOptimizer and Nadam optimizer were compared in trainning and the Nadam optimizer was faster as it utilizes momentum, but both optimizer yield similar accuracy in the end.

* Overfitting and Underfitting: As shown in the learning curve, the accuracy keeps almost constant after the 30th epoch. Increasing the complexity also didn't improve the predict(Adding a convolutional layer degraded the accuracy from 0.953 to 0.948 and adding a fully connected layer didn't have a noticeable effect), so the network is not underfitting, maybe increasing training data will be effective to improve the result further.

* Hyperparameter Tuning: To let the model to achieve a higher accuracy, I tried different learning rate, regularization coefficient, batch size, epochs, whenever a higher accuracy was reached, I use it as a new baseline and adjust other parameters to see if further improvment is possible. Learning rate higher than 0.001 converge faster at first yet the final accuracy is lower. Learning rate lower than 0.001 converge slower but also can not deliver better result. The optimal value for regularization war found to be near 0.0001 as both higher or lower value degrade the result.

* Model Evaluation: The trained model can be evaluated through the prediction accuracy and loss function on test dataset or the certainty of prediction on new images. The prediction accuracy is the first metric and with similar accuracy, the smaller the loss function, the more stable the model.

Test a Model on New Images
---
It is shown that overall the network can classify signs correctly, yet the probabilities of correct results are still not always very certain as most of the probabilities fail to exceed 50%. As the result shows, the brightness doesn't affect the result much. The contrast and clearity of image should have larger influence on the accuracy. 
