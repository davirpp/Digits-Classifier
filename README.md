# Handwritten Digits Classifier

Exists a very famous dataset called *MNIST digits classification*. This dataset contains images of handwritten digits from 0 to 9. Each instance is a 784 values in range 0 to 255 that represents the pixel intensity at a given location(every image is 28x28 size). 

But here I'm gonna try a different approach. Instead of use "784 features" (28x28), I'm gonna apply a Dimension Reduction and use only 2 features that will be calculated __Intensity__ and __Symmetry__ of the images.

Other different aproach is that I'm gonna use only linear classifiers to classify the digits. I'm gonna use __Linear Regression__, __Logistic Regression__ and __Perceptron__ all algorithms __made by me__. And classify the digits 0, 1, 4 and 5.

Disclaimer: The data was previous manipulated to keep a "default behavior" that will be seen in the Notebook.
