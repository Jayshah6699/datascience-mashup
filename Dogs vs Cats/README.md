# Dogs vs Cats 

This Project is based on **Convolutional Neural Networks**. Python Programming language has been used to train and test the model. It recognizes the images of Dogs and Cats. I have used ***Google Colab*** to train and test the model as we can train the model using GPU in Colab and that is very fast as compared to CPU. The more number of epochs takes more time in training the model. A **convolution** multiplies a matrix of pixels with a filter matrix or ‘kernel’ and sums up the multiplication values. Then the convolution slides over to the next pixel and repeats the same process until all the image pixels have been covered.

## Dataset

I have used the dataset of the project from Kaggle. The **train folder** contains ***25,000 images of dogs and cats***. Each image in this folder has the label as part of the filename. The **test folder** contains ***12,500 images*** named according to a numeric id. For each image in the test set, we would predict a probability that the image is a dog (1 = dog, 0 = cat).
 
 [Link for the dataset](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) 

## Required Python Libraries (Installation using pip command)
- numpy : `pip install numpy`
- keras : `pip install keras`
- matplotlib : `pip install matplotlib`
- opencv : `pip install opencv-python`

## About the Model
- Importing datasets from Kaggle

- Importing libraries

- Visualisation of the dataset
<img src="https://user-images.githubusercontent.com/62782231/104853225-6cdcbe00-5925-11eb-93ab-0327f1f9134a.png">

- Making a model using Keras

- Compiling the Model

- Training the model
<img src = "https://user-images.githubusercontent.com/62782231/104853759-ec1fc100-5928-11eb-88f5-a31e3b0e49a1.png">

- Saving the Model

- Summary of the model

- Plotting the Train and Validation Curves
<img src = "https://user-images.githubusercontent.com/62782231/104853801-3c971e80-5929-11eb-88de-0ff55b91d7ab.png">

## Output of the Model

<img src="https://user-images.githubusercontent.com/62782231/104853903-ee364f80-5929-11eb-8982-87592ca12ecc.png">
