<h1 align="center">
Amazon Reviews Sentiment Analysis using Linear Classifiers
</h1>

Implemented and compared three types of linear classifiers to use for **sentiment analysis** of Amazon product reviews.

The goal of this project is to design a classifier to use for sentiment analysis of product reviews. The training set consists of reviews written by Amazon customers for various food products. The reviews, originally given on a 5 point scale, have been adjusted to a +1 or -1 scale, representing a positive or negative review, respectively.

Below are two example entries from the dataset. Each entry consists of the review and its label. The two reviews were written by different customers describing their experience with a sugar-free candy.

|   Review	                                                                                            |   label   |
|-------------------------------------------------------------------------------------------------------| :-------: |
| Nasty No flavor. The candy is just red, No flavor. Just plan and chewy. I would never buy them again	|    −1     |
| YUMMY! You would never guess that they're sugar-free and it's so great that you can eat them pretty much guilt free! i was so impressed that i've ordered some for myself (w dark chocolate) to take to the office. These are just EXCELLENT!  |      1     |
 

In order to automatically analyze reviews, I did the following tasks:
* The three classifiers used: the **perceptron** algorithm, the **average perceptron** algorithm, and the **Pegasos** algorithm.
* Use your classifiers on the **food review** dataset, using some simple text features.
* Experiment with additional features and explore their impact on **classifier performance**.

-------

**Setup Details:**

For this project, I used **Python 3.6** with some additional libraries. I strongly recommend that you take note of how the NumPy numerical library is used in the code provided, and read through the on-line NumPy tutorial. NumPy arrays are much more efficient than Python's native arrays when doing numerical computation. In addition, using NumPy will substantially reduce the lines of code you will need to write.

1. *Note on software: For this project, you will need the **NumPy** numerical toolbox, and the **matplotlib** plotting toolbox.*

2. Clone/download the repo and unzip into a working directory. The *Amazon Reviews Sentiment Analysis using Linear Classifiers* folder contains the following:
  * *dataset* folder which contains various data files in .tsv format
  * **project1.py** contains various useful functions and function templates that I used to implement the learning algorithms.
  * **main.py** is a script skeleton where these functions are called ran my experiments.
  * **utils.py** contains various utility functions used throughout the project
  * **test.py** is a script which runs tests on a few of the methods I implemented. Note that these tests are provided just to help debug the implementation. Feel free to add more test cases locally to further validate the correctness of the code.
  * **main.ipynb** is the same as *main.py* and is just for viewing the results of the classifiers.
  
**How to Test Locally:** 

In your terminal, navigate to the directory where your project files reside. Execute the command `python test.py` to run all the available tests.

**How to Run the Project 1 Functions Locally:** 

In your terminal, enter `python main.py`.
