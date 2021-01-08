**Natural Language Processing (or NLP)** is applying Machine Learning models to text and language. Teaching machines to understand what is said in spoken and written word is the focus of Natural Language Processing.While NLP is a vast field, here I have used some preprocessing techniques and **Bag of Words** model.

**Data Import**
For importing the data, we are using pandas read_csv() method.

**Cleaning and Preprocessing**
Real-world text contain excessive punctuation,  multiple spaces, etc. We’ll try to normalize most of it.
We’ll do it using [Regular Expressions](https://docs.python.org/3/library/re.html), and  [Natural Language Toolkit](https://www.nltk.org/).

**Tokenization**
Now that our reviews are “clean”, we can further prepare them for our Bag of Words model. Convert them to lowercase letters and split them into individual words. This process is known as tokenization.

The last step of our pre-processing is to remove stop words using those defined in the NLTK library. **Stop words** are usually frequently occurring words that might not significantly affect the meaning of the text. Some examples in English are: “is”, “the”, “and”.An additional benefit of removing stop words is speeding up our models since we’re removing the amount of train/test data.

**Naive Bayes**
Naive Bayes models are probabilistic classifiers that use the Bayes theorem and make a strong assumption that the features of the data are independent.

The *Bayes theorem* is defined as:
![Bayes Theorem](https://miro.medium.com/max/310/1*wIiRjb6thTR2xkTJs0sH0Q.png)

where A and B are some events and P(.) is a probability.
This equation gives us the conditional probability of event A occurring given B has happened. In order to find this, we need to calculate the probability of B happening given A has happened and multiply that by the probability of A (known as Prior) happening. All of this is divided by the probability of B happening on its own.
The naive assumption allows us to reformulate the Bayes theorem for our example as:
![enter image description here](https://miro.medium.com/max/688/1*2aHJ2sqOQQcjZEmEyfgKUw.png)

**Data split and evaluation**
 The train_test_split module is for splitting the dataset into training and testing set.GaussianNB implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features is assumed to be GaussianWe have built a GaussianNB classifier. The classifier is trained using training data. We can use fit() method for training it. After building a classifier, our model is ready to make predictions. We can use predict() method with test set features as its parameters.


