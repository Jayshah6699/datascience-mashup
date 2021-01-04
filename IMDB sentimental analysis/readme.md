# Sentimental analysis on IMDB Movies review

[Sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis) is a natural language processing technique used
 to determine whether data is positive, negative or neutral in text.We are using Sentimental
 Analysis for Movies Review.

Predicting the Movies Review

Dataset Preperation

- We load the dataset from keras.dataset
- Divide the dataset into four parts XT,yT,Xt,yt
- We are using 10000 words which are occuring more frequently.

Data Preprocessing
- We create a reverse dictionary in which index map with words.
- vectorize the data.

Modelling and Evaluation
- Build the CNN model.
- To avoid overfitting we use Earlystopping and checkpoint.
- Evaluate our model on test data.
- Predict the results.

<h3>Conclusion</h3>
Movies reveiw is postive if prediction is greater than 0.7 otherwise it is negative.

We got 93% accuracy on train data and  88% on test data.
