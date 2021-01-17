# Fake-News-Detector
Fake news Detection using TfidfVectorizer

Dataset : https://s3.amazonaws.com/assets.datacamp.com/blog_assets/fake_or_real_news.csv

- The fake news detector was made using *TfidfVectorizer*  and intializing the *Passive Agressive classiﬁer*.It helps in detection of fake news.
- **TF (Term Frequency)** - The number of times a word appears in a document is its Term Frequency. 
- **IDF** - IDF is a measure of how significant a term is in the entire corpus.
- **TfidfVectorizer** - Transforms text to feature vectors that can be used as input to estimator. 
- **Passive Agressive classiﬁer**-The passive-aggressive algorithms are a family of algorithms for large-scale learning.
- The libraries used are :
  - Numpy
  - Pandas
  - Scikit Learn
  - Itertools
  - Matplotlib and Seaborn(for plotting the confusion matrix)
- The accuracy was predicted taking different values of max_iter and confusion matrix was plotted. Accuracy  is the number of correct   predictions made divided by the total number of predictions made, multiplied by 100 to turn it into a percentage.The confusion matrix  summarizes the performance of a classification algorithm.
- The combination of a TF-IDF Vectorizer and a Passive Aggressive Classifier  gives accuracy of about **93 %**.
