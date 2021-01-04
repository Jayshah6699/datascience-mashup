**Naive Bayes** is a learning algorithm commonly applied to text classification.
Some of the applications of the Naive Bayes classifier are:

**(Automatic) Classification of emails in folders**, so incoming email messages go into folders such as: “Family”, “Friends”, “Updates”, “Promotions”, etc.
**(Automatic) Tagging of job listings.** Given a job listing in raw text format, we can assign it tags such as: “software development”, “design”, “marketing”, etc.
**(Automatic) Categorization of products**. Given a product description, we can assign it into categories such as: “Books”, “Electronics”, “Clothing”, etc.

Background and intuition to build a Naive Bayes classifier:

**Step 1. Identify the prerequisites to train a Naive Bayes classifier**
The only prerequisite is to have an existing set of examples for each category (class) that we wish to bucket/categorize pieces of text into.

**Step 2. Computing the Term-Document Matrix (TDM) for each class**
A term-document matrix (TDM) consists of a list of word frequencies appearing in a set of documents. The TDM matrix is a sparse rectangular matrix of n words and m  documents. And it’s said to be sparse because it contains mostly zeros. The entry (i,j) of the TDM matrix represents the frequency of word “i” in document “j”.

**Step 3. Compute frequencies**
Once the TDM matrices are computed for each class, the next step is to compute the frequency and occurrence of each term in each document.

**Step 4. Recall the Naive Bayes rule**
The probability of an event A happening given that another event B has also happened or in order words, the probability of A given B is
![Probability of A given B](https://miro.medium.com/max/875/1*EbKPutw-9THW0meeQA_2hA.png)

This can also be seen from this Venn Diagram

![enter image description here](https://miro.medium.com/max/875/1*MhvDQ3NI7o1_Oka1N4qj0g.png)

we can see that the probability of A given B is the probability that A and B (the intersection) have happened divided by the probability that B has happened.

Now the probability that B has happened, given that A has also happened
![Image for post](https://miro.medium.com/max/875/1*HpyCz-mxVAqm5C2nGxLkMw.png)

Notice from the Venn diagram that
![enter image description here](https://miro.medium.com/max/875/1*dFw36kPKrTkV7Q7mz6sCfQ.png)

Thus, by equating above equations, we get the Bayes theorem:
![enter image description here](https://miro.medium.com/max/875/1*DrxI1XrTAEMVIMruqUg9hQ.png)

**Step 5. Compute the probability of an incoming text as belonging to any of the given class** 
Let x is a feature vector containing the words coming from the training dataset where
![enter image description here](https://miro.medium.com/max/875/1*26Tx1YxhfrYaq060XbNL9g.png)
The “Naive” assumption that the Naive Bayes classifier makes is that the probability of observing a word is independent of each other. The result is that the “likelihood” is the product of the individual probabilities of seeing each word in the set of given classes.
Given that we computed a database of probabilities for terms appearing in training dataset, we can proceed to the last step of the Naive Bayes Classifier, which is the classification.
The formal decision rule is:
![enter image description here](https://miro.medium.com/max/875/1*X7JStLkiPeI_1zyA3loVuw.png)

What it means is that for every incoming text piece we have to compute the probability of such text for each class, and our final veridic will be given by the largest probability.

Reference: https://towardsdatascience.com/implementing-a-naive-bayes-classifier-for-text-categorization-in-five-steps-f9192cdd54c3
>Written with [StackEdit](https://stackedit.io/).