<h1 align="center" style="font-size:60px"><b>Shakespeare Style Text Generation</b></h1>
<p align="center">
  <img src="https://i.ibb.co/dj6rMQF/shakespearetext.png" alt="shakespearetext" width="500" height="250">
<br>
</p>
<p align="center">
<b>Can you write like Shakespeare? <br>
What if we try to train an AI to write like Shakespeare?</b>
</p>

# Text Generation
Text generation is one of the state-of-the-art applications of NLP. Deep learning techniques are being used for a variety of text generation tasks such as writing poetry, generating scripts for movies, and even for composing music.

> The notebook, <b>Shakespeare.ipynb</b> aims at generating text in William Shakespeare's style.

# Model architecture
The model designed to generate text in Shakespeare's style is a sequential model consisiting of <b>Embedding, Bi-direction LSTM, LSTM and a Dense layer.</b>
<p align="center">
<img src="https://i.ibb.co/YT5S2NV/summary.png" alt="summary-LSTM" width="450" height="280" >
</p>
<b>More details about the model is documented inside the notebook itself.</b> <br>

> The model is trained in Google Colab using GPU.

# Examples of some generated text
> 1. The prince  of wales from such a field as this jest and had it in this reproach be done may have it. <br>
> 2. My dear Othello fight picked but all i see you well and even to like thee to my grave my friends and the very good grace of dead talbot come in such a courtesy swain aside to sack.

# Dataset source
> <a href="https://www.kaggle.com/kingburrito666/shakespeare-plays?select=Shakespeare_data.csv">Shakespeare plays: All of shakespeares plays, characters, lines, and acts in one CSV</a>
# Requirments to run the notebook
> 1. TensorFlow 2
> 2. Numpy
> 3. Pandas
> 4. Matplotlib
> 5. WordCloud (Optional)


