from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



file_path = "/content/(1) The Hunger Games.txt"
file_handle = open(file_path).read()
stopwords = set(STOPWORDS)
stopwords.add("will")

image_file = "/content/412hOEZ3jPL._SL1200_.jpg"
# create mask
a_mask = np.array(Image.open(image_file))

wc = WordCloud(background_color="black", mask=a_mask,
               stopwords=stopwords)
wc.generate(file_handle)

plt.figure(figsize=(8,6), dpi=120)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
