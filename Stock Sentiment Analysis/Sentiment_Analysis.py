from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import pandas_datareader as pdr

#Scraping data
finviz_url="https://finviz.com/quote.ashx?t="
tickers=['TSLA','GOOGL','AMZN']

news_tables={}
for ticker in tickers:
    url=finviz_url+ticker

    req= Request(url=url, headers={'user-agent':'my-app'})
    response=urlopen(req)

    html= BeautifulSoup(response,'html')

    news_table=html.find(id='news-table')
    news_tables[ticker]=news_table

#parsing data
parsed_data=[]
for ticker, news_table in news_tables.items():

    for row in news_table.findAll('tr'):

        title=row.a.get_text()

        time_data=row.td.text.split(' ')
        if len(time_data)==1:
            time=time_data[0]
        else:
            date=time_data[0]
            time=time_data[1]

        parsed_data.append([ticker, date, time,title])


#Sentiment analysis
df=pd.DataFrame(parsed_data, columns=['ticker','date','time','title'])
vader=SentimentIntensityAnalyzer()

df['compound']=df['title'].apply(lambda title: vader.polarity_scores(title)['compound'])


#Visualization
df['date']=pd.to_datetime(df.date).dt.date
plt.figure(figsize=(10,8))
mean_df=df.groupby(['ticker','date']).mean()
mean_df=mean_df.unstack()
mean_df=mean_df.xs('compound',axis='columns').transpose()
mean_df.plot(kind='bar')
plt.show()


end=datetime.datetime.today()
DD = datetime.timedelta(days=7)
start=end - DD
end=end.strftime("%Y-%m-%d")
start=start.strftime("%Y-%m-%d")
tsla_df = yf.download('TSLA',
                      start=start,
                      end=end ,
                      progress=False)
tsla_df.head()
plt.figure(figsize=(10,10))
plt.plot(tsla_df.index, tsla_df['Close'])
plt.plot(mean_df.index, mean_df['TSLA']*100+800)
plt.xlabel("date")
plt.show()


















