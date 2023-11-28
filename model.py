import datetime
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer, word_tokenize
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

#connect to a database
def connectToDB():
    pass

#TO DO
#Implement time frequency so line plot for sentiment doesn't look like a box
#Do extra preprocessing to remove some things like numbers and non-words
#Might also want to remove handles with the @ and hashtags, could potentially remove @'s by removing words ending in a colon

#build a wordcloud for the specified data
def wordcloud(dirpath: str, words: list):
    print('Creating wordcloud...')
    wordcloud = WordCloud(width=600, 
                     height=400, 
                     random_state=2, 
                     max_font_size=75).generate(words)
    #plot the wordcloud
    plt.figure(figsize=(10, 7))
    plt.imsave(dirpath+'/wordcloud.png', wordcloud)
    print('Done creating wordcloud!')

#perform frequency analysis on the specified data
def frequency_analysis(dirpath: str, words: list):
    print('Starting frequency analysis...')
    #word frequency analysis
    words = word_tokenize(words)
    fd = FreqDist(words)
    top_10 = dict(fd.most_common(10))
    #plot the 10 most common words with their frequencies
    plt.bar(*zip(*top_10.items()))
    plt.ylabel('Frequency')
    plt.title('Top 10 most Frequent Words')
    plt.tight_layout()
    plt.savefig(dirpath+'/frequencies.png')
    plt.clf()
    print('Done with frequency analysis!')
    return set(top_10)

def sentiment_checker(compound):
    return 'positive' if compound >0 else 'neutral' if compound==0 else 'negative'

#perform sentiment analysis on the specified data
def sentiment_analysis(df: pd.DataFrame, dirpath: str, time_freq: str):
    print('Starting sentiment analysis...')
    #sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    df['polarity'] = df['text_string_lem'].apply(analyzer.polarity_scores)
    df = pd.concat([df[['Text', 'Created_at']], df['polarity'].apply(pd.Series)], axis=1)
    df['sentiment'] = df['compound'].apply(sentiment_checker)
    #plot the results
    time_groups = df.groupby(pd.Grouper(key='Created_at', freq=time_freq))
    mean_values = time_groups['compound'].mean()
    ax = mean_values.plot(kind='line')
    yabs_max = abs(max(ax.get_ylim(), key=abs))
    ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    plt.xlabel('Time')
    plt.ylabel('Sentiment')
    plt.title('Lineplot of Sentiment Over Time')
    plt.tight_layout()
    plt.savefig(dirpath+'/sentiment_lineplot.png')
    plt.clf()
    plt.tight_layout()
    sns.boxplot(y='compound', x='sentiment', data=df[df['sentiment'] != 'neutral']).set(title='Boxplot of Sentiment with Compound Score')
    plt.savefig(dirpath+'/sentiment_boxplot.png')
    plt.clf()
    print('Done with sentiment analysis')
    return df

def heatmap(df: pd.DataFrame, dirpath: str, top_10: set, time_freq: str):
    print('Creating heatmap...')
    #finds the words from the top_10 that are in each tweet
    df['top_10_only'] = df['Text'].apply(lambda x: set(x.split(' ')).intersection(top_10))
    #create a new dataframe
    data = df[['Created_at', 'top_10_only', 'compound']]
    #only include the rows which have a word in the top_10
    data = data[data['top_10_only'].apply(len) > 0]
    #explode the top_10 words into separate rows and drop entries with no top_10 words
    data = data.explode('top_10_only').dropna()
    #group the data by time and word
    time_word_groups = data.groupby([pd.Grouper(key='Created_at', freq=time_freq), 'top_10_only'])
    #find the mean compound score of each time and word group
    mean_values = time_word_groups['compound'].mean()
    #unstack the data into a format to be converted to a heatmap
    heatmap_data = mean_values.unstack(fill_value=0)
    # Creating the heatmap
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".2f", center=0)
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label('Compound Score')
    plt.title('Compound Scores by Time and Word')
    plt.xlabel('Word')
    plt.ylabel('Time')
    plt.xticks(rotation=45)
    y_ticks = heatmap.get_yticklabels()
    for tick in y_ticks:
        tick.set_text(tick.get_text()[:-10].replace('T', ' '))
    heatmap.set_yticklabels(y_ticks)
    plt.tight_layout()
    plt.savefig(dirpath+'/heatmap.png')
    plt.clf()
    # print('Done creating heatmap!')


def remove_stopwords(x):
    stop_words = set(stopwords.words('english'))
    return [item for item in x if item not in stop_words]

#preprocess the data and return the preprocessed dataframe with all steps saved as columns
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print('Preprocessing...')
    #data preprocessing for analysis
    #convert everything to lowercase
    df['Text'] = df['Text'].astype(str).str.lower()
    #convert time to datetime
    df['Created_at'] = pd.to_datetime(df['Created_at'])
    #remove regular expressions
    regexp = RegexpTokenizer('\w+')
    df['text_token'] = df['Text'].apply(regexp.tokenize)
    #remove stopwords
    df['text_token'] = df['text_token'].apply(remove_stopwords)
    #remove words with length < 2
    df['text_string'] = df['text_token'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
    #create a frequency distribution
    all_words = ' '.join([word for word in df['text_string']])
    tokenized_words = word_tokenize(all_words)
    fdist = FreqDist(tokenized_words)
    #remove all words that occur < 1 time
    df['text_string_fdist'] = df['text_token'].apply(lambda x: ' '.join([item for item in x if fdist[item] >= 1 ]))
    #lemmatize the words to standardize their format
    wordnet_lem = WordNetLemmatizer()
    df['text_string_lem'] = df['text_string_fdist'].apply(wordnet_lem.lemmatize)
    print('Done preprocessing!')
    return df

#function that analyzes a set of data from a given table and between specified dates
def analyze(table: str, maxtime: datetime, mintime: datetime, save_df: bool, time_freq: str) -> pd.DataFrame:
    nltk.download(['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'vader_lexicon'])
    #create a new file for the graphs to be created
    parent_path = os.getcwd()
    dirname = table + "_"+ str(mintime).replace('/', '-') + "_" + str(maxtime).replace('/', '-')
    dirpath = os.path.join(parent_path, dirname)
    if not os.path.exists(dirpath): os.mkdir(dirpath)

    #craft a sequel query for the specified table and time range
    sql_query:str = f'SELECT text, datetime FROM {table} WHERE datetime <= {maxtime} AND datetime >= {mintime}'

    #get the data from the sql query in a dataframe
    #df:pd.DataFrame = pd.read_sql(sql=sql_query, conn=conn, parse_dates=["datetime"])
    df = pd.read_csv('sample.csv')
    
    #preprocess the data
    df = preprocess(df)

    all_words_lem = ' '.join([word for word in df['text_string_lem']])
    
    #create a wordcloud
    wordcloud(dirpath, all_words_lem)

    #perform some frequency analysis
    top_10 = frequency_analysis(dirpath, all_words_lem)

    #perform sentiment analysis with vader
    df = sentiment_analysis(df, dirpath, time_freq)

    #generate a heatmap for the top 10 most common words and their associated sentiment scores
    heatmap(df, dirpath, top_10, time_freq)

    if save_df: df.to_csv(dirpath+'/output.csv')

    return df

start = time.time()
analyze('hello', '10/16/2019', '10/15/2019', True, '30T')
print('Total Time:', time.time()-start)