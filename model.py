import datetime
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ssl
import sys
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer, word_tokenize
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer


# print the python version from python --version
print('Python version:', sys.version)

# connect to a database
def connectToDB():
    pass

#build a wordcloud for the specified data

def wordcloud(dirpath: str, words: str):
    print('Creating wordcloud...')
    
    # Combine words into a single string if they are not already
    words_combined = ' '.join(words) if isinstance(words, list) else words
    
    # Define a set of stopwords to exclude from the word cloud
    stopwords = set(STOPWORDS)
    
    # Create the word cloud object with better contrast and visibility settings
    wordcloud_gen = WordCloud(width=2048,
                          height=1536,
                          max_words=200, # Limit the number of words to avoid clutter
                          max_font_size=50, # Adjust max font size to prevent overlap
                          stopwords=stopwords,
                          background_color='black', # Use a background color with high contrast
                          font_path="/Users/charliedeane/Library/Fonts/AmsiProCond-Regular.otf", # Verify the font path
                          colormap='PuRd',
                          ).generate(words_combined)
    
    # Display the word cloud
    plt.figure(figsize=(18, 12)) # Increase the size of the resulting plot
    plt.axis("off")
    plt.imshow(wordcloud_gen, interpolation='bilinear')
    plt.imsave(dirpath+'/wordcloud.png', wordcloud_gen)
    print('Wordcloud created and saved to directory!')

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
    df = pd.concat([df[['Tweets', 'Date']], df['polarity'].apply(pd.Series)], axis=1)
    df['sentiment'] = df['compound'].apply(sentiment_checker)
    #plot the results
    time_groups = df.groupby(pd.Grouper(key='Date', freq=time_freq))
    mean_values = time_groups['compound'].mean()
    ax = mean_values.plot(kind='line')
    yabs_max = abs(max(ax.get_ylim(), key=abs))
    ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    #Set our labels and titles be specific
    plt.xlabel('Time - Daily Intervals').set_fontsize(20)
    plt.ylabel('Average Sentiment Score').set_fontsize(20)
    plt.title('Donald Trump Sentiment During 2016 Election Cycle').set_fontsize(20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(dirpath+'/final_sentiment_lineplot_trump.png')
    plt.clf()
    plt.tight_layout()
    #set font size to 20
    palette = {"positive": "#77dd77", "negative": "#ff6961"}  # pastel green and pastel red in hex
    sns.set_context("paper", font_scale=2) # 2 is roughly equivalent to a font size of 20
    sns.boxplot(y='compound', x='sentiment', data=df[df['sentiment'] != 'neutral'], palette=palette, linewidth=2.0).set(title='Donald Trump Sentiment (Compound Score)', xlabel='Compound Sentiment', ylabel='Compound Score')
    plt.savefig(dirpath+'/final_sentiment_boxplot_trump.png')
    plt.clf()
    print('Done with sentiment analysis')
    return df

def heatmap(df: pd.DataFrame, dirpath: str, top_10: set, time_freq: str):
    print('Creating heatmap...')
    
    #finds the words from the top_10 that are in each tweet
    df['top_10_only'] = df['Tweets'].apply(lambda x: set(x.split(' ')).intersection(top_10))
    #create a new dataframe
    data = df[['Date', 'top_10_only', 'compound']]
    #only include the rows which have a word in the top_10
    data = data[data['top_10_only'].apply(len) > 0]
    #explode the top_10 words into separate rows and drop entries with no top_10 words
    data = data.explode('top_10_only').dropna()
    #group the data by time and word
    time_word_groups = data.groupby([pd.Grouper(key='Date', freq=time_freq), 'top_10_only'])
    #find the mean compound score of each time and word group
    mean_values = time_word_groups['compound'].mean()
    #unstack the data into a format to be converted to a heatmap
    heatmap_data = mean_values.unstack(fill_value=0)
    # Creating the heatmap
    plt.figure(figsize=(24, 16))
    heatmap = sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".2f", center=0)
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label('Compound Score')
    plt.title('Donald Trump Compound Scores by Time and Word')
    plt.xlabel('Word')
    plt.ylabel('Date')
    plt.xticks(rotation=45)
    y_ticks = heatmap.get_yticklabels()
    for tick in y_ticks:
        tick.set_text(tick.get_text()[:-15].replace('T', ' '))
        #format dates to be just MM/DD
        tick.set_text(tick.get_text().split(' ')[0])

        tick.set_text(tick.get_text().replace('2016-', ''))
        
    heatmap.set_yticklabels(y_ticks)
    plt.tight_layout()
    plt.savefig(dirpath+'/final_heatmap_trump.png')
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
    df['Tweets'] = df['Tweets'].astype(str).str.lower()
    #convert time to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    #remove regular expressions
    regexp = RegexpTokenizer('\w+')
    df['text_token'] = df['Tweets'].apply(regexp.tokenize)
    #remove stopwords
    df['text_token'] = df['text_token'].apply(remove_stopwords)
    #remove words with length < 2
    df['text_string'] = df['text_token'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
    #remove all words that contain any mention of either x80 or x99
    df['text_string'] = df['text_string'].str.replace("x80", "")
    df['text_string'] = df['text_string'].str.replace("x99", "")
    #remove "x80 x99s" and "x80 x99t" from the text
    df['text_string'] = df['text_string'].str.replace("x80 x99s", "")
    df['text_string'] = df['text_string'].str.replace("x80 x99t", "")
    #remove urls from the text
    df['text_string'] = df['text_string'].str.replace(r'http\s+', '')
    df['text_string'] = df['text_string'].str.replace(r'https\s+', '')
    df['text_string'] = df['text_string'].str.replace(r'www\s+', '')

    #remove trump twitter handles and his name
    df['text_string'] = df['text_string'].str.replace("@realdonaldtrump", "")
    df['text_string'] = df['text_string'].str.replace("@donaldtrump", "")
    df['text_string'] = df['text_string'].str.replace("donald", "")
    df['text_string'] = df['text_string'].str.replace("trump", "")

    #remove hilary twitter handles and her name
    df['text_string'] = df['text_string'].str.replace("@hillaryclinton", "")
    df['text_string'] = df['text_string'].str.replace("@hillary", "")
    df['text_string'] = df['text_string'].str.replace("hillary", "")
    df['text_string'] = df['text_string'].str.replace("clinton", "")

    #remove amp from the text


    #remove all the domain endings like com, org, net, etc
    df['text_string'] = df['text_string'].str.replace("com", "")
    df['text_string'] = df['text_string'].str.replace(r'\.com\s+', '')
    #remove all words that look like hexidecimal numbers
    df['text_string'] = df['text_string'].str.replace(r'x[a-fA-F0-9]+', '')
    #remove all non-ascii characters
    df['text_string'] = df['text_string'].str.replace(r'[^\x00-\x7F]+', '')
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
# what is the emum for time_freq? - '30T' is 30 minutes, '1H' is 1 hour, '1D' is 1 day, '1W' is 1 week
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
    df = pd.read_csv('dataset/trump2016.csv', usecols=['Tweets', 'Date'])

    # read the trump2016.csv data and only take the columns Tweets and Date. Then append it to our "df" existing var. Only take the first 250k rows
    #df_2 = pd.read_csv('trump2016.csv', usecols=['Tweets', 'Date'], nrows=250000)

    #df = pd.concat([df,df_2])

    # format the dates to datetime format, and then filter dates within the range mintime and maxtime
    #df['Date'] = pd.to_datetime(df['Date'])
    #df = df[(df['Date'] >= mintime) & (df['Date'] <= maxtime)]

    
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

    if save_df: df.to_csv(dirpath+'/final_output.csv')

    return df

start = time.time()
analyze('results', '10/16/2016', '11/10/2016', True, '1D')
print('Total Time:', time.time()-start)