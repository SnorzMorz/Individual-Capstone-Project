



<div id="top" align="center">
<img src="images/screenshot.png" alt="Project Image">
</div>



<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png">
  </a>

  <h3 align="center">Stock Dashboard </h3>

  <p align="center">
    Capstone Project by Andris Vaivods
      <p align="center">
    <br />
    University Of Essex
    <br />
    <br />
    <a href="https://cseegit.essex.ac.uk/ce301_21-22/CE301_vaivods_andris_j"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="">View Demo</a>
    ·
    <a href="">Report Bug</a>
    ·
    <a href="">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <li><a href="#features">Features</a></li>
        <li><a href="#built-with">Built With</a></li>
    </li>
    <li><a href="#getting-started">Getting Started</a></li>
     <li><a href="#technical-documention">Code Walkthrough</a></li>
      <ul>
        <li><a href="#prerequisites">Web Scraping</a></li>
        <li><a href="#installation">Sentiment Analyisis</a></li>
        <li><a href="#installation">Topic Extraction</a></li>
        <li><a href="#installation">Additional Data Metrics and Feature Extraction</a></li>
        <li><a href="#installation">Future Price Prediction</a></li>
        <li><a href="#installation">Database</a></li>
        <li><a href="#installation">Database Schema</a></li>
        <li><a href="#installation">Unit Testing</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage Examples</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>




<!-- ABOUT THE PROJECT -->
## About The Project

The Stock Dashboard is a website where the user can search for specific stocks and view live data, such as their article sentiment, analyst target prices, future stock predictions, article topics and more. The website's goal is to help the user make future investing decisions. The data analysis is performed in the backend Python pipeline and stored on the MongoDB cloud database.

The article data is web scraped from various financial news publishers using Requests and Yahoo Finance API. The data is then cleaned using Beautiful Soup and Pandas. After performing more vigorous data pre-processing like lemmatization, topics are extracted from the text using Gensim and sentiment is analyzed using a lexicon-based approach with the VADER sentiment analyzer. Stock price predictions are made using historical prices and features extracted from historical prices like Moving Averages. The data is passed through a Long short-term memory (LSTM) neural network, after which predictions are made for the next ten days. The Python code is tested using unit testing.   All the data is stored on MongoDB Atlas and is updated daily.

The website consists of a search page, dashboard and stock screener. On the search page, the user can search for the desired stock; the dashboard contains all the data visualizations made using ChartJS. A NodeJS server is used in conjunction with ExpressJS to access the data. In the stock screener, the user can filter by various metrics, like price or market cap, to search for stocks, which can help make future investing decisions

<p align="right">(<a href="#top">back to top</a>)</p>


## Features

In the <a href="#landing-page">Landing Page</a> the user can:

* Search for specific stocks
* Autocomplete text with the available stocks
* Open the dashborad of a random stock

In the Dashboard, the user can view data such as:

* <a href="#stock-fundemental-metrics">Stock Fundemental metrics</a>
	* <a href="#historic-price">Historic Price</a>
	*  <a href="#insider-own-%">Insider Ownership</a>
	* <a href="#relative-strength-index">Relative Strength Index</a>
	* <a href="#dividend">Dividend</a>
	* <a href="#volatility">2-week Volatility</a>
	* <a href="#price/earning-ratio">Price/Earning ratio</a>
	* <a href="#analyst-target-price">Analyst Target Price</a>
	* <a href="#informational-tooltips">Informational Tooltips</a>
* Articles
	* <a href="#latest-articles">Latest articles</a>
	* <a href="#historical-article-sentiment">Historical Article Sentiment and Sentiment Trends</a>
	* <a href="#latest-article-topics">Latest Article Topics</a>
* Analyst ratings
	* <a href="#latest-analyst-ratings">Latest analyst ratings</a>
	* <a href="#historical-analyst-ratings">Historical analyst ratings by month</a>
	* <a href="#analyst-ratings-in-the-last-12-months">Analyst ratings in the last 12 months </a>
*  <a href="#stock-price-predictions">Stock price predictions for the next two weeks days </a>
*  <a href="#textual-sumary">A summary of all the information in a textual form </a>
* <a href="#rating-gauge">Buy/Hold/Sell rating gauge</a>
* <a href="#interactive-graphs">Interactive Graphs</a>

In the <a href="#stock-screener">Stock Screener</a> the user can Filter by 12 different metrics.


<div id="image" align="center">
<img src="images/screenshot2.png" alt="Project Image">
</div>



#### Landing Page
On the Landing Page, users can search for specif stocks, by their tickers, as well as open a random stock dashboard by clicking the "I'm Feeling Lucky" button. If the user searches for a stock that is not available an error message will appear. 

<div id="image" align="center">
<img src="images/search.png" alt="Search">
</div>


#### Stock Fundemental metrics
The Stock Fundamental Metrics section in the top left of the Dashboard,  gives the user a quick overview of some of the important information about the stock, like price, price/earnings ratio, insider ownership, dividend etc. This information is valuable to both long-term and short term investors. The information is colour-coded to better portray what the specific value means.

<div id="image" align="center">
<img src="images/metrics.png" alt="Metrics">
</div>

#### Historic Price

In this graph, the user can view the price of the stock over the last year as well as the 20-day and 50-day moving averages. This graph is an important metric for viewing the historic trends in the stock as well as viewing medium-term support in the term of moving averages. 

<div id="image" align="center">
<img src="images/historic-price.png" alt="Historic Price">
</div>

#### Insider Ownership

Insider Ownership is calculated as the total number of shares owned by insiders. A high value of insider ownership means that those working for the company have a large stake in the success of the company, which can be bullish for the stock.

<div id="image" align="center">
<img src="images/insider-own.png" alt="Insider Own">
</div>

#### Relative Strength Index

The relative strength index (RSI) is a momentum indicator used in technical analysis that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the stock. Traditional interpretation and usage of the RSI are that values of 70 or above indicate that a stock is becoming overbought and a reading of 30 or below indicates an oversold condition.
To better visualize the index,  I colour coded the number (green for under 35% and red for over 70%) as well as used a speedometer/progress bar.

<div id="image" align="center">
<img src="images/rsi.png" alt="RSI">
</div>

#### Dividend

A dividend is the distribution of corporate profits to eligible shareholders, usually measured in terms of a percentage relative to stock price. It can be used as a value indicator as well as an indicator of how profitable a stock is. 
To better visualize the dividend,  I colour coded the number (green for over 5% and red for 0%) as well as used a speedometer/progress bar.

<div id="image" align="center">
<img src="images/dividend.png" alt="Dividend">
</div>

#### Volatility 

Volatility is the rate at which the price of a stock increases or decreases over a particular period, in this case, the last 2 weeks. Higher stock price volatility often means higher risk and helps an investor estimate the fluctuations that may happen in the future. High volatility can be a positive indicator for users who are looking for short-term investments.

<div id="image" align="center">
<img src="images/volatility.png" alt="Volatility">
</div>

#### Price / Earning-Ratio

Volatility is the rate at which the price of a stock increases or decreases over a particular period, in this case, the last 2 weeks. Higher stock price volatility often means higher risk and helps an investor estimate the fluctuations that may happen in the future.

<div id="image" align="center">
<img src="images/pe_ratio.png" alt="Price / Earning-Ratio">
</div>

####  Analyst Target Price
A price target is a price at which an analyst believes a stock to be fairly valued relative to its projected and historical earnings. The average price target is calculated using the average of the latest analyst price targets for the next 12 months.

<div id="image" align="center">
<img src="images/analyst_target.png" alt=" Analyst Target Price">
</div>


#### Informational Tooltips

<div id="image" align="center">
<img src="images/volatility.png" alt="Volatility">
</div>

#### Latest Articles
In the middle right section of the dashboard, there are the latest articles about the stock. The user can view the publisher, title and the sentiment of the text. The user can also view the whole article by clicking on it. 

<div id="image" align="center">
<img src="images/latest_articles.png" alt="Latest Articles">
</div>

#### Historical Article Sentiment
In the bottom right section of the dashboard, the user can view daily sentiment scores for articles as well as the Moving averages of this sentiment to get a better understanding of the public sentiment trends for the stock.

<div id="image" align="center">
<img src="images/hist_article_sent.png" alt="Historical Article Sentiment">
</div>

#### Latest Article Topics
In the middle left section, the user can view the latest topics for articles. This gives the user a quick glimpse into what topics the public is talking about and could give ideas into what the user should do more research into.

<div id="image" align="center">
<img src="images/volatility.png" alt="Volatility">
</div>

#### Latest analyst ratings
The top right section contains the latest analyst ratings. This can either be used as a guide to whether "Smart money" is bullish or bearish on the stock or it can be a motive for future research on why these specific financial institutes have the rating. The ratings can be as follows from worst to best: Sell, Underweight, Neutral, Overweight, Buy and are colour coded accordingly

<div id="image" align="center">
<img src="images/latest_analyst.png" alt="Latest analyst ratings">
</div>

#### Historical analyst ratings

The bottom middle section contains a stacked bar chart of the analyst ratings in the last 12 months in month segments. This can be useful in understanding the trend in positive/negative ratings for the stock as well as the popularity of the stock among finficla analysts.

<div id="image" align="center">
<img src="images/hist_analyst_ratings.png" alt="Historical analyst ratings">
</div>

#### Analyst Ratings in the last 12 months

The bottom left section contains a doughnut chart with all the analyst ratings in the last 12 months. This is a good summary of what financial analysts think of the stock. 

<div id="image" align="center">
<img src="images/analyst_ratings_12.png" alt="Analyst Ratings in the last 12 months">
</div>

#### Stock Price Predictions

In the middle right section is the predicted stock price for the two weeks. The price is calculated using a neural network with historical prices and moving averages. The graph also has a lower and upper bound which increases as the uncertainty of the price increases. It would not be recommended to follow these predictions precisely but they could be sued as a momentum indicator. 

<div id="image" align="center">
<img src="images/prediction.png" alt="Prediction">
</div>

#### Textual Sumary

In the middle left section below the rating gauge is a textual summary of all the information available in the Dashboard. The user can use this as a starting point when looking for potential investments. The text is colour coded depending on whether or not the information is positive or negative about the stock. This section can also be self for users who are new to investing and are not sure whether or not a specif data point is positive or negative. 

<div id="image" align="center">
<img src="images/textual_summary.png" alt="Textual Sumary">
</div>

#### Rating Gauge

Above the textual summary is a rating gauge. It uses the information from the textual summary to generate a rating between 1 and 5 for stock and assigned a Sell/Hold/Buy rating. This gauge should not be used as investment advice, but as a proxy for the risk of a stock. For example, if the stock seems to be low risk (low volatility, low Price/Earning ratio, high dividend), the rating gauge is more likely to give it a Buy rating. 

<div id="image" align="center">
<img src="images/rating_gauge.png" alt="Rating Gauge">
</div>

#### Interactive Graphs

All of the graphs in the dashboard can be interacted with, for example, if the user only wants to see the 20-day moving average in the price graph, they can uncheck the Price or if the user wants to see the price at a specific day they can highlight the spot on the graph and see the price. 

<div id="image" align="center">
<img src="images/interactive_gif.gif" alt="Interactive Graphs">
</div>

#### Stock Screener
In the Stock, Screener users can filter by twelve different metrics to find stocks. Users can pick if they want the value to be Below, Above or Exactly the specified value.

<div id="image" align="center">
<img src="images/screener.png" alt="Screener">
</div>

After clicking submit the result of the filtering will appear. Here the user can open the Dashboard of the stock by clicking on the ticker and sort by specif values by cling on the table header. 

<div id="image" align="center">
<img src="images/volatility.png" alt="Volatility">
</div>

The user can also pick from 3 different preset filters - Value, Growth and Oversold. These can be helful for new investors looking for a specifc investment.
<div id="image" align="center">
<img src="images/presets.png" alt="Presets">
</div>

### Built With


* [Node.js](https://nodejs.org/en/)
* [Chart.js](https://www.chartjs.org/)
* [Express.js](https://expressjs.com/)
* [NLTK](https://www.nltk.org/)
* [Beautiful soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* [Gensim](https://radimrehurek.com/gensim/)
* Finviz.com
* Yahoo Finanace API
* Tensorflow
* Hosted with [Heroku](https://www.heroku.com/)
* Cloud storage with [MongoDB]()

<p align="right">(<a href="#top">back to top</a>)</p>


## Code Walkthrough 
In this section, I discuss the code and walk through the main functions.

~~~python
~~~

### Web Scraping
The first step for each stock is extracting the relevant information from the web such as the articles and fundamental information. Data such as the historic price and analyst ratings are gathered using the Yahoo finance API. For web scraping, I used the Requests library to open the corresponding stock page on FINVIZ as well as to open the original article page. 
~~~python
def get_all_articles(ticker):
    finviz_url = "https://finviz.com/quote.ashx?t="
    url = finviz_url + ticker # Create URL
    req = Request(url=url, headers={"user-agent": "my-app"}) # Send Request
    response = urlopen(req) # Recieve back
    html = bs(response, features="html.parser") # Parse HTML
    news_table = html.find(id="news-table") # Extract Table element with articles
    
    return news_table
~~~

To extract the article information such as publisher, title, original link and date I used the BeutifulSoup library.  After acquiring the``` news_table``` text contents, I traversed every row in the table to extract the date, after which I extract the link to the original article and call ```get_article_text``` to open the article page and extract the text.

~~~python
def parse_articles(ticker, name, tr):
    parsed_data = []
    for row in tr:
        title = row.a.text  # Get title
        date_data = row.td.text.split(" ")  # Get date
        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]
        date_time = date + " " + time
        date_time = dt.datetime.strptime(date_time, "%b-%d-%y %H:%M%p ") # Generate datetime objects
~~~

Only extract articles that were puvblished before today. Articles published today will be added the following day. 
~~~python
        if(date_time.date() < dt.datetime.today().date()):
            article_link = row.find("a", {"class": "tab-link-news"}).attrs[
                "href"]
            publisher = row.span.text.strip().lower()
            article_text = get_article_text(article_link, publisher)
            if(date_time.date() == dt.datetime.today().date()):
                break
            if article_text != "" and is_article_important(ticker.lower(), name, article_text, title):
                parsed_data.append(
                    [ticker, date_time, publisher, title,
                        article_link, article_text]
                )
    return parsed_data
~~~

To get the full text of the article I use the ```get_article_text``` function. Depending on the publisher, the text is extracted from the correct HTML element.  After extracting the text, I remove any unnecessary whitespace.
~~~python
def  get_article_text(href, publisher):
 yahoo_publishers = [
        "Bloomberg",
        ...
        "investorplace"
    ]
    try:
        req = Request(url=href, headers={"user-agent": "my-app"})
        response = urlopen(req)
        html = bs(response, features="html.parser")  # Get HTML of article
        if publisher == "motley fool": # Depending on publisher, extract article text
            article_text = html.find(class_="tailwind-article-body").text
        elif publisher in yahoo_publishers:
            article_text = html.find(class_="caas-body").text
            ...
        else:
            article_text = ""
    except Exception as e:
        article_text = ""
        
    article_text = " ".join(article_text.split()) # Remove unnecessary whitespace
    return article_text
~~~


If successful the article title and text are checked in ``` is_article_important``` function. For an article to be considered important the title has to contain either the name or the ticker or it has to be mentioned at least 5 times in the article. I decided on these rules because firstly if the title comments the name of the stock it is very likely that it is important. Secondly because after trying out a different number of times, I concluded that 5 results in a good balance of importance and amount of articles filtered. 

~~~python
def is_article_important(ticker, name, article, title):
    article_clean = re.sub(r'\W+', ' ', article)
    title_clean = re.sub(r'\W+', ' ', title)
    if(count_occurrences(ticker, title_clean) >= 1 or count_occurrences(name, title_clean) >= 1):
        return True
    if(count_occurrences(ticker, article_clean) + count_occurrences(name, article_clean) >= 5):
        return True

    return False
~~~

To make it easier to filter, analyze and save the articles I put them in a Pandas data frame

~~~python
df = pd.DataFrame(
	parsed_data, columns=["Ticker", "Date",
                          "Publisher", "Title", "Link", "Text"])
~~~


After extracting the articles, I move on to getting the analyst ratings. To do this I use the yahoo finance API. For the <a href="#latest-analyst-ratings">Latest analyst ratings</a> section I extract the 4 newest analyst ratings and save them in a dictionary, to later save into the database

~~~python
df_recom = tickerData.recommendations
df_recom_last_4 = df_recom.iloc[-4:]
dict_recom_last_4 = create_dict_for_anaylst_ratings(df_recom_last_4, ticker)
~~~

For the <a href="#stock-fundemental-metrics">Stock Fundamental metrics</a> section I again use finviz to extract the data. First I make a list of the metrics I would like to get.


~~~python
metric = [
    "P/B", "P/E", "Target Price",
    "RSI (14)", "Volatility", "EPS (ttm)",
    "Dividend %", "ROE", "ROI", "Price",
    "Insider Own",
]

~~~
Then I use the ```get_fundamental_data()``` function, BeutifulSoup and Requests to extract the information from the website, similar to the articles. This extracts all of the metrics later used in the Dashboard and Stock Screener. 

~~~python
def get_fundamental_data(df):
    for symbol in df.index:
        url = "http://finviz.com/quote.ashx?t=" + symbol.lower()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0"
        }
        soup = bs(requests.get(url, headers=headers).content, features="lxml")
        for m in df.columns:
            df.loc[symbol, m] = fundamental_metric(soup, m)
    return df
~~~
When the data about the stock has been scraped I clean it, by removing the percentages and make the values into numerics using Pandas ```to_numeric```.
~~~python
    df_info["Dividend %"] = df_info["Dividend %"].str.replace("%", "")
    ...
    df_info["Ticker"] = df_info.index

~~~


### Topic Extraction
Topics from the text are extracted using an LDA model that's available through the Gensim library.
First sets of stopwords, punctions and extra words like the name of the company and other words that would not add any value to the topics are finalized.  Also, a word Lemmatizer ius inilized that will change thw ords to their base forms. 
~~~python
def generate_topics(df, name, ticker):
    stop = set(stopwords.words("english"))  
    exclude = set(string.punctuation)  
    lemma = WordNetLemmatizer() 
    extra_words = set(["inc",  name, ... "share"])
~~~
Since topics will be genearted oin articles if the last 2 weeks, the start and end dates are initilized,
~~~python
    start_date = dt.datetime.now() - dt.timedelta(days=14)
    end_date = dt.datetime.now()
~~~
Now the article dataframe is filtered to find articles in the last 2 weeks using the two dates. 
~~~python
    mask = (df["Date"] > start_date) & (df["Date"] <= end_date)
    period = df.loc[mask]
~~~

After the artciles are filtered, the text ofd the artciles is saved in a list and each article is cleaned usign the ``` clean_topic_text``` function.
~~~python
    doc_complete = list(period["Text"])
    doc_clean = [
        clean_topic_text(doc, lemma, exclude, stop, extra_words).split()
        for doc in doc_complete
    ]
~~~
In the clean topic text funtion stopwords and puncation are removed. Then the rmeaining words are lemmitized and finally the extra words are removed. 
~~~python

def clean_topic_text(doc, lemma, exclude, stop, extra_words):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    final = " ".join([i for i in normalized.lower().split() if i not in extra_words])
    return final
~~~

The articles are then put into a term dictionary where each term is assigned an index.
~~~python
    dictionary = corpora.Dictionary(doc_clean)
~~~
Now the list of filtered docucents are put into a document-term matrix. A document-term matrix describes the frequency of terms that occur in all ther articles.
~~~python
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
~~~

Now the LDA model is inilized using the Gensim library. The document-term matrix togetger wiht the term dictionary are used to train the model. It genreates 8 topics and passes trhpugh the articles 18 times. 
~~~python
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=8, id2word=dictionary, passes=18)
~~~

After the model is trained the generated topics are extracted and the amount of articles for each topic is counted.
~~~python
    topic_list = []
    for i in range(len(doc_term_matrix)):
        topic_list.append(ldamodel[doc_term_matrix[i]][0][0])
    topic_count = Counter(topic_list)
~~~

The topics togthetr with the percetage of all articles that are about the topic are saved to a dictionar. The topics are used in the <a href="#latest-article-topics">Latest Article Topics</a> section.
~~~python

    dict_topics = []
    i = 0
    for item in ldamodel.print_topics(num_topics=8, num_words=2):
        dict_topics.append(
            {
                "Ticker": ticker,
                "Topics": re.findall('"([^"]*)"', item[1]),
                "Percentage": (topic_count[i] / len(topic_list)) * 100,
            }
        )
        i += 1

    return dict_topics

~~~


### Text Sentiment

I get the sentiment of each article using a lexicon approach with the Vader ``` SentimentIntensityAnalyzer()```.  I chose the Vader sentiment analyzer because it does not require a test and training set and it already has a given set of dictionary words. The analysis involves calculating the sentiment from the semantic orientation of words or phrases that occur in each article.
I analyse the sentiment of the tile as well as the text, but only for the articles that are marked as important. Each article is given as core from -1 to 1

I start off by getting the sentiment of the titles of the articles with the ```get_title_sentiment()``` function.
~~~python
def get_title_sentiment(articles):
    sia = SentimentIntensityAnalyzer()
    scores = articles["Title"].apply(sia.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    return articles.join(scores_df, rsuffix="_right")
~~~

I get the sentiment of the text using a similar approach. However since ``` SentimentIntensityAnalyzer()``` is mostly used for short texts, to improve performance I split the text into sentences and use the average of all the sentences. I also delete the last two sentences of the text, because these often include sponsored or promotional material that we shouldn't include in the sentiment analysis.

~~~python
def get_text_semtiment(df):
    texts = df.Text
    sent_mean = []
    for index, text in texts.items():
        sent = []
        sia = SentimentIntensityAnalyzer()
        text_token = tokenize.sent_tokenize(text)
        del text_token[-2:]
        for sentence in text_token:
            sent.append(sia.polarity_scores(sentence))
        df = pd.DataFrame(sent)
        sent_mean.append(df.mean())
    return sent_mean
~~~

And finally, I generate the final score using the mean of the negative, positive, neutral and compound scores from the title and article text, as well as drop the rows that won't be needed anymore.

~~~python
    df["average_neg"] = round(df[["neg", "neg_text"]].mean(axis=1), 2)
    df["average_pos"] = round(df[["pos", "pos_text"]].mean(axis=1), 2)
    df["average_neu"] = round(df[["neu", "neu_text"]].mean(axis=1), 2)
    df["average_comp"] = round(
        df[["compound", "compound_text"]].mean(axis=1), 2)

    df = df.drop(
        columns=[
            "neu",
            ...
            "compound_text",
        ]
    )
~~~

For the data in the <a href="#historical-article-sentiment">Historical Article Sentiment</a> section I use the ```summerize_article_sentiment_by_day()``` function. I traverse the article data frame and get articles that were published each day. After which I store the mean of these articles in a dictionary that I will later store in the database.

~~~python
def summerize_article_sentiment_by_day(df, ticker):
    start_date = min(df['Date'])
    end_date = max(df['Date'])
    df_avg_comp = pd.DataFrame()
    current_date = start_date
    while current_date <= end_date:

        mask = (pd.to_datetime(df["Date"]) >= current_date) & (
            pd.to_datetime(df["Date"]) <= (current_date + dt.timedelta(days=1)))
        period = df.loc[mask]

        dict = {
            "Average Comp": round(period["average_comp"].mean(), 4),
            "Day": current_date.day,
            "Month": current_date.month,
            "Ticker": ticker,
        }
        df_avg_comp = df_avg_comp.append(dict, ignore_index=True)
        current_date = current_date + dt.timedelta(days=1)
~~~
 Also, get the 3-day  and 7-day moving average of the daily article sentiment to have a better idea of the trend in the sentiment. For days that don't have daily sentiment, I use the previous day's value. The moving averages are used in the <a href="#historical-article-sentiment">Historical Article Sentiment</a> section as well as the 
~~~python
       df_avg_comp["3MA"] = (
        df_avg_comp["Average Comp"]
        .fillna(method="backfill")
        .rolling(window=3, min_periods=0)
        .mean())
    df_avg_comp["7MA"] = (
        df_avg_comp["Average Comp"]
        .fillna(method="backfill")
        .rolling(window=7, min_periods=0)
        .mean())

    return df_avg_comp
~~~

Finally, for the <a href="#latest-articles">Latest articles</a> section  I use the ```get_latest_articles()``` function, whcih gets the last 4 articles from the dataframe.

~~~python
def get_latest_articles(df):
    df = df.sort_values(by="Date")
    df_latest = df.tail(4)
    return df_latest
~~~

### Additional Data metrics
To acquire the historic price for <a href="#historic-price">Historic Price</a> section. I use the ```pandas_datareader``` library. On the website, I only use the closing price, so I drop the other columns. I also round the price to two decimal places for convenience and generate the 20 day moving average prices using the ```df. rolling()``` function.

~~~python
def get_stock_price(ticker, days):
    start = dt.datetime.today() - timedelta(days=days)
    end = dt.datetime.today()
    df_price = web.DataReader(ticker, "yahoo", start, end)
    df_price.drop(["Close", "High", "Low", "Open"], axis=1, inplace=True)
    df_price["Ticker"] = ticker
    df_price = df_price.round(2)
    df_price["Date"] = df_price.index
    df_price.index = range(1, len(df_price) + 1)
    return df_price
~~~

For the <a href="#textual-sumary">Summary</a> section, depending on the values in the data, append a short summary text with the rating to a list of dictionaries.  The dictionary contains the summary text, the rating from 0 to 5 and the ticker of the stock. For example, if the Dividend for the stock is 0%, the summary section will say "The Stock does not pay a dividend" and since the rating is also zero the text is coloured red. 

~~~python
  dicts = list()
    rating = 0
    if math.isnan(df["Dividend %"]): # DIVIDEND
        dicts.append({"Ticker": ticker, "Text": "The Stock does not pay a Dividend", "Rating": 0})
    elif df["Dividend %"].values < 1.5:
        dicts.append({"Ticker": ticker, "Text": "The Stock pays a small Dividend", "Rating": 2})
        rating = rating + 2
	...
    else:
        dicts.append(
            {
                "Ticker": ticker, "Text": "Insiders own a very small part of the company", "Rating": 1,
            })
        rating = rating + 1
    rating_avg = rating / 6.0
    dict_rating = {"Ticker": ticker, "Rating": rating_avg}
    return dicts, dict_rating
~~~
The extracted anlayst ratings from Yahoo Finacne are saved to dictioanreis. These are used in the <a href="#latest-analyst-ratings">Latest analyst ratings</a> section.

~~~python
def create_dict_for_anaylst_ratings(df, ticker):
    firms = df["Firm"].to_list()
    grades = df["To Grade"].to_list()
    dates = df.index.to_list()
    dict = {"Firms": firms, "Grades": grades, "Dates": dates, "Ticker": ticker}
    return dict
~~~


### Price Prediction
To predict the future price of the stock I used the historic price as well as Moving Averages as a feature. I used a Longo Short Term Memory (LSTM) Neural Network because it is good for predicting time series data. To prepare the dataset and model I used the TensorFlow and gensim  libraries.
First I acquired the historic price for the last 2500 days (or the maximum available if there isn't any data going back so far). I picked 2500 because it is enough days to make reasonable predictions without making the model slow. Since I have to make these predictions for each stock in the database, speed is quite important. I gather the price using Yahoo finance.
~~~python
start = dt.datetime.today() - timedelta(days=2500)
    end = dt.datetime.today()
    df_price = web.DataReader(ticker, "yahoo", start, end)
~~~
I get the latest price of the stock, which will be used in the <a href="#stock-fundemental-metrics">Stock Fundamental metrics</a> section in the website. After which I get the Open values for the stock and reshape them into a 1d array.
~~~python

    latest_price = df_price['Open'][-1]

    df = df_price
    df = df['Open'].values
    df = df.reshape(-1, 1)

~~~
For a neural network, it's important to have a training and testing set, so I split them 80/20.
~~~
    dataset_train = np.array(df[:int(df.shape[0]*0.8)]) #Training and Testing sets
    dataset_test = np.array(df[int(df.shape[0]*0.8):])
~~~
For better peroftamce, it is useful to normalize the values. For that, I use the ``` MinMaxScaler()```, which normalizes all the values between 0 to 1. I do this for both sets.
~~~

    scaler = MinMaxScaler(feature_range=(0, 1)) # Normalize values to scale
    dataset_train = scaler.fit_transform(dataset_train)
    dataset_test = scaler.transform(dataset_test)
~~~
Since I am using the last 50 days to predict the next daym, the datasets must be split to accomodate this.
~~~python
    x_train, y_train = create_prediction_dataset(dataset_train)
    x_test, y_test = create_prediction_dataset(dataset_test)
~~~

Now its time to setup the LSTM model. The model consists of 4 LSTM layers, 4 Dropout layers and 1 Dense layer. 
~~~python
    model = Sequential()
    model.add(LSTM(units=96, return_sequences=True,
              input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
~~~


Now its time to compile the model. The loss fuction used is mean squered error and the opimzer is the Adam optimzer. Adam is a replacement optimization algorithm for  gradient descent for training deep learning models. AFter this the model is fitted with the training data. 25 epochs means that the data is passed trhpoug the model 25 times. 
~~~python
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=25, batch_size=32)
~~~

Now that the data has been fitted the model can be saved for later use. It will be sued tomorrow to predict the stock prices again. 
~~~python
    model.save('stock_prediction_'+ticker+'.h5')
    model = load_model('stock_prediction_'+ticker+'.h5')
~~~

The price for the next two weeks is predicted. In each predcition the newly predicrted price is added to the end of the dataset and reshaped again so the next day can be predicted. 
~~~python
    for i in range(10):
        predictions = model.predict(x_test)
        dataset_test = np.append(dataset_test, predictions.item(predictions.shape[0] - 1, 0))
        dataset_test = np.reshape(dataset_test, (-1, 1))
        x_test, y = create_prediction_dataset(dataset_test)
~~~

After the predictions are made the data is scaled back to its orginal form and then stored in a dictionray that includess the stock tikcer, prections and dates for the predictions. 
~~~python
    predictions = scaler.inverse_transform(predictions)
    predictions_list = predictions.tolist()[-10:]
        return {
        "Ticker": ticker,
        "Prediction": predicted_prices,
        "Dates": pd.date_range(
            dt.datetime.today(), dt.datetime.today() - timedelta(days=11), freq="d"
        ).tolist(),
   
    }
    
~~~


~~~python

~~~




### Database
For the database, I use the MongoDB Atlas cloud database. I chose Atlas because it offers 500MB of free storage, I have been wanting to learn how to use MongoDB for quite some time, and because the Python MongoDB library is easy to learn and has good documentation. Before deleting or inserting data, initialise the ``` MongoClient()```, this will be sued for connection to the collection
~~~python
client = MongoClient(
    "mongodb+srv://andrisvaivods11:password@cluster0.4ozrv.mongodb.net/test")
~~~

For stocks that have been already added (stocks that are part of the pipeline to update stocks), I delete the old data, before inserting the new data. All of the data is stored in the ```capstone_project``` collection.
~~~python
    delete_from_collection(ticker, "stocks_info", client)
    ...
    delete_from_collection(ticker, "topics", client)
~~~
For new stocks and old stocks I insert the new data, depending on the type of data, either a dictionary or a dataframe, I call ```insert_dict_into_db``` or ```insert_df_into_db``` 
~~~python
    for df_recom in df_recoms:
        insert_dict_into_db(df_recom, "analyst_ratings", client)
    for dict in dicts_info_text:
        insert_dict_into_db(dict, "text_summary", client)
    for dict in dict_topics:
        insert_dict_into_db(dict, "topics", client)
    insert_dict_into_db(dict_rating, "stock_rating", client)
    ...
    insert_df_into_db(df_info, "stocks_info", client)
    if(not df.empty):
        insert_df_into_db(df, "articles", client)
~~~
In total there are 13 collections in the database:
* **stocks** - contains all the names of the stocks and their tickers. Used for the autocomplete feature on the search page
* **analyst_ratings** - Contains all the analyst ratings in the last 12 montsh for a stock, devided into one month sections. Used in the <a href="#historical-analyst-ratings">Historical analyst ratings</a> sections in the website.
* **analyst_total** - Contains all the ratings for a stock in the last 12 months, in a more condensed form. Used for the analyst total ratings doughnut chart on the website
* **articles** - By far the biggest collection. Contains all the articles about all the stock in the database, the title, text, link and sentiment scores. 
* **avg_sentiment** - Conatins the avaegae sentiment for each stock by mont as well as the moving averages. Used for the <a href="#historical-article-sentiment">Historical Article Sentiment</a> section in the website.
* **latest_analyst_ratings** - Contains the latest analyst ratings that are used in the <a href="#latest-analyst-ratings">Latest analyst ratings</a> section. 
* **latest_articles** - Contains the latest artcilesm this includes the title, publisher,  lin kto the original article and sentiment score. Used in the <a href="#latest-articles">Latest articles</a> section.
* **price_prediction** - Contains the predcited future prices. Used in the <a href="#stock-price-predictions">Stock price prediction</a> section.
* **stock_price** - Contains the historic stock price and moving averages. Used in the <a href="#historic-price">Historic Price</a> section.
* **stock_rating** - Contains a rating from 1 to 5 for each stock. Used for the <a href="#rating-gauge">Buy/Hold/Sell rating gauge</a>
* **text_summary** - Contains  a textual summary of the fundemental metrics of the stock. Used in the <a href="#textual-sumary">summary</a> section. 
* **stocks_info** - Conatains all the fundemental infomation for each stock. Used in the <a href="#stock-screener">Stock Screener</a> for filtering and the <a href="#stock-fundemental-metrics">Stock Fundemental metrics</a> section. 
* **topics** - Conatins the genreted topics from articles. Used in the <a href="#latest-article-topics">Article Topics</a> section of the dashboard.


### Schema

<div id="image" align="center">
<img src="images/erd.png" alt="ERD">
</div>

### Testing
The backend code contains many unit tests. Here are a few examples.


~~~python

~~~

~~~python

~~~

~~~python

~~~

~~~python

~~~

~~~python

~~~




<!-- GETTING STARTED -->
## Getting Started

To get a local copy of the website up and running follow these simple example steps.

### Prerequisites

* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation



1. Clone the repo or download it from Gitlab
   ```sh
   git clone https://cseegit.essex.ac.uk/ce301_21-22/CE301_vaivods_andris_j.git
   ```
2. Install NPM packages
   ```sh
   npm install
   ```
3. Start the node.js server
   ```js
   npm start
   ```
 4.  Go to [http://localhost:5000/](http://localhost:5000)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage


Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.



<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->



<p align="right">(<a href="#top">back to top</a>)</p>







<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Andris Vaivods -  av19256@essex.ac.uk - andris.vaivods11@gmail.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#top">back to top</a>)</p>




