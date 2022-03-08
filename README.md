


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
      <li>Aims and Objectives </li>
      <li><a href="#features">Features</a></li>
        <li><a href="#built-with">Built With</a></li>
    </li>
    <li><a href="#getting-started">Getting Started</a></li>
     <li><a href="#technical-documention">Technical Documention</a></li>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The Stock Dashboard is a website where the user can view live data on stocks, such as their article sentiment, analyst target prices, stock predictions and more, all made in the backend Python pipeline. The article data is web scraped and then cleaned using BeutifulSoup and Pandas. Topics are extracted from the text and sentiment is analyzed using a lexicon-based approach with the VADER sentiment analyzer. Stock price predictions are made using historical prices and a Long short-term memory (LSTM) neural network. All the data is stored on MongoDB Atlas and is visualized on the website using Node.js and Chart.js
<p align="right">(<a href="#top">back to top</a>)</p>

## Aims and Objectives

### MVP

The aims for the Minimal Viable product were as follows:
- [x]  Create the layout of the Dashboard with sample data
- [x] Setup the MongoDb Database
- [x] Create the pipeline for newly added stocks
- [x] Web scrape infoamtion about the stocka such as articles and funetmantal infomation
- [x] Clean the articles - remove punction, lowercase
- [x] Analyze the sentiment of the articles using a alexicon approuch
- [x] Create the landing page / search page
- [x] Basic Neural Netwokr to predict future stock price
- [ ] Stock Screener


### Final Product
In addition to the features in the MVP, the objective for the final Product was to:
- [x]  Only analyze the setimnint of imporatnat artilces, that is artrilces taht mention the stock multimple times
- [x] Extract topics from the articles
- [x] Create the pipeline to update stock already in the database
- [x] Stock Screener
- [x] 
<p align="right">(<a href="#top">back to top</a>)</p>

## 

## Features

Landing page:

* Search for specifin stocks suing either the tikcer or the full name
* Autocomplete text wit hthe available stock

In the Dashboard the user can view data such as:

* <a href="#stock-fundemental-metrics">Stock Fundemental metrics</a>
	* <a href="#historic-price">Historic Price</a>
	*  <a href="#insider-own-%">Insider Ownership</a>
	* <a href="#relative-strength-index">Relative Strength Index</a>
	* <a href="#dividend">Dividend</a>
	* <a href="#volatility">2-week Volatility</a>
	* <a href="#price/earning-ratio">Price/Earning ratio</a>
	* <a href="#analyst-target-price">Analyst Target Price</a>
	*  Informational Tooltips
* Articles
	* <a href="#latest-articles">Latest articles about the stock, including publisher,  sentiment and a link to the article</a>
	* <a href="#historical-article-sentiment">Historical Article Sentiment and Sentiment Trends</a>
	* <a href="#latest-article-topics">Latest Article Topics</a>
* Analyst ratings
	* <a href="#latest-analyst-ratings">Latest analyst ratings</a>
	* <a href="#historical-analyst-ratings">Historical analyst ratings by month</a>
	* <a href="#analyst-ratings-in-the-last-12-months">Analyst ratings in the last 12 months </a>
*  <a href="#stock-price-predictions">Stock price predictions for the next 7 days </a>
*  <a href="#textual-sumary">A summary of all the inforamtion in a textual form </a>
* <a href="#rating-gauge">Buy/Hold/Sell rating gauge</a>
* <a href="#interactive-graphs">Interactive Graphs</a>


<div id="image" align="center">
<img src="images/screenshot2.png" alt="Project Image">
</div>

#### Stock Fundemental metrics
The Stock Fundemetal Metrics section in the top left of the Dashboard,  gives the user a quick overview of some of the imprant inforamtion abou the stock, like price, price/earnings ratio, insider ownership, dividend etc. This ifomarion is vlauble to both long-temr and short term investors

<div id="image" align="center">
<img src="images/metrics.png" alt="Metrics">
</div>

#### Historic Price

In this graph the user can view the price of the stock over the last year as well as the 20-day moving average. This graph is an imprantm etric for viewing the hisric trends in the stock as well as view mediuim term support in the term of a moving average. 

<div id="image" align="center">
<img src="images/historic-price.png" alt="Historic Price">
</div>

#### Insider Ownership

Insider Ownership is calculated as the total number of shares owned by insiders. A high value of insider ownership means that those working for the company have a large stake in the success of the company, which can be bullish for the stock.

<div id="image" align="center">
<img src="images/insider-own.png" alt="Insider Own">
</div>

#### Relative Strength Index

The relative strength index (RSI) is a momentum indicator used in technical analysis that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset. Traditional interpretation and usage of the RSI are that values of 70 or above indicate that a security is becoming overbought and reading of 30 or below indicates an oversold condition.
To better visulize the index,  I color coded the number (green for under 35% and red for over 70%) as well as used a speedometer / progress bar.

<div id="image" align="center">
<img src="images/rsi.png" alt="RSI">
</div>

#### Dividend

A dividend is the distribution of corporate profits to eligible shareholders, ususally messured in terms of a percentage realtive to stock price. It can be used a value indicator as well as an inidicator for how profitable a stock is. 
To better visulize the dividend,  I color coded the number (green for over 5% and red for 0%) as well as used a speedometer / progress bar.

<div id="image" align="center">
<img src="images/dividend.png" alt="Dividend">
</div>

#### Volatility 

Volatility is the rate at which the price of a stock increases or decreases over a particular period, in this case the last 2 weeks. Higher stock price volatility often means higher risk and helps an investor to estimate the fluctuations that may happen in the future. High volatility can be a psotive indactor for users who are looking for short-term investments.

<div id="image" align="center">
<img src="images/volatility.png" alt="Volatility">
</div>

#### Price / Earning-Ratio

Volatility is the rate at which the price of a stock increases or decreases over a particular period, in this case the last 2 weeks. Higher stock price volatility often means higher risk and helps an investor to estimate the fluctuations that may happen in the future.

<div id="image" align="center">
<img src="images/volatility.png" alt="Volatility">
</div>

####  Analyst Target Price
A price target is a price at which an analyst believes a stock to be fairly valued relative to its projected and historical earnings. The average price target is caluacted using the avergae of the latest analyst price targets for the the next 12 months.

<div id="image" align="center">
<img src="images/volatility.png" alt="Volatility">
</div>

#### Latest Articles
In the middle right section of the dashboard there are the latest articles about the stock. The user can view the publisher, title and the color coded sentiment of the text. The user can also view the whole article by clicking on it. The setniment is calculated in the backend Python pipeline.

#### Historical Article Sentiment
In the bottom right section of the dashboard the user can view daily sentiment scores for articles as well as the Moving averages of this sentiment to get a better undertanding of the public sentiment trends for the stock.

<div id="image" align="center">
<img src="images/volatility.png" alt="Volatility">
</div>

#### Latest Article Topics
In the middle left section the user can view the latest topics for articles. This gives the user a quick glimpse into what topics the public are talking about and could give ideas into what user should do more research into.

<div id="image" align="center">
<img src="images/volatility.png" alt="Volatility">
</div>

#### Latest analyst ratings
The top right section contains the latest anlyst ratings. This can can either be used as a guide to whether "Smart money" are bullish or bearish on the stock or it can be a motive for future research on why these specific fianincail instutnies have the rating. The ratings can be as follows from worst to best: Sell, Underweight, Neutral, Overweight, Buy and are color coded accordingly

#### Historical analyst ratings

The bottom middle section contains a stacked bar-chart of the analyst ratings in the last 12 months in one month segemnts. This can be useful in understanging the trend in positve/negativce ratings for the stock as well as the popularity of the stock between finficla alysts.

<div id="image" align="center">
<img src="images/volatility.png" alt="Volatility">
</div>

#### Analyst Ratings in the last 12 months

The bottom left section contains a dougnut chart with all the analyst ratings in the last 12 months. This is a good summary of what fincial analysts think of the stock. 

<div id="image" align="center">
<img src="images/volatility.png" alt="Volatility">
</div>

#### Stock Price Predictions

In the middle right section is the predictiated stock price for the next 10 days. The price is calcuated using a neural network with hsitorical price ans sentiment as the features. The graph also has a lower and upper bound which increases as the uncerntiny of the rpice increses. It would not be recommended to follow these predictions wightout considering other factors, but it could be an indictor of what the  stock could do all other facotrs beeing equal.

<div id="image" align="center">
<img src="images/volatility.png" alt="Volatility">
</div>

#### Textual Sumary

In the middle left section below the rating gauge is a textual summary of all the infomration vailable in the Dashboard. The suer can use this as a starting point when looking for potential investments. The text is color coded dpeneinf on hwether or not the iforamtion is postive or negative towards the stock. This sectio nca also be suelf for suers who are new to investing and are not sure whether or not a specif data point is postive or negative. 

#### Rating Gauge

Above the textual summary is a rating gauge. It uses the infomation from the textual sumary to generate a rating between 1 and 5 for a stock and assigne a Sell/Hold/Buy rating. This guage should not be sued as investment advice, but as a proxy for the risk of a stock. For example if the stopck seems to be low risk (low volatility, low Price/Earning ration, high dividend), the rating gauge is more likely to give it a Buy rating. 

<div id="image" align="center">
<img src="images/volatility.png" alt="Volatility">
</div>

#### Interactive Graphs

All of the graphs in the dashboard can be interacted with, for exmaple if the user only want ot see the 20 day moving avergage in the price graph, they can uncheck the Price or if the user want to see the price at a specifi day they can hilight the spot o nthe graph and see the price. 

<div id="image" align="center">
<img src="images/volatility.png" alt="Volatility">
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
* Tenserflow
* Hosted with [Heroku](https://www.heroku.com/)
* Cloud storage with [MongoDB]()

<p align="right">(<a href="#top">back to top</a>)</p>


## Technical Documention


~~~python
~~~

### Web Scraping
I extracted data such as the articles and fundematal infonmarion,  from Finviz, and used the yahoo finance API to get the historeic price and analyst ratings. For web scraping I used the Requests library to open the corresponing sotkc page on finviz as well as to open the original article page. 
~~~python
def get_all_articles(ticker):
    finviz_url = "https://finviz.com/quote.ashx?t="
    url = finviz_url + ticker
    req = Request(url=url, headers={"user-agent": "my-app"})
    response = urlopen(req)
    html = bs(response, features="html.parser")
    news_table = html.find(id="news-table")
    
    return news_table
~~~

To extract the artilce infomation such as publisher, title, origanl link and date I used the BeutifulSoup library.  After aquiring the news_table text contents, I traversed every row in the table to extract the date , after hcih I extract the link to the origanl article and call ```get_article_text``` to open the article page and extract the text.

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
        date_time = dt.datetime.strptime(date_time, "%b-%d-%y %H:%M%p ")
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

To get the full text of the artilce I use the ```get_article_text``` function. Depening on the publisher, the etext is extarced from the correct html element.  After extracing the text, I remove any unnecesarry whitespace.
~~~python
def  get_article_text(href, publisher):
 yahoo_publishers = [
        "bloomberg",
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


If successful the article title is and text are achecked in ``` is_article_important``` function. For an article to be considered important the title has to cointant either the name or the ticker or it has to be mentoioned atleast 5 times in the article. I decided on these rules because  firstly if the title coanints the name of the stock it is very likely that it is important. Secondly because after trying out differnet number of times, I came to the conclusion that 5 results in  a good balance of importance and amount of articles filtered. 

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
To make it easear to filter, analyze and save the articles I put them in a Pandas dataframe

~~~python
df = pd.DataFrame(
	parsed_data, columns=["Ticker", "Date",
                          "Publisher", "Title", "Link", "Text"])
~~~


After extracting the articles, I move on to gettign the analyst ratings. To do this I use the yahoofinace API. For the <a href="#latest-analyst-ratings">Latest analyst ratings</a> section I extract the 4 newest analyst ratings and save them in a dictionary, to later save into the database

~~~python
df_recom = tickerData.recommendations
df_recom_last_4 = df_recom.iloc[-4:]
dict_recom_last_4 = create_dict_for_anaylst_ratings(df_recom_last_4, ticker)
~~~

For the <a href="#stock-fundemental-metrics">Stock Fundemental metrics</a> section I again use finviz to extract the data. Fiirst I make a list of the metrics I owuld liek to get.


~~~python
metric = [
    "P/B", "P/E", "Target Price",
    "RSI (14)", "Volatility", "EPS (ttm)",
    "Dividend %", "ROE", "ROI", "Price",
    "Insider Own",
]

~~~
Then I use the ```get_fundamental_data()``` function, BeutifulSoup and Requests to extract the infomration from the website, simillar to the articles.

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
    df_info["ROE"] = df_info["ROE"].str.replace("%", "")
    df_info["ROI"] = df_info["ROI"].str.replace("%", "")
    df_info["Volatility"] = volatility[0][0].replace("%", "")
    df_info["Insider Own"] = df_info["Insider Own"].str.replace("%", "")
    df_info = df_info.apply(pd.to_numeric, errors="coerce")
    df_info["Ticker"] = df_info.index

~~~


### Topic Extraction

### Text Sentiment

I get the sentiment of each article using a lexicon approach with the Vader ```SentimentIntensityAnalyzer()```.  I chose the Vader sentiment analyzer because it does not requere a test and trainign set and that it already has a given set of dictionary words. The anlysis  involves calculating the sentiment from the semantic orientation of word or phrases that occur in a text.

I get the sentiment of each article using a lexicon approach with the Vader Sentiment Intensity Analyser. I analyse the sentiment of the tile as well as the text, but only for the articles that are marked as important. Each artricle is given as core from -1 to 1

I start off by getting the sentiment of the titles of the articles with the ```get_title_sentiment()``` function.
~~~python
def get_title_sentiment(articles):
    sia = SentimentIntensityAnalyzer()
    scores = articles["Title"].apply(sia.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    return articles.join(scores_df, rsuffix="_right")
~~~

I get the sentiment of the text usign a smillar aprouch. Howevr since ```SentimentIntensityAnalyzer()``` is moslyy used for short texts, to improve peformenace I split the text into senteces and use the average of all the senteces. I also delete the last two sentences of the text, becasue these often include sposered or premotional material that we shouldnt include in the sentimnet analysis.

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

And finally I generate the final socres using the mean of the negative, positive , neurtral and compund scores form the title and article text, as well as drop the rows that wont be needed anymore.

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

For the data in the <a href="#historical-article-sentiment">Historical Article Sentiment</a> section I use the ```summerize_article_sentiment_by_day()``` function. I raverese the article dataframe, and get aaticles that were published each day. After which I store the mean of these articles in a dictionary that I will later store in the databse. Also get the 3-day moving average of the dailiy article senimernt to have better idae of the trendin the sentiment. For days that dont have as dailiy sentiment I use the previus days value.

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

    df_avg_comp["3MA"] = df_avg_comp["Average Comp"].fillna(
        method="backfill").rolling(window=3, min_periods=0).mean()

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
To acquire the historic price for <a href="#historic-price">Historic Price</a> section. I use the ```pandas_datareader``` library. In the wevsite I only use the closing price, so I drop the other colllumns. I also round the price to two decina lplaces for convience and generate the 20 day-movinf avaerge price using the ```df.rolling()``` function.

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

For the <a href="#textual-sumary">Summary</a> section, depending on the values in the data, append a short summary text with the rating to a list of dictonaries.  The dictionary contains the the sumary text, the rainting from 0 to 5 and the ticker of the stock. For exapmple if the Dividend for the stock is 0%, the sumamasry section will say "The Stock does not pay a dividend" and since the rating is also zero the text is colored red. 

~~~python
  dicts = list()
    rating = 0

    if math.isnan(df["Dividend %"]): # DIVIDEND
        dicts.append(
            {"Ticker": ticker, "Text": "The Stock does not pay a Dividend", "Rating": 0}
        )
    elif df["Dividend %"].values < 1.5:
        dicts.append(
            {"Ticker": ticker, "Text": "The Stock pays a small Dividend", "Rating": 2}
        )
        rating = rating + 2
	...
    else:
        dicts.append(
            {
                "Ticker": ticker, "Text": "Insiders own a very small part of the company", "Rating": 1,
            }
        )
        rating = rating + 1
    rating_avg = rating / 6.0
    dict_rating = {"Ticker": ticker, "Rating": rating_avg}
    return dicts, dict_rating
~~~
### Price Prediction
To predict the future price of the stock I used the historic price as a feature. I sued a Longo Short Term Memory Neural Network, because it its good for predicting time searies data. To prepare the dataset and model I used the Tenserflow and gensim  libraries
First I acquired the historic price for the last 2500 days (or the maximum avalible if there isnt any data going back so far). I picked 2500 because it is enough days to makle reaonable predictions without making the model slow. Since I have to make these predictions for each stock in the database, speed is quite important. I asquite the price using Yahoo finance.
~~~python
start = dt.datetime.today() - timedelta(days=2500)
    end = dt.datetime.today()
    df_price = web.DataReader(ticker, "yahoo", start, end)
~~~
I get the latest price of the stock, whic hwill be used in the <a href="#stock-fundemental-metrics">Stock Fundemental metrics</a> section in the website. After which I get the Open values for the stock and reshape into an 1d array.
~~~python

    latest_price = df_price['Open'][-1]

    df = df_price
    df = df['Open'].values
    df = df.reshape(-1, 1)

~~~
For a neural network its important to have a trainign  and testing set, so I split them 80/20.
~~~
    dataset_train = np.array(df[:int(df.shape[0]*0.8)]) #Training and Testing sets
    dataset_test = np.array(df[int(df.shape[0]*0.8):])
~~~
For better peroftamce it is sueful to normlaize the values. For that I use the ```MinMaxScaler()```, which normalizes all the values between 0 to 1. I do this for both sets.
~~~

    scaler = MinMaxScaler(feature_range=(0, 1)) # Normalize values to scale
    dataset_train = scaler.fit_transform(dataset_train)
    dataset_test = scaler.transform(dataset_test)
~~~

~~~python
    x_train, y_train = create_prediction_dataset(dataset_train)
    x_test, y_test = create_prediction_dataset(dataset_test)
~~~

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

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(x_train, y_train, epochs=25, batch_size=32)
    model.save('stock_prediction_'+ticker+'.h5')

    model = load_model('stock_prediction_'+ticker+'.h5')

~~~

    for i in range(14):
        predictions = model.predict(x_test)
        dataset_test = np.append(
            dataset_test, predictions.item(predictions.shape[0] - 2, 0))
        dataset_test = np.reshape(dataset_test, (-1, 1))
        x_test, y = create_prediction_dataset(dataset_test)

    predictions = scaler.inverse_transform(predictions)
    predictions_list = []

    for i in range(15):
        predictions_list.append(predictions.tolist()[i - 15][0])


    return {"Ticker": ticker, "Prediction": predictions_list}
~~~
~~~python

~~~




### Database
For the databse I use the MognoDB Atlas Cloud databse. I chose Atlas becasue it offers 500MB of free storage, I have been wanitng to learn how to sue MongoDB for quite some time, and becasue the Pyhton MongoDB library is easy to learn and has good documentaiton. Before deleting or inserting data, initlize the ```MongoClient()```, this will be sued for connection to the collection
~~~python
client = MongoClient(
    "mongodb+srv://andrisvaivods11:password@cluster0.4ozrv.mongodb.net/test")
~~~

For stock that have been laready added (stocks that are part of the pipeline to update stocks), I delete the old data, before inserting the new data. All of the data is stored in the ```capstone_project``` collection.
~~~python
    delete_from_collection(ticker, "stocks_info", client)
    delete_from_collection(ticker, "stock_price", client)
    ...
    delete_from_collection(ticker, "topics", client)
    clear_collection("stocks", client)
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
In total there are 11 collections in the database:
* stocks - contains all the names of the stocks and their tickers. Used for the autocomple feature in the search page
* analyst_ratings - Contains all the analyst ratings in the last 12 montsh for a stock, devided into one month sections. Used in the <a href="#historical-analyst-ratings">Historical analyst ratings</a> sections in the website.
* analyst_total - Contains all the ratings for a stock in the last 12 monhts, in a more condensed form. Used for the analayst total ratings dougnut chart in the website
* articles - By far the biggest collection of. Contains all the articles about all the stock in the database, the title, text, link and sentiment scores. 
* avg_sentiment - conatins the avaegae sentiment for each stock by mont as well as the moving averages. Used for the <a href="#historical-article-sentiment">Historical Article Sentiment</a> section in the website.
* latest_analyst_ratings - 
* latest_articles
* price_prediction
* stock_price
* stock_rating
* text_summary
* stocks_info
* topics

#### stocks

~~~json
{
Ticker: "AMZN"
Name: "amazon"
}
~~~

#### analyst_ratings

~~~python

~~~
~~~python

~~~
### Schema

### Testing

### Literture Survey




<!-- GETTING STARTED -->
## Getting Started

To get a local copy of the website up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation



1. Clone the repo
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
 4.  Go to [http://localhost:8000/](http://localhost:8000)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage


Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.



<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Planning Record

### Jira

### Gitlab

### Gantt


<p align="right">(<a href="#top">back to top</a>)</p>


## Conclusion and  Future Imrpovements



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Andris Vaivods -  av19256@essex.ac.uk - andris.vaivods11@gmail.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#top">back to top</a>)</p>





