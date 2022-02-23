from urllib.request import urlopen, Request
import re
import string
from nltk import tokenize
from bs4 import BeautifulSoup as bs
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from pymongo import MongoClient
import requests
from nltk.tokenize import word_tokenize
import csv
from gensim import matutils, models
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.feature_extraction import text
import datetime as dt
from datetime import datetime, timedelta
import certifi
import pandas_datareader.data as web
import yfinance as yf
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
import math
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim import corpora

pd.options.mode.chained_assignment = None

def insert_df_into_db(df, collection_name):
    client = MongoClient(
        "mongodb+srv://andrisvaivods11:andyPandy2000!@cluster0.4ozrv.mongodb.net/test"
    )

    db = client["capstone_project"]
    collection = db[collection_name]

    data_dict = df.to_dict("records")

    collection.insert_many(data_dict)

def insert_dict_into_db(dict, collection_name):
    client = MongoClient(
        "mongodb+srv://andrisvaivods11:andyPandy2000!@cluster0.4ozrv.mongodb.net/test"
    )

    db = client["capstone_project"]
    collection = db[collection_name]

    collection.insert_one(dict)

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

def get_stock_price_from_db(ticker):
    client = MongoClient(
        "mongodb+srv://andrisvaivods11:andyPandy2000!@cluster0.4ozrv.mongodb.net/test"
    )

    db = client["capstone_project"]
    mycol = db["stocks"]

    cursor = mycol.find(
        {},
        {"Ticker": ticker, "_id": 1, "Date": 1, "Volume": 1, "Adj Close": 1, "20ma": 1},
    )
    df = pd.DataFrame(list(cursor))

    return df

def get_articles_from_db(ticker):
    client = MongoClient(
        "mongodb+srv://andrisvaivods11:andyPandy2000!@cluster0.4ozrv.mongodb.net/test",
        tlsCAFile=certifi.where(),
    )

    mydb = client["capstone_project"]
    mycol = mydb["articles"]

    cursor = mycol.find({"Ticker": ticker})
    df = pd.DataFrame(list(cursor))

    return df

def count_occurrences(word, text):
    return text.lower().split().count(word)

def is_article_important(ticker, name, article, title):

    article_clean = re.sub(r'\W+', ' ', article)
    title_clean = re.sub(r'\W+', ' ', title)

    #print("Title:" + title_clean)
    #print("Article:" + article_clean)
    #print("Ticker:" + ticker)
    #print("Name:" + name)

    #title_num = count_occurrences(ticker, title_clean) + count_occurrences(name, title_clean)
    #print("Occurences in title: " + str(title_num))


    #title_num = count_occurrences(ticker, article_clean) + count_occurrences(name, article_clean)
    #print("Occurences in Text: " + str(title_num))



    if(count_occurrences(ticker, title_clean) >=1 or count_occurrences(name, title_clean) >= 1):
        return True
    if(count_occurrences(ticker, article_clean) + count_occurrences(name, article_clean) >=5):
        return True

    return False

def get_all_articles(ticker):
    finviz_url = "https://finviz.com/quote.ashx?t="
    url = finviz_url + ticker

    req = Request(url=url, headers={"user-agent": "my-app"})
    response = urlopen(req)

    html = bs(response, features="html.parser")
    news_table = html.find(id="news-table")

    return news_table

def parse_articles(ticker, name, tr):
    parsed_data = []

    i = 0

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
                "href"
            ]  # Link for article
            # Get publisher, strip whitespace, lowercase
            publisher = row.span.text.strip().lower()

            article_text = get_article_text(article_link, publisher)

            if(date_time.date() == dt.datetime.today().date()): 
                break


            if article_text != "" and is_article_important(ticker.lower(), name, article_text, title):
                parsed_data.append(
                    [ticker, date_time, publisher, title, article_link, article_text]
                )

            #i=i+1

            #if(i == 10):
                #break

    return parsed_data

def get_article_text(href, publisher):
    yahoo_publishers = [
        "bloomberg",
        "yahoo finance video",
        "simply wall st",
        "cnw group",
        "yahoo finance",
        "black enterprise",
        "reuters",
        "insider monkey",
        "zacks",
        "gurufocus.com",
        "benzinga",
        "investing.com",
        "simply wall st.",
        "business wire",
        "investorplace"
    ]
    try:
        req = Request(url=href, headers={"user-agent": "my-app"})
        response = urlopen(req)

        html = bs(response, features="html.parser")  # Get HTML of article

        print(publisher)
        if publisher == "motley fool":
            article_text = html.find(class_="usmf-new article-body").text
            # print("Worked")
        elif publisher in yahoo_publishers:
            article_text = html.find(class_="caas-body").text
            # print("Worked")
        elif publisher == "investor's business daily":
            article_text = html.find(
                class_="single-post-content post-content drop-cap"
            ).text
            # print("Worked")
        elif publisher == "investopedia":
            article_text = html.find(
                class_="comp article-body-content mntl-sc-page mntl-block"
            ).text
            # print("Worked")
        elif publisher == "thestreet.com":
            article_text = html.find(class_="m-detail--body").text
            # print("Worked")
        elif publisher == "marketwatch":
            article_text = html.find(
                class_="article__body article-wrap at16-col16 barrons-article-wrap"
            ).text
            # print("Worked")
        elif publisher == "barrons.com":
            article_text = html.find(class_="snippet__body").text
            # print("Worked")
        elif publisher == "quartz":
            article_text = html.find(class_="au9XZ").text
            # print("Worked")
        elif publisher == "quartz":
            article_text = html.find(class_="au9XZ").text
            # print("Worked")
        elif publisher == "quartz":
            article_text = html.find(class_="au9XZ").text
            # print("Worked")
        else:
            article_text = ""
            print("Didnt Work")
    except Exception as e:
        print("ERROR")
        article_text = ""

    # Remove unnecessary whitespace
    article_text = " ".join(article_text.split())

    return article_text

def fetch_from_db(db_name, collection_name, query):

    client = MongoClient(
        "mongodb+srv://andrisvaivods11:andyPandy2000!@cluster0.4ozrv.mongodb.net/test"
    )

    mydb = client[db_name]
    mycol = mydb[collection_name]

    cursor = mycol.find({}, {query})
    df = pd.DataFrame(list(cursor))
    return df

# TAKE ARTICLES AS INPUT, GET ID OF OLD ARTICLES AND DELETE THOSE
def delete_old_articles(ticker, days):
    df = get_articles_from_db(ticker)

    df = df.loc[: dt.datetime.now() - dt.timedelta(days=days)]

    myclient = MongoClient(
        "mongodb+srv://andrisvaivods11:andyPandy2000!@cluster0.4ozrv.mongodb.net/test"
    )
    mydb = myclient["capstone_project"]
    mycol = mydb["articles"]

    for index, row in df.iterrows():
        mycol.delete_one({"_id": row["_id"]})

    # for index, row in df.iterrows():
    # print(row)
    # if(row["date"] < dt.datetime.now() - dt.timedelta(days = days)):
    # mycol.delete_one({"_id": row["_id"]})


def get_title_sentiment(articles):

    sia = SentimentIntensityAnalyzer()

    scores = articles["Title"].apply(sia.polarity_scores).tolist()

    scores_df = pd.DataFrame(scores)

    return articles.join(scores_df, rsuffix="_right")


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


def fundamental_metric(soup, metric):
    return soup.find(text=metric).find_next(class_="snapshot-td2").text


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


def parse_stock_info(df):

    df["Dividend %"] = df["Dividend %"].str.replace("%", "")
    df["ROE"] = df["ROE"].str.replace("%", "")
    df["ROI"] = df["ROI"].str.replace("%", "")
    df["EPS Q/Q"] = df["EPS Q/Q"].str.replace("%", "")
    df["Insider Own"] = df["Insider Own"].str.replace("%", "")
    df = df.apply(pd.to_numeric, errors="coerce")

    return df


def clear_collections():
    client = MongoClient(
        "mongodb+srv://andrisvaivods11:andyPandy2000!@cluster0.4ozrv.mongodb.net/test"
    )
    collections = [
        "articles",
        "stocks",
        "stocks_info",
        "analyst_ratings",
        "latest_analyst_ratings",
        "text_summary",
        "analyst_total",
        "stock_rating",
        "avg_sentiment",
        "article_info",
        "price_prediction",
        "latest_articles"
    ]
    mydb = client["capstone_project"]
    for collection in collections:
        mycol = mydb[collection]
        x = mycol.delete_many({})

def delete_from_collection(ticker, collection):
    client = MongoClient(
        "mongodb+srv://andrisvaivods11:andyPandy2000!@cluster0.4ozrv.mongodb.net/test"
    )
    mydb = client["capstone_project"]
    mycol = mydb[collection]
    x = mycol.delete_many({"Ticker":ticker})



# TAKE ARTICLES AS INPUT, TAKE AMOUNT OF DAYS FOR AVG, COMPARE IF DAY IS INCLUDED, IF NOT BREAK, IF IS ADD TO AVG, INCREASE COUNT BY ONE
def get_avg_sentiment(articles, days):
    for index, row in articles.iterrows():
        return 0


def update_moving_avg(df, window, name):
    df_price[name] = df_price["Adj Close"].rolling(window=window, min_periods=0).mean()
    return df


def parse_analyst_ratings(df, ticker):
    start_date = dt.datetime.now() - timedelta(days=13 * 31)
    df["Date"] = pd.to_datetime(df_recom.index)

    end_date = start_date + timedelta(days=30)

    df_list = list()

    # 0: vpos, 1: pos, 2: neu, 3: neg,  4: vneg
    total = [0, 0, 0, 0, 0]

    for i in range(12):

        mask = (df["Date"] > start_date) & (df["Date"] <= end_date)

        period = df.loc[mask]

        period["To Grade"] = period["To Grade"].replace({"Overweight": "Outperform"})
        period["To Grade"] = period["To Grade"].replace(
            {"Market Outperform": "Outperform"}
        )
        period["To Grade"] = period["To Grade"].replace({"Positive": "Outperform"})
        period["To Grade"] = period["To Grade"].replace({"Underweight": "Underperform"})
        period["To Grade"] = period["To Grade"].replace({"Sector Perform": "Neutral"})
        period["To Grade"] = period["To Grade"].replace({"Market Perform": "Neutral"})
        period["To Grade"] = period["To Grade"].replace({"Hold": "Neutral"})
        period["To Grade"] = period["To Grade"].replace({"Equal-Weight": "Neutral"})
        period["To Grade"] = period["To Grade"].replace({"In-Line": "Neutral"})

        # df.loc[df["To Grade"] == "Market Outperform", "To Grade"] = "Outperform"

        values = period["To Grade"].value_counts().keys().tolist()
        counts = period["To Grade"].value_counts().tolist()

        for i in range(0, len(values)):

            if values[i] == "Buy":
                total[0] = total[0] + counts[i]
            if values[i] == "Outperform":
                total[1] = total[1] + counts[i]
            if values[i] == "Neutral":
                total[2] = total[2] + counts[i]
            if values[i] == "Underperform":
                total[3] = total[3] + counts[i]
            if values[i] == "Sell":
                total[4] = total[4] + counts[i]

        dict = {
            "Ticker": ticker,
            "Values": values,
            "Counts": counts,
            "Month": start_date.month,
            "Year": start_date.year,
        }

        df_list.append(dict)

        start_date = start_date + timedelta(days=30)
        end_date = end_date + timedelta(days=30)

        # print(pd.DataFrame(data=period).transpose)

    dict_total = {
        "Ticker": ticker,
        "Total": total,
    }

    return df_list, dict_total


def summerize_article_sentiment_by_day(df, ticker):
    start_date =min(df['Date'])
    end_date = max(df['Date'])

    #start_date = dt.datetime.now() - timedelta(days=4)
    #end_date = start_date + timedelta(days=1)


    df_avg_comp = pd.DataFrame() 

    current_date = start_date

    while current_date <= end_date:

        mask = (pd.to_datetime(df["Date"]) >= current_date) & (pd.to_datetime(df["Date"]) <= (current_date + dt.timedelta(days=1)))

        #mask = (df["Date"] >= current_date) & (df["Date"] <= current_date + dt.timedelta(hours=24-current_date.hours))

        period = df.loc[mask]

        dict = {
            "Average Comp": round(period["average_comp"].mean(), 4),
            "Day": current_date.day,
            "Month": current_date.month,
            "Ticker": ticker,
        }
        print(dict)

        df_avg_comp = df_avg_comp.append(dict, ignore_index=True)

        print(df_avg_comp)

        current_date = current_date + dt.timedelta(days = 1)

    df_avg_comp["3MA"] = df_avg_comp["Average Comp"].fillna(method="backfill").rolling(window=3, min_periods=0).mean()

    return df_avg_comp


def get_latest_articles(df):
    df = df.sort_values(by="Date")
    df_latest = df.tail(4)
    return df_latest


def create_dict_for_anaylst_ratings(df, ticker):
    firms = df["Firm"].to_list()
    grades = df["To Grade"].to_list()
    dates = df.index.to_list()
    dict = {"Firms": firms, "Grades": grades, "Dates": dates, "Ticker": ticker}

    return dict


def generate_text_summary_from_info(df, ticker):

    dicts = list()
    rating = 0


    # DIVIDEND
    if math.isnan(df["Dividend %"]):
        dicts.append(
            {"Ticker": ticker, "Text": "The Stock does not pay a Dividend", "Rating": 0}
        )
    elif df["Dividend %"].values < 1.5:
        dicts.append(
            {"Ticker": ticker, "Text": "The Stock pays a small Dividend", "Rating": 2}
        )
        rating = rating + 2
    else:
        dicts.append(
            {"Ticker": ticker, "Text": "The Stock pays a good  Dividend", "Rating": 3}
        )
        rating = rating + 3

    # RSI

    if (df["RSI (14)"].values < 30.0):
        dicts.append(
            {
                "Ticker": ticker,
                "Text": "The RSI is showing the stock is oversold",
                "Rating": 4,
            }
        )
        rating = rating + 4
    elif(df["RSI (14)"].values < 70.0):
        dicts.append(
            {
                "Ticker": ticker,
                "Text": "The RSI is not showing a trend in the stock",
                "Rating": 2,
            }
        )
        rating = rating + 2
    else:
        dicts.append(
            {
                "Ticker": ticker,
                "Text": "The RSI is showing the stock is overbought",
                "Rating": 0,
            }
        )

    # TARGET PRICE

    if (df["Target Price"].values / df["Price"].values > 1.2):
        dicts.append(
            {
                "Ticker": ticker,
                "Text": "The average analyst price for the stock in the next 12 months is a lot higher",
                "Rating": 4,
            }
        )
        rating = rating + 4
    elif(df["Target Price"].values / df["Price"].values > 1.0):
        dicts.append(
            {
                "Ticker": ticker,
                "Text": "The average analyst price for the stock in the next 12 months is slightly higher",
                "Rating": 3,
            }
        )
        rating = rating + 3
    else:
        dicts.append(
            {
                "Ticker": ticker,
                "Text": "The average analyst price for the stock in the next 12 months is lower than the current price",
                "Rating": 0,
            }
        )

    # PE RATIO

    print(df["P/E"].values)

    if df["P/E"].values < 40.0:
        dicts.append(
            {
                "Ticker": ticker,
                "Text": "The P/E ratio for the stock is quite low",
                "Rating": 3,
            }
        )
        rating = rating + 3
    elif df["P/E"].values < 100.0:
        dicts.append(
            {
                "Ticker": ticker,
                "Text": "The P/E ratio for the stock is quite high",
                "Rating": 1,
            }
        )
        rating = rating + 1
    else:
        dicts.append(
            {
                "Ticker": ticker,
                "Text": "The P/E ratio for the stock is very high",
                "Rating": 0,
            }
        )

    # VOLATILITY

    if df["Volatility"].values < 3.0:
        dicts.append(
            {
                "Ticker": ticker,
                "Text": "The one week Volatility for the stock is low",
                "Rating": 3,
            }
        )
        rating = rating + 3
    elif df["Volatility"].values < 6.0:
        dicts.append(
            {
                "Ticker": ticker,
                "Text": "The one week Volatility for the stock is quite high",
                "Rating": 2,
            }
        )
        rating = rating + 2
    else:
        dicts.append(
            {
                "Ticker": ticker,
                "Text": "The one week Volatility for the stock is high",
                "Rating": 1,
            }
        )
        rating = rating + 1

    # ISIDER OWN

    if df["Insider Own"].values > 8.0:
        dicts.append(
            {
                "Ticker": ticker,
                "Text": "Insiders own a large part of the company",
                "Rating": 3,
            }
        )
        rating = rating + 3
    elif df["Insider Own"].values > 3.0:
        dicts.append(
            {
                "Ticker": ticker,
                "Text": "Insiders own a small part of the company",
                "Rating": 2,
            }
        )
        rating = rating + 2
    else:
        dicts.append(
            {
                "Ticker": ticker,
                "Text": "Insiders own a very small part of the company",
                "Rating": 1,
            }
        )
        rating = rating + 1

    rating_avg = rating / 6.0

    dict_rating = {"Ticker": ticker, "Rating": rating_avg}

    return dicts, dict_rating

def create_prediction_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y 


def predict_price_new(ticker):

    start = dt.datetime.today() - timedelta(days=2500)
    end = dt.datetime.today()
    df_price = web.DataReader(ticker, "yahoo", start, end)

    latest_price = df_price['Open'][-1]

    df = df_price


    df.shape

    df = df['Open'].values
    df = df.reshape(-1, 1)

    dataset_train = np.array(df[:int(df.shape[0]*0.8)])
    dataset_test = np.array(df[int(df.shape[0]*0.8):])

    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_train = scaler.fit_transform(dataset_train)

    dataset_test = scaler.transform(dataset_test)



    x_train, y_train = create_prediction_dataset(dataset_train)
    x_test, y_test = create_prediction_dataset(dataset_test)


    model = Sequential()
    model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
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


    for i in range(14):
        predictions = model.predict(x_test)
        print(predictions.shape[0])
        dataset_test =  np.append(dataset_test, predictions.item(predictions.shape[0] - 2,0))
        dataset_test = np.reshape(dataset_test, (-1, 1))
        x_test, y = create_prediction_dataset(dataset_test)

    predictions = scaler.inverse_transform(predictions)
    predictions_list = []

    for i in range(15):
        predictions_list.append(predictions.tolist()[i - 15][0])

    predictions_list_perc = [1.0]

    for x in range(len(predictions_list)-1):
        predictions_list_perc.append(predictions_list[x+1] / predictions_list[0])

    predicted_prices = []

    for x in range(len(predictions_list)-1):
        predicted_prices.append(round(latest_price * predictions_list_perc[x], 2))

    return {"Ticker": ticker, "Prediction": predicted_prices}


def predict_price_old(ticker):
    start = dt.datetime.today() - timedelta(days=2500)
    end = dt.datetime.today()
    df_price = web.DataReader(ticker, "yahoo", start, end)

    latest_price = df_price['Open'][-1]

    df = df_price


    df.shape

    df = df['Open'].values
    df = df.reshape(-1, 1)

    dataset_train = np.array(df[:int(df.shape[0]*0.8)])
    dataset_test = np.array(df[int(df.shape[0]*0.8):])

    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_train = scaler.fit_transform(dataset_train)

    dataset_test = scaler.transform(dataset_test)



    x_train, y_train = create_prediction_dataset(dataset_train)
    x_test, y_test = create_prediction_dataset(dataset_test)


    model = load_model('stock_prediction_'+ticker+'.h5')

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #model.compile(loss='mean_squared_error', optimizer='adam')

    #model.fit(x_train, y_train, epochs=50, batch_size=32)


    for i in range(14):
        predictions = model.predict(x_test)
        dataset_test =  np.append(dataset_test, predictions.item(predictions.shape[0] - 2,0))
        dataset_test = np.reshape(dataset_test, (-1, 1))
        x_test, y = create_prediction_dataset(dataset_test)

    predictions = scaler.inverse_transform(predictions)
    predictions_list = []

    for i in range(15):
        predictions_list.append(predictions.tolist()[i - 15][0])

    predictions_list_perc = [1.0]

    for x in range(len(predictions_list)-1):
        predictions_list_perc.append(predictions_list[x+1] / predictions_list[0])

    predicted_prices = []

    for x in range(len(predictions_list)-1):
        predicted_prices.append(round(latest_price * predictions_list_perc[x], 2))

    return {"Ticker": ticker, "Prediction": predicted_prices}


def parse_yesterdays_articles(ticker, name, tr):
    parsed_data = []

    for row in tr:

        title = row.a.text  # Get title
        date_data = row.td.text.split(" ")  # Get date
        article_link = row.find("a", {"class": "tab-link-news"}).attrs[
            "href"
        ]  # Link for article
        # Get publisher, strip whitespace, lowercase
        publisher = row.span.text.strip().lower()

        article_text = get_article_text(article_link, publisher)

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]

        date_time = date + " " + time

        date_time = dt.datetime.strptime(date_time, "%b-%d-%y %H:%M%p ")

        print(date_time.date())
        print(dt.datetime.today().date())
 

        if(date_time.date() < dt.datetime.today().date() - dt.timedelta(days=1)): 
            break

        if article_text != "" and date_time.date() == dt.datetime.today().date() - dt.timedelta(days=1) and is_article_important(ticker.lower(), name, article_text, title):
            print("Added")
            parsed_data.append(
                [ticker, date_time, publisher, title, article_link, article_text]
            )

    return parsed_data

def get_volume(tickerData, period):
    data = tickerData.history(period=period)
    volume = data.loc[:,"Volume"]
    return volume

def generate_topics(df, name, ticker):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop.add(name)
    stop.add("stock")
    stop.add("company")
    stop.add("price")

    doc_complete = list(df["Text"])

    doc_clean = [clean_topic_text(doc, lemma, exclude, stop).split() for doc in doc_complete]


    # Creating the term dictionary of our courpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(doc_clean)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=25)

    dict_topics = []
    for item in ldamodel.print_topics(num_topics=10, num_words=2):
        dict_topics.append({"Ticker": ticker, "Topics": re.findall('"([^"]*)"', item[1])})

    print(dict_topics)

    for i in range(10):
        print(ldamodel[doc_term_matrix[i]])



    return dict_topics

def clean_topic_text(doc, lemma, exclude, stop):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized



new_tickers = ["QCOM"]
new_names = ["qualcomm"]




metric = [
        "P/B",
        "P/E",
        "Target Price",
        "RSI (14)",
        "Volatility",
        "EPS (ttm)", 
        "Dividend %",
        "ROE",
        "ROI",
        "Price",
        "Insider Own",
    ]


# LOOP FOR NEW TICKERS

for index, ticker in enumerate(new_tickers):

    name = new_names[index]


    parsed_data = []

    tickerData = yf.Ticker(ticker)

    df_recom = tickerData.recommendations
    df_cal = tickerData.calendar


    volume_df = get_volume(tickerData, "1y")
    print(volume_df)



    df_recom_last_4 = df_recom.iloc[-4:]
    dict_recom_last_4 = create_dict_for_anaylst_ratings(df_recom_last_4, ticker)

    df_recoms, dict_total_in_year = parse_analyst_ratings(df_recom, ticker)




    print("GETTING ARTICLES")
    articles = get_all_articles(ticker)
    parsed_data = parse_articles(ticker, name, articles.find_all("tr"))


    df = pd.DataFrame(
        parsed_data, columns=["Ticker", "Date", "Publisher", "Title", "Link", "Text"]
    )


    print("GENERATING TOPICS")
    dict_topics = generate_topics(df, name, ticker)

    print("GETTING TITLE SENTIMENT")
    df = get_title_sentiment(df)


    print("GETTING TEXT SENTIMENT")

    df_text_sent = get_text_semtiment(df)
    df_text_sent = pd.DataFrame(df_text_sent)

    df_text_sent = df_text_sent.rename(
        columns={
            "neg": "neg_text",
            "neu": "neu_text",
            "pos": "pos_text",
            "compound": "compound_text",
        }
    )

    print(df_text_sent)


    df = df.join(df_text_sent, rsuffix="_right")

    print("GETTING PRICE")
    df_price = get_stock_price(ticker, 365)
    df_price["20MA"] = df_price["Adj Close"].rolling(window=20, min_periods=0).mean()


    print("GETTING VOLUME")

    print("MAKING PREDICTIONS")
    dict_prediction = predict_price_new(ticker)



    print("GETTING STOCK INFO")
    df_info = pd.DataFrame(index=[ticker], columns=metric)
    df_info = get_fundamental_data(df_info)

    volatility = df_info["Volatility"].str.split()

    df_info["Dividend %"] = df_info["Dividend %"].str.replace("%", "")
    df_info["ROE"] = df_info["ROE"].str.replace("%", "")
    df_info["ROI"] = df_info["ROI"].str.replace("%", "")
    df_info["Volatility"] = volatility[0][0].replace("%", "")
    df_info["Insider Own"] = df_info["Insider Own"].str.replace("%", "")

    df_info = df_info.apply(pd.to_numeric, errors="coerce")


    df_info["Ticker"] = df_info.index
    df_info.index = range(1, len(df_info) + 1)


    dicts_info_text, dict_rating = generate_text_summary_from_info(df_info, ticker)



    df["average_neg"] = round(df[["neg", "neg_text"]].mean(axis=1), 2)
    df["average_pos"] = round(df[["pos", "pos_text"]].mean(axis=1), 2)
    df["average_neu"] = round(df[["neu", "neu_text"]].mean(axis=1), 2)
    df["average_comp"] = round(df[["compound", "compound_text"]].mean(axis=1), 2)

    df = df.drop(
        columns=[
            "neu",
            "pos",
            "neg",
            "neu_text",
            "pos_text",
            "neg_text",
            "compound",
            "compound_text",
        ]
    )   


    latest_article_df = get_latest_articles(df)
    df_avg_comp = summerize_article_sentiment_by_day(df, ticker)
    print(df_avg_comp)

    print("INSERTING NEW DATA")
    for df_recom in df_recoms:
        insert_dict_into_db(df_recom, "analyst_ratings")
    for dict in dicts_info_text:
        insert_dict_into_db(dict, "text_summary")
    for dict in dict_topics:
        insert_dict_into_db(dict, "topics")

    insert_dict_into_db(dict_rating, "stock_rating")
    insert_dict_into_db(dict_prediction, "price_prediction")
    insert_dict_into_db(dict_recom_last_4, "latest_analyst_ratings")
    insert_dict_into_db(dict_total_in_year, "analyst_total")
    insert_df_into_db(df_price, "stocks")
    insert_df_into_db(df_avg_comp, "avg_sentiment")
    insert_df_into_db(latest_article_df, "latest_articles")
    insert_df_into_db(df_info, "stocks_info")
    insert_df_into_db(df, "articles")


    # TODO REMOVE LAST 3 SENTENCES FOE ARTICLES
    # TODO UPDATE STOCK PREDICTOR
    # TODO GET LAST ITEM IN LIST IN PREDICTOR

    # TODO MAKE STOCK SCREENER
    # TODO ADD TOOLTIPS FOR GRAPHS


# LOOP FOR OLD TICKERS (UPDATE)


old_tickers = ["AMZN", "TSLA", "BABA", "AAPL", "GOOGL", "MA", "DIS", "INTC", "NFLX", "NVDA", "PINS", 
            "GME", "MSFT", "JPM", "NKE", "SPOT", "PLTR", "ADBE", "PYPL", "ROKU"]
old_names = ["amazon", "tesla", "alibaba", "apple", "google", "mastercard", "disney", "intel", "netflix", "nvidia", "pinterest", 
            "gamestop", "microsoft", "jpmorgan", "nike", "spotify", "palantir", "adobe", "paypal", "roku"]

for index, ticker in enumerate(old_tickers):

    name = old_names[index]

    tickerData = yf.Ticker(ticker)

    df_recom = tickerData.recommendations
    df_cal = tickerData.calendar

    df_recom_last_4 = df_recom.iloc[-4:]
    dict_recom_last_4 = create_dict_for_anaylst_ratings(df_recom_last_4, ticker)

    df_recoms, dict_total_in_year = parse_analyst_ratings(df_recom, ticker)


    print("GETTING ARTICLES FROM DB")

    articles_from_db_df = get_articles_from_db(ticker)


    print("GETTING PRICE")
    df_price = get_stock_price(ticker, 365)
    df_price["20MA"] = df_price["Adj Close"].rolling(window=20, min_periods=0).mean()

    dict_prediction = predict_price_old(ticker)

    print("GETTING LATEST ARTICLES")
    articles = get_all_articles(ticker)
    parsed_data = parse_yesterdays_articles(ticker, name, articles.find_all("tr"))

    df = pd.DataFrame(
        parsed_data, columns=["Ticker", "Date", "Publisher", "Title", "Link", "Text"]
    )


    print("GETTING TITLE SENTIMENT")
    df = get_title_sentiment(df)

    print("GETTING TEXT SENTIMENT")
    df_text_sent = get_text_semtiment(df)
    df_text_sent = pd.DataFrame(df_text_sent)

    df_text_sent = df_text_sent.rename(
        columns={
            "neg": "neg_text",
            "neu": "neu_text",
            "pos": "pos_text",
            "compound": "compound_text",
        }
    )

    df = df.join(df_text_sent, rsuffix="_right")

    try:
        df["average_neg"] = round(df[["neg", "neg_text"]].mean(axis=1), 2)
        df["average_pos"] = round(df[["pos", "pos_text"]].mean(axis=1), 2)
        df["average_neu"] = round(df[["neu", "neu_text"]].mean(axis=1), 2)
        df["average_comp"] = round(df[["compound", "compound_text"]].mean(axis=1), 2)

        df = df.drop(
        columns=[
            "neu",
            "pos",
            "neg",
            "neu_text",
            "pos_text",
            "neg_text",
            "compound",
            "compound_text",
        ]
    )

        df_concat = pd.concat([articles_from_db_df, df])

    except:
        print("No Important articles")
        df_concat = articles_from_db_df

    

    print("GENERATING TOPICS")
    dict_topics = generate_topics(df_concat, name, ticker)



    print("GETTING STOCK INFO")
    df_info = pd.DataFrame(index=[ticker], columns=metric)
    df_info = get_fundamental_data(df_info)

    volatility = df_info["Volatility"].str.split()

    df_info["Dividend %"] = df_info["Dividend %"].str.replace("%", "")
    df_info["ROE"] = df_info["ROE"].str.replace("%", "")
    df_info["ROI"] = df_info["ROI"].str.replace("%", "")
    df_info["Volatility"] = volatility[0][0].replace("%", "")
    df_info["Insider Own"] = df_info["Insider Own"].str.replace("%", "")
    df_info = df_info.apply(pd.to_numeric, errors="coerce")

    df_info["Ticker"] = df_info.index
    df_info.index = range(1, len(df_info) + 1)


    dicts_info_text, dict_rating = generate_text_summary_from_info(df_info, ticker)

    print("Getting Latest Analyst ratings")
    latest_article_df = get_latest_articles(df_concat)

    df_avg_comp = summerize_article_sentiment_by_day(df_concat, ticker)




    latest_article_df =  latest_article_df.set_index('Date')
    latest_article_df =  latest_article_df.drop(columns=['_id'])

    print("DELETING OLD DATA")
    delete_from_collection(ticker, "stocks_info")
    delete_from_collection(ticker, "stocks")
    delete_from_collection(ticker, "analyst_ratings")
    delete_from_collection(ticker, "price_prediction")
    delete_from_collection(ticker, "text_summary")
    delete_from_collection(ticker, "analyst_total")
    delete_from_collection(ticker, "avg_sentiment")
    delete_from_collection(ticker, "stock_rating")
    delete_from_collection(ticker, "latest_articles")
    delete_from_collection(ticker, "latest_analyst_ratings")
    delete_from_collection(ticker, "topics")


    print("INSERTING NEW DATA")
    for df_recom in df_recoms:
        insert_dict_into_db(df_recom, "analyst_ratings")
    for dict in dicts_info_text:
        insert_dict_into_db(dict, "text_summary")
    for dict in dict_topics:
        insert_dict_into_db(dict, "topics")
    insert_dict_into_db(dict_rating, "stock_rating")
    insert_df_into_db(df_avg_comp, "avg_sentiment")
    insert_dict_into_db(dict_prediction, "price_prediction")
    insert_dict_into_db(dict_recom_last_4, "latest_analyst_ratings")
    insert_dict_into_db(dict_total_in_year, "analyst_total")
    insert_df_into_db(df_price, "stocks")
    insert_df_into_db(latest_article_df, "latest_articles")
    insert_df_into_db(df_info, "stocks_info")

    if(not df.empty):
        insert_df_into_db(df, "articles")



