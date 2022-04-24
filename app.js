const http = require("http");
let fs = require('fs');
var jsonQuery = require('json-query')
const Chart = require('chart.js');
const express = require('express')
const app = express()

const port = process.env.PORT || 5000

app.use(express.static('public'));
app.use('/css', express.static(__dirname + 'public/css'))
app.use('/js', express.static(__dirname + 'public/js'))
app.use('/img', express.static(__dirname + 'public/img'))

app.set('views', './views')
app.set('view engine', 'ejs')


var globalresult;
var global_info;
var global_analyst;
var global_analyst_latest;
var global_info_all;
var global_articles;
var global_text_summary;
var global_prediction;
var global_stock_names;
var global_stock_rating;

const { MongoClient } = require("mongodb");
const { Console } = require("console");

const uri = "mongodb+srv://andrisvaivods11:andyPandy2000%21@cluster0.4ozrv.mongodb.net/test?authSource=admin&replicaSet=atlas-110lu1-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true";
MongoClient.connect(uri, function (err, db) {
    if (err) throw err;
    var dbo = db.db("capstone_project");

    dbo.collection("stock_price").find({}).toArray(function (err, result) {
        if (err) throw err;
        globalresult = result
    });

    dbo.collection("analyst_ratings").find().toArray(function (err, result) {
        if (err) throw err;
        global_analyst = result
    });

    dbo.collection("stock_rating").find().toArray(function (err, result) {
        if (err) throw err;
        global_stock_rating = result
    });

    dbo.collection("latest_articles").find().toArray(function (err, result) {
        if (err) throw err;
        global_articles = result
        console.log(global_articles)
    });
    dbo.collection("latest_analyst_ratings").find().toArray(function (err, result) {
        if (err) throw err;
        global_analyst_latest = result
    });
    dbo.collection("price_prediction").find().toArray(function (err, result) {
        if (err) throw err;
        global_prediction = result
    });

    dbo.collection("text_summary").find().toArray(function (err, result) {
        if (err) throw err;
        global_text_summary = result
    });
    dbo.collection("analyst_total").find().toArray(function (err, result) {
        if (err) throw err;
        global_analyst_total = result
    });

    dbo.collection("avg_sentiment").find().toArray(function (err, result) {
        if (err) throw err;
        global_article_sent = result
    });

    dbo.collection("stocks_info").find().toArray(function (err, result) {
        if (err) throw err;
        global_info = result
        //db.close();
    });
    dbo.collection("stocks").find().toArray(function (err, result) {
        if (err) throw err;
        global_stock_names = result
        //db.close();
    });
    dbo.collection("topics").find().toArray(function (err, result) {
        if (err) throw err;
        global_topics = result
        //db.close();
    });
});

app.get('/screener', (req, res) => {
    res.render("screener", { text: "123" })
})

app.get('/stock/:ticker', (req, res) => {
    var stock = req.params.ticker
    res.render("index", { ticker: stock })
})

app.get('', (req, res) => {
    res.render("search", { stock_names: "test" })
})


app.get('/prediction/:ticker', (req, res) => {
    var query = req.params.ticker
    var result = []
    global_prediction.forEach(element => {
        if(element["Ticker"] === query){
            result.push(element)
        }
    });
    res.json(result)
})

app.get('/price/:ticker', (req, res) => {
    var query = req.params.ticker
    var result = []
    globalresult.forEach(element => {
        if(element["Ticker"] === query){
            result.push(element)
        }
    });
    res.json(result)
})

app.get('/info/:ticker', (req, res) => {
    var result = []
    var query = req.params.ticker
    global_info.forEach(element => {
        if(element["Ticker"] === query){
            result.push(element)
        }
    });
    res.json(result)
})

app.get('/rating/:ticker', (req, res) => {
    var result = []
    var query = req.params.ticker
    global_stock_rating.forEach(element => {
        if(element["Ticker"] === query){
            result.push(element)
        }
    });
    res.json(result)
})


app.get('/analyst/:ticker', (req, res) => {
    var result = []
    var query = req.params.ticker
    global_analyst.forEach(element => {
        if(element["Ticker"] === query){
            result.push(element)
        }
    });
    res.json(result)
})


app.get('/text_summary/:ticker', (req, res) => {
    var result = []
    var query = req.params.ticker
    global_text_summary.forEach(element => {
        if(element["Ticker"] === query){
            result.push(element)
        }
    });
    res.json(result)
})



app.get('/analyst_total/:ticker', (req, res) => {
    var result = []
    var query = req.params.ticker
    global_analyst_total.forEach(element => {
        if(element["Ticker"] === query){
            result.push(element)
        }
    });
    res.json(result)
})

app.get('/articles_latest/:ticker', (req, res) => {
    var result = []
    var query = req.params.ticker
    global_articles.forEach(element => {
        if(element["Ticker"] === query){
            result.push(element)
        }
    });
    res.json(result)
})

app.get('/analyst_latest/:ticker', (req, res) => {
    var result = []
    var query = req.params.ticker
    global_analyst_latest.forEach(element => {
        if(element["Ticker"] === query){
            result.push(element)
        }
    });
    res.json(result)
})

app.get('/info_all', (req, res) => {
    res.json(global_info)
})

app.get('/stock_names', (req, res) => {
    res.json(global_stock_names)
})

app.get('/article_sent/:ticker', (req, res) => { 
    var result = []
    var query = req.params.ticker
    global_article_sent.forEach(element => {
        if(element["Ticker"] === query){
            result.push(element)
        }
    });
    res.json(result)
})


app.get('/topics/:ticker', (req, res) => { 
    var result = []
    var query = req.params.ticker
    global_topics.forEach(element => {
        if(element["Ticker"] === query){
            result.push(element)
        }
    });
    res.json(result)
})










app.listen(port, () => console.info('Listening on port'))

