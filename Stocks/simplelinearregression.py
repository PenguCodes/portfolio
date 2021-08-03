### In this file I will attempt to create an additive linear regression model to predict stock price ###
### This is a work in progress ###

import pandas_datareader as web
import datetime as dt
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plot
import seaborn as sns


ticker = "GME" ### Ticker Symbol for the company you would like the prediction for
prediction = "Adj Close" #could also be Open or Close

train_percent = 0.8
test_percent = 1 - train_percent
### these lines set our percentages for training and testing our model


train_days = int(round(252*train_percent, 0))
test_days = int(252 - train_days)
### these calculate the days for prediction for training and testing our model


###
start_time = dt.datetime.now() - dt.timedelta(days=365)
end_time = dt.datetime.now()
### these lines of code tell the program the date ranges that it should consider in the calculation


ticker_dataframe = web.DataReader(ticker, "yahoo", start_time, end_time)
### the line above takes the information from Yahoo Finance and imports it into the program so we can use it

ticker_dataframe["Date"] = ticker_dataframe.index.to_frame(index=True)
### Adding Date as a column in the data frame to use in other calculations

ticker_dataframe["Time"] = np.arange(1,253,1)
ticker_dataframe["Month_Name"] = ticker_dataframe["Date"].dt.month_name()
### Adding Time & Month Name as a column in the data frame to use in other calculations


ticker_dataframe_train = ticker_dataframe.head(train_days)
ticker_dataframe_test = ticker_dataframe.iloc[train_days:]
###  Setting our training and testing frames

x_train_data = ticker_dataframe_train[["Time"]].values
y_train_data = ticker_dataframe_train[prediction].values
x_test_data = ticker_dataframe_test[["Time"]].values
y_test_data = ticker_dataframe_test[prediction].values
###  Setting our training data to value and making sure they have the right dimensions

# print(ticker_dataframe_test.head())


model_trend = LinearRegression()
model_trend.fit(x_train_data,y_train_data)
y_fitted = model_trend.predict(x_train_data)
y_forecast = model_trend.predict(x_test_data)

#print the data

plot.figure(figsize=(12, 8))
plot.plot(ticker_dataframe_train["Date"], y_train_data, "bo:")
plot.plot(ticker_dataframe_train["Date"], y_fitted, "b")
plot.plot(ticker_dataframe_test["Date"], y_test_data, "o:", color="black")
plot.plot(ticker_dataframe_test["Date"], y_forecast, "b", color="red")
plot.show()

#plot the data

