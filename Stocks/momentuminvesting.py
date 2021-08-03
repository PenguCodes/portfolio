### In this file I will attempt to create a simple equal weight momentum portfolio using Yahoo Finance and select stocks ###
# This is NOT financial advice. Past performance does not indicate future returns. This is for showcase purposes only ###
# ### This is a work in progress ###


import pandas_datareader as web
import pandas as pd
from yahoo_fin import stock_info
from scipy import stats
from statistics import mean
import math

positions_to_consider = 15
#How many stocks we want to consider for our final portfolio!


stocks_data = pd.DataFrame()
#Empty data frame to add our results and columns too

tickers = stock_info.tickers_dow()
# stock_info.tickers_sp500() also works! We use the DOW tickers so the code runs a bit faster and we can see the whole table at the end!

stocks_data["Tickers"] = tickers
stocks_data["Shares to Buy"] = "N/A"
stocks_data["200 Day Average Change Percent"] = web.get_quote_yahoo(tickers)["twoHundredDayAverageChangePercent"].values
stocks_data["50 Day Average Change Percent"] = web.get_quote_yahoo(tickers)["fiftyDayAverageChangePercent"].values
stocks_data["52 Week High Change Percent"] = web.get_quote_yahoo(tickers)["fiftyTwoWeekHighChangePercent"].values
stocks_data["52 Week Low Change Percent"] = web.get_quote_yahoo(tickers)["fiftyTwoWeekLowChangePercent"].values
stocks_data["Price"] = web.get_quote_yahoo(tickers)["price"].values
stocks_data["200 Day Average Change Percentile"] = "N/A"
stocks_data["50 Day Average Change Percentile"] = "N/A"
stocks_data["52 Week High Change Percentile"] = "N/A"
stocks_data["52 Week Low Change Percentile"] = "N/A"
stocks_data["Score"] = "N/A"


#Creating all our  colums and populating some of them from Yahoo Finance

for row in stocks_data.index:
    stocks_data["200 Day Average Change Percentile"].loc[row] = stats.percentileofscore(stocks_data["200 Day Average Change Percent"], stocks_data["200 Day Average Change Percent"].loc[row]) / 100
    stocks_data["50 Day Average Change Percentile"].loc[row] = stats.percentileofscore(stocks_data["50 Day Average Change Percent"], stocks_data["50 Day Average Change Percent"].loc[row]) / 100
    stocks_data["52 Week High Change Percentile"].loc[row] = stats.percentileofscore(stocks_data["52 Week High Change Percent"], stocks_data["52 Week High Change Percent"].loc[row]) / 100
    stocks_data["52 Week Low Change Percentile"].loc[row] = stats.percentileofscore(stocks_data["52 Week Low Change Percent"], stocks_data["52 Week Low Change Percent"].loc[row]) / 100

#Populating the Percentile columns for our DataFrame

for row in stocks_data.index:
    stocks_data["Score"].loc[row] = mean([stocks_data["200 Day Average Change Percentile"].loc[row],stocks_data["50 Day Average Change Percentile"].loc[row],stocks_data["52 Week High Change Percentile"].loc[row], stocks_data["52 Week Low Change Percentile"].loc[row]])

#to get a better momentum estimator its good to not only take short or only take long inputs, we take a mean of the percentiles of different varibales to get a better estimator.

stocks_data = stocks_data.sort_values(by="Score", ascending=False)
stocks_data = stocks_data.reset_index(drop=True)
stocks_data = stocks_data[:positions_to_consider]

#Sorting and filtering our columns to only consider the desired amounts of stocks


portfolio_size = input("Enter the value of your portfolio:")

#How much money do we want to invest?

try:
    val = float(portfolio_size)
except ValueError:
    print("That's not a number! :( \n Try again:")
    portfolio_size = input("Enter the value of your portfolio:")

position_size = float(portfolio_size) / len(stocks_data.index)

#Calculating the amount of money per position

for i in range(0, len(stocks_data['Tickers'])):
    stocks_data.loc[i, "Shares to Buy"] = math.floor(position_size / stocks_data['Price'][i])

#Calculating the amount of shares to buy and adding them to our dataframe

print(stocks_data.iloc[:, [0, 1]])

#Printing our result
