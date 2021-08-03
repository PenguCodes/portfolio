### In this file I will attempt to create a program that will calculate an equal weight version of different popular indices ###
# it is currently set to the DOW but can be changed to sp500 or others as desired.
# This is NOT financial advice. Past performance does not indicate future returns. This is for showcase purposes only ###
# ### This is a work in progress ###


import pandas_datareader as web
import datetime as dt
import pandas as pd
import math
from yahoo_fin import stock_info


start = dt.datetime.now()
end = start
#setting time to now so we get the latest data on price

stocks_data = pd.DataFrame()
#creating an empty data frame to add info to


tickers = stock_info.tickers_dow()
#stock_info.tickers_sp500() also works! We use the DOW tickers so the code runs a bit faster and we can see the whole table at the end!

stocks_data["Tickers"] = tickers
stocks_data["Price"] = web.get_quote_yahoo(tickers)["price"].values
stocks_data["Market Cap"] = web.get_quote_yahoo(tickers)["marketCap"].values
stocks_data["Shares to Buy"] = "N/A"

#adding information to the data frame. Keeping shares to buy as NA to modify later

portfolio_size = input("Enter the value of your portfolio:")
#input for the user for their portfolio size

try:
    val = float(portfolio_size)
except ValueError:
    print("That's not a number! \n Try again:")
    portfolio_size = input("Enter the value of your portfolio:")
#Make sure the user submits a valid portfolio size!


position_size = float(portfolio_size) / len(stocks_data.index)
#getting the amount to spend on each of our stocks


for i in range(0, len(stocks_data['Tickers'])):
    stocks_data.loc[i, "Shares to Buy"] = math.floor(position_size / stocks_data['Price'][i])
#adding the correct amount of stocks to each ticker. We use floor as we do not want to round up as we might run out of funds.
#we also dont use fractions as some brokers do not allow for fractional shares.


print(stocks_data)

#print the outcome! if you have a large amount of stocks not all of them might show :(
