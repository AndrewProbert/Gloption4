import yfinance as yf

start_date = "2019-01-01"
end_date = "2023-11-19"

# Get the data for the stock AAPL
data = yf.download("AAPL", start=start_date, end=end_date)
print(data)

