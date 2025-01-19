# Exploring_Stock_Prediction
Exploring Stock market price prediction using LSTM 

**Resources:**
- [AlphaVantage](https://www.alphavantage.co/)
- [DataCamp](https://www.datacamp.com/tutorial/lstm-python-stock-market)
- [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-to-time-series-analysis/)


This project uses LSTM to predict mid stock prices for a defined stock ticker. The data is pulled from the AlphaVantage API, and the frameworks used for the actual model is pytorch rather than tensorflow. Mid prices is defined as:

$mid = (high + low) / 2$


The data used for these is for the Berkshire B ticker (BRK.B).
