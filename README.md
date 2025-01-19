# Exploring_Stock_Prediction
Exploring Stock market price prediction using LSTM 

## Resources:
- [AlphaVantage](https://www.alphavantage.co/)
- [DataCamp](https://www.datacamp.com/tutorial/lstm-python-stock-market)
- [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-to-time-series-analysis/)


This project uses LSTM to predict mid stock prices for a defined stock ticker. The data is pulled from the AlphaVantage API, and the frameworks used for the actual model is pytorch rather than tensorflow. Mid prices is defined as:

$mid = (high + low) / 2$


The data used for these is for the Berkshire B ticker (BRK.B).

### 1. EDA
Exploratory Data Analysis reveals various Trends among the features against Date.

- Date vs Low Prices
![image](https://github.com/user-attachments/assets/4e4d4316-2e3f-438c-9c28-f6b1f6427411)


- Date vs High Prices
![image](https://github.com/user-attachments/assets/537c6381-c7c4-48dc-b27f-110879e129e0)

- Date vs Open Prices
![image](https://github.com/user-attachments/assets/71db2a17-5f91-48bf-b621-3e27685b13f4)

- Date vs Cloase Prices
![image](https://github.com/user-attachments/assets/68c105af-1d1d-4fa7-becb-355ba946b3db)

- Date vs Volume
![image](https://github.com/user-attachments/assets/0a032ccf-90b8-43d3-a0e7-bf7b4bca4a42)


The plots clearly show a sharp fall in all prices between 2008 - 2012 when the volume of stock traded sky rockets. The volume of traded stocks remain relatively high to pre 2008 period and the prices remain relatively low.

- Date vs Low and High Prices
![image](https://github.com/user-attachments/assets/3d39e781-304c-429c-82e5-5dd8636902d2)

- Date vs Open and Close Prices
![image](https://github.com/user-attachments/assets/174bfcea-3f9b-4083-99bc-1bcba4011c81)

The ADF(Augmented Dickey Fuller) Test reveals, as expected, that expect for volume traded, all other features(not including Date) are Non stationary.
```
For Low : p-value = 0.5481909423783445 : Series is Non Staionary!
For High : p-value = 0.5967818325454737 : Series is Non Staionary!
For Open : p-value = 0.5634536440565823 : Series is Non Staionary!
For Close : p-value = 0.552665311315926 : Series is Non Staionary!
For Volume : p-value = 2.420874225174702e-16 : Series is Stationhary!
```

###2. Preprocessing
The preprocessing includes normalization of mid prices on a smoothing window. The size of the window here was taken to be 1000. The train set includes first $80%$ of the data while the test set contains the remaining $20%$ of the data. The data itself is sorted by date, before doing the train test split.

### 3. LSTM
**-Data Loading**
The `DataGeneratorSequence` class handles loading the data in approriate chunks to the lstm model. It Divides the input data into batches based on `batch_size` and length of data available and returns a tensor containing `batch_size` number of stacked tensor each having a sequence length equal to `num_unrolls`. `num_unrolls` signifies the number of timesteps to be handled by the lstm in conjunction by maintaining hidden and cell states. It also returns the ground truth which is randomly sampled from the next values availabe in the batch rather than the exact price of the next day.

**-Model Architecture**
The model follows the same architecture as defined in the [Datacamp](https://www.datacamp.com/tutorial/lstm-python-stock-market) guide collated into a single class implemented using pytorch rather than tensorflow. The hidden and cell state retention can be controlled using the `pass_state` variable. The model is configured to predict only one value, normalized mid prices for the next few days.

**-Training**
The training has been done for 30 - 50 epochs with equivalent hyperparameters as given in the Datacamp Guide with states retained only for the single batch of inputs with sequence lengths equal to `num_unrolls` as retaining for the enire training sequence generates worse results.

**-Testing and Prediction**
For prediction purposes, the model is first primed(to generate hidden and cell state information) using data for the first `num_unrolls` datees, then these states are retained and prediction is done for the next `num_pred` days.

### 4. Conclusion
Although the LSTM seems more robust in predictions for shorter lengths, especially when primed with train data, it seems to suffer as the prediction window is increased and increasingly as the model is primed with unseen test data.
