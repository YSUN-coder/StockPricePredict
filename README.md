# StockPricePredict

## Introduction
The outcome (The Stock Price) should be continuous, so this is a regression problem.

## Future Work
- For Regression, the way to evaluate the model performance is with a metric called RMSE (Root Mean Squared Error). It is calculated as the root of the mean of the squared differences between the predictions and the real values.


```
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

```


However, in Stock Price Prediction scenario, we should be more interested in the directions, rather than the closeness of their values to the real stock price. That's why the RMSE as the loss founction doesn't make sense.

- Train it on the past 10 years.
- Increase the number of timesteps, e.g. 60 to 120
- Adding other indictors
- Adding more LSTM Layers
- Adding more neurons in the LSTM layers to respond better to the complexity of the problem.

## Data Source
Yahoo Finance
