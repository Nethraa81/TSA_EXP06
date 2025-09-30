# Ex.No: 06                                       HOLT WINTERS METHOD
### Date: 30.09.2025



### AIM:
To implement the Holt Winters Method Model using Python.
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative
trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and
Evaluate the model predictions against test data
6. Create teh final model and predict future data and plot i
### PROGRAM:

Import libraries:
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

```
Load & clean dataset:
```
data = pd.read_csv("India_GDP.csv", header=1)
data.columns = data.columns.str.strip()

data = data[['Year', 'GDP in (Billion) $']]
data.columns = ['Year', 'GDP']
data['Year'] = data['Year'].astype(int)
data = data.sort_values('Year')
data.set_index('Year', inplace=True)

gdp_series = pd.Series(data['GDP'].values,
                       index=pd.PeriodIndex(data.index, freq='Y'))
gdp_series.head()

```

Scale data & plot:
```
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(gdp_series.values.reshape(-1,1)).flatten(),
                        index=gdp_series.index)

scaled_data.plot(title="Scaled GDP Data", figsize=(10,6))
plt.show()

```

Decomposition:
```
gdp_series_dt = pd.Series(gdp_series.values,
                          index=pd.to_datetime(gdp_series.index.astype(str)))

decomposition = seasonal_decompose(gdp_series_dt, model="additive", period=5)
decomposition.plot()
plt.show()

```

Train/Test split:
```
scaled_data = scaled_data + 1  
train = scaled_data[:int(len(scaled_data)*0.8)]
test  = scaled_data[int(len(scaled_data)*0.8):]

train.tail(), test.head()

```

Holt-Winters model & test prediction:
```
model = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
test_predictions = model.forecast(len(test))

ax = train.plot(label="Train", figsize=(10,6))
test.plot(ax=ax, label="Test")
test_predictions.plot(ax=ax, label="Test Predictions")
plt.legend()
plt.title("Test Prediction (Scaled GDP)")
plt.show()

```

Model performance metrics:
```
rmse = np.sqrt(mean_squared_error(test, test_predictions))
std_dev = np.sqrt(scaled_data.var())
mean_val = scaled_data.mean()

print("Model performance metrics:")
print("RMSE:", rmse)
print("Standard Deviation:", std_dev)
print("Mean:", mean_val)

```

Final model & forecast:
```
final_model = ExponentialSmoothing(scaled_data, trend='add', seasonal=None).fit()
final_predictions = final_model.forecast(5)

ax = scaled_data.plot(title="Final GDP Prediction (Scaled)", figsize=(10,6))
final_predictions.plot(ax=ax, label="Forecast", style="--")
plt.legend()
plt.show()

print("Final Predictions (next 5 years, scaled):")
print(final_predictions)

```

### OUTPUT:

## Scaled_data plot:


<img width="1033" height="629" alt="image" src="https://github.com/user-attachments/assets/8c1cad52-5cd5-469c-9a41-bc5c365fc572" />

## Decomposed plot:


<img width="889" height="573" alt="image" src="https://github.com/user-attachments/assets/3aeefc20-d17c-4325-bf7c-554320f317a6" />

## Test prediction:


<img width="1043" height="656" alt="image" src="https://github.com/user-attachments/assets/4a05f7b5-3a31-45f7-8979-852dcd1f5159" />

## Model performance metrics:


<img width="348" height="69" alt="image" src="https://github.com/user-attachments/assets/f0b95592-6088-4328-a65b-c5723452ec2c" />

## Final prediction:


<img width="1085" height="653" alt="image" src="https://github.com/user-attachments/assets/3ac3551b-e8c0-4725-b7b8-76518c0df45d" />


### RESULT:
Thus we have successfully implemented the Holt Winters method using python.
