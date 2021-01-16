# Importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
# Importing the dataset
df = pd.read_csv('Global.csv',usecols=['Category', 'Market','Date','Profit','Sales'])
df1 = df.loc[(df['Category'] == 'Technology')  & (df['Market'] == 'APAC')]
#df1 = df.loc[(df['Category'] == 'Office Supplies')  & (df['Market'] == 'EU')]           
#print(df1)
#print(df2)
df3 = df1[['Date','Sales']]
#print(df3)
df3['Date']=pd.to_datetime(df3['Date'])
#print(df3)
#print(df3.dtypes)
ts=df3.set_index('Date').resample('M', 'sum')
#print(ts)
X = ts.values
size = int(len(X) * 0.9)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(steps=1, exog=None, alpha=0.05)
    yhat = output[0]    
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
print(np.mean(np.abs((test - predictions) / test)) * 100)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()  