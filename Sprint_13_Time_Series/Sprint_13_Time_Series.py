#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Thanks for taking the time to improve the project! It is now accepted. Good luck on the next sprint!
# 
# </div>

# **Review**
# 
# Hi, my name is Dmitry and I will be reviewing your project.
#   
# You can find my comments in colored markdown cells:
#   
# <div class="alert alert-success">
#   If everything is done successfully.
# </div>
#   
# <div class="alert alert-warning">
#   If I have some (optional) suggestions, or questions to think about, or general comments.
# </div>
#   
# <div class="alert alert-danger">
#   If a section requires some corrections. Work can't be accepted with red comments.
# </div>
#   
# Please don't remove my comments, as it will make further review iterations much harder for me.
#   
# Feel free to reply to my comments or ask questions using the following template:
#   
# <div class="alert alert-info">
#   For your comments and questions.
# </div>
#   
# First of all, thank you for turning in the project! You did a pretty good job overall, but there are some issues that need to be fixed before the project is accepted. Let me know if you have questions!

# # Project description
# 
# Sweet Lift Taxi company has collected historical data on taxi orders at airports. To attract more drivers during peak hours, we need to predict the amount of taxi orders for the next hour. Build a model for such a prediction.
# 
# The RMSE metric on the test set should not be more than 48.
# 
# ## Project instructions
# 
# 1. Download the data and resample it by one hour.
# 2. Analyze the data.
# 3. Train different models with different hyperparameters. The test sample should be 10% of the initial dataset. 
# 4. Test the data using the test sample and provide a conclusion.
# 
# ## Data description
# 
# The data is stored in file `taxi.csv`. The number of orders is in the '*num_orders*' column.

# ## Preparation

# In[1]:


get_ipython().run_line_magic('pip', 'install pmdarima --user')


# Note: I had to restart the kernel before getting the pmdarima library to work.

# In[2]:


from pmdarima import auto_arima


# In[98]:


# import libraries
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import arma_order_select_ic
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MaxAbsScaler


# In[4]:


# load dataframe by parsing the time series
data = pd.read_csv('/datasets/taxi.csv', index_col=[0], parse_dates=[0])


# In[5]:


# display
display(data)


# In[6]:


data.info()
# 6 months


# In[7]:


# make sure time series is in order
data.sort_index(inplace=True)


# In[8]:


# Check if the time is monotonic
print(data.index.is_monotonic)


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Good idea to check that the time series is in correct order
# 
# </div>

# In[9]:


# Check for nulls
data.isna().sum()


# In[10]:


# check for duplicates
data[data.duplicated()]


# The duplicates would make sense since they should be at different times.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Right!
# 
# </div>

# In[11]:


print(data['num_orders'].unique())
print(data['num_orders'].nunique())


# In[12]:


data.value_counts()


# In[13]:


# view the descriptive statistics
data.describe()


# In[14]:


# Creating plot
plt.figure(figsize=(11, 3))
plt.boxplot(data, vert = 0)
plt.xlabel("num_orders")
plt.title("Taxi Requests")
#plt.figure(figsize=(8, 6))
# show plot
plt.show()


# There appear to be many outliers. We will need to check the seasonality to see if there are repeating peaks.
# 
# Resampling the data for Sweet Lift Taxi company because they are looking at peak hours.

# In[15]:


# Using the mean to resample by the hour
#data_hour = data.resample('1H').mean()
data_hour = data.resample('1H').sum()
decomposed = seasonal_decompose(data_hour)


# <div class="alert alert-danger">
# <s><b>Reviewer's comment</b>
# 
# We're looking to predict the total number of orders in an hour, not a mean value of orders in 10-minute intervals in an hour:)
# 
# </div>

# <div class="alert alert-info">
#   Changed the resampling to the sum instead of the mean.
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Great!
# 
# </div>

# In[16]:


display(data_hour)


# The data has been inspected and prepared.

# ## Analysis
# 
# We will check the decomposed graphs for trend, seasonality and residuals by the hour.

# In[17]:


data_hour.plot(figsize=(14,7))


# In[18]:


# Create the decomposed results
plt.figure(figsize=(10, 12))
plt.subplot(311)
decomposed.trend.plot(ax=plt.gca())
plt.title('Trend')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Seasonality')
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca()) 
plt.title('Residuals')
plt.tight_layout()


# In[19]:


# Now we'll look at a smaller period to get a better picture of the seasonality of the data
#data_week = data['2018-03-01':'2018-03-07'].resample('1H').sum()
#print(data_week)


# <div class="alert alert-warning">
# <b>Reviewer's comment</b>
# 
# It would make more sense to take a smaller slice of `decomposed.seasonal`, rather then decomposing the data for one week (how can we talk about any seasonality if we're looking at only one week's worth of data?)
# 
# </div>

# <div class="alert alert-info">
#   Made slices to the seasonal data below for better comparison.
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Alright!
# 
# </div>

# In[20]:


decomposed.seasonal['2018-03-01':'2018-03-07'].plot()


# In[21]:


decomposed.seasonal['2018-08-25':'2018-08-31'].plot()


# In[22]:


decomposed.seasonal


# In[23]:


#decomposed_week = seasonal_decompose(data_week)
#decomposed_week.seasonal.plot()


# In[24]:


#decomposed_week.trend.plot()


# In[25]:


#decomposed_week.resid.plot()


# In[26]:


#data_week['roll_mean'] = data_week['num_orders'].rolling(7).mean()
#data_week.plot()


# When we narrowed down our timeframe to a week, we saw that there is definitely seasonality in our data. The taxi orders peaked usually around midnight.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Alright, you explored the data and made some useful observations!
# 
# </div>

# ## Training

# Train different models with different hyperparameters. The test sample should be 10% of the initial dataset.

# In[27]:


train_all, test = train_test_split(data_hour, shuffle=False, test_size=0.1)
print(train_all.shape, test.shape)


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The data was split without shuffling
# 
# </div>

# In[28]:


train, valid = train_test_split(train_all, shuffle=False, test_size=0.1)
print(train.shape, valid.shape)


# <div class="alert alert-info">
#   Created a validation set for training.
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Awesome!
# 
# </div>

# In[29]:


# run the adfuller test to check for stationarity
df_stationarityTest = adfuller(train_all, autolag='AIC')
print("P-value: ", df_stationarityTest[1])


# This shows that the data is stationary.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Great!
# 
# </div>

# In[30]:


#plt.figure(figsize=(12, 8))

lags_to_check = 50
pacf = plot_pacf(x=train_all, lags=lags_to_check)  # fill in arguments here
pacf.set_figheight(8)
pacf.set_figwidth(12)
plt.xlabel("Lags")
plt.ylabel("PACF")
plt.show()


# The pacf shows that the lags to use is 2.

# In[97]:


lags_to_check = 50
pacf = plot_acf(x=train_all, lags=lags_to_check)  # fill in arguments here
pacf.set_figheight(8)
pacf.set_figwidth(12)
plt.xlabel("Lags")
plt.ylabel("ACF")
plt.show()


# ### LinearRegression

# In[31]:


train_lr = train.copy()
valid_lr = valid.copy()
test_lr = test.copy()
print(test_lr)  # train_lr)


# In[32]:


# Create features function
def create_features(data, max_lag, rolling_mean_size):
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    data['hours'] = data.index.hour
    
    #data['lag_1'] = data['num_orders'].shift(1)
    #data['lag_2'] = data['num_orders'].shift(2)
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['num_orders'].shift(lag)
    
    data['rolling_mean'] = data['num_orders'].shift().rolling(rolling_mean_size).mean()
    data = data.dropna()
    feature_data = data.drop(['num_orders'], axis=1)
    target_data = data['num_orders']
    return feature_data, target_data


# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Good job on feature engineering!
# 
# </div>

# In[71]:


feature_train_lr, target_train_lr = create_features(train_lr, 24, 24)
print(feature_train_lr, target_train_lr)


# In[72]:


feature_valid_lr, target_valid_lr = create_features(valid_lr, 24, 24)
print(feature_valid_lr, target_valid_lr)


# In[73]:


feature_test_lr, target_test_lr = create_features(test_lr, 24, 24)
print(feature_test_lr, target_test_lr)


# In[ ]:


# Features for Test set
# test_lr = test.copy()
# test_lr['year'] = test_lr.index.year
# test_lr['month'] = test_lr.index.month
# test_lr['day'] = test_lr.index.day
# test_lr['dayofweek'] = test_lr.index.dayofweek
# test_lr['hours'] = test_lr.index.hour
# test_lr['lag_1'] = test_lr['num_orders'].shift(1)
# test_lr['lag_2'] = test_lr['num_orders'].shift(2)
# test_lr['rolling_mean'] = test_lr['num_orders'].shift().rolling(7).mean()
# test_lr = test_lr.dropna()


# In[ ]:


# feature_train_lr = train_lr.drop(['num_orders'], axis=1)
# target_train_lr = train_lr['num_orders']
# print(feature_train_lr)
# print(target_train_lr.shape)


# In[ ]:


# Scaling data sets
# scaler = MaxAbsScaler()
# scaler.fit(feature_train_lr)
# feature_train_lr = scaler.transform(feature_train_lr)
# print(feature_train_lr)


# In[ ]:


# scaler = MaxAbsScaler()
# scaler.fit(feature_train_lr)
# feature_valid_lr = scaler.transform(feature_valid_lr)
# print(feature_valid_lr)


# In[ ]:


# scaler = MaxAbsScaler()
# scaler.fit(feature_train_lr)
# feature_test_lr = scaler.transform(feature_test_lr)
# print(feature_test_lr)


# <div class="alert alert-warning">
# <b>Reviewer's comment V2</b>
# 
# Note that if you want to apply scaling, the same scaler fitted on the train set should be applied to validation and test sets as well
# 
# </div>

# In[74]:


model_lr = LinearRegression()
model_lr.fit(feature_train_lr, target_train_lr)
predict_lr = model_lr.predict(feature_valid_lr)
#print(predict_lr)
lr_mse_val = mean_squared_error(target_valid_lr, predict_lr)
print(lr_mse_val**0.5)  # 40.51389934320261 no scaler ; w/ maxabsscaler 52.09222002492444


# In[ ]:


# 40.36210044638575 @ 3 lags; 40.85702350774111 @ 24 roll; 40.623924609297674 @ 5, 24; 40.12970007661318 @ 12,12
# 40.17809996406857 @ 12, 24; 34.135414700079366 @ 24,24


# ### AutoReg

# In[103]:


# scaler = MaxAbsScaler()
# scaler.fit(train)
# train_scaled = scaler.transform(train)
# print(train_scaled)
# valid_scaled = scaler.transform(valid)
# print(len(valid_scaled))


# In[104]:


# df_train_scaled = pd.DataFrame(train_scaled, index=train.index, columns=['num_orders'])
# df_train_scaled


# In[105]:


# df_valid_scaled = pd.DataFrame(valid_scaled, index=valid.index)
# df_valid_scaled


# In[89]:


for i in range(20, 31):
    # find optimal number of lags
    mod = ar_select_order(endog=train, maxlag=i)  # df_train_scaled
    ar_order = mod.ar_lags
    # print(ar_order)
    # create AR model and fit it using the training set
    ar_model = AutoReg(train, lags=ar_order, seasonal=True)  # df_train_scaled
    ar_model = ar_model.fit()

    # make predictions
    start_value = len(train)  # df_train_scaled
    end_value = len(train) + len(valid) - 1  # df_train_scaled
    ar_pred = ar_model.predict(start=start_value, end=end_value, dynamic=True)

    # evaluate model
    ar_mse_value = mean_squared_error(valid, ar_pred) # calculate MSE   # df_train_scaled
    print(i, ':', ar_mse_value**0.5)   # maxlag=24  44.26447543767068 no scale; 


# ### auto_arima

# In[106]:


model = auto_arima(train, seasonal=True, m=7, scoring='mse', error_action='ignore', trace=True,
                      suppress_warnings=True, maxiter=5)  # df_train_scaled max_p=1, max_q=1, 

model.summary()  # Best model:  ARIMA(5,1,3)(2,0,0)[7] intercept Total fit time: 312.934 seconds
# make predictions
#predictions = model.predict(len(df_valid_scaled))

# evaluate the model predictions
#auto_arima_mse = mean_squared_error(df_valid_scaled, predictions)
#print(auto_arima_mse**0.5)  # 58.0112387752594 @ 1; 46.19702866457665 @ 7; failed @ 12


# In[107]:


predictions = model.predict(n_periods = len(valid))
mean_squared_error(valid, predictions)**0.5


# In[81]:


model_aa = auto_arima(train, start_p=5, d=1, start_q=3, max_p=5, max_d=1, max_q=3,
                   seasonal=True, m=7, scoring='mse', error_action='ignore', trace=True,
                   suppress_warnings=True, maxiter=5)

model_aa.summary()


# In[83]:


# make predictions
predictions = model_aa.predict(len(valid))

# evaluate the model predictions
auto_arima_mse = mean_squared_error(valid, predictions)
print(auto_arima_mse**0.5)  # 46.217425948563886


# ### ARIMA

# In[91]:


model_arima = ARIMA(train, order=(5,1,3))
model_arima = model_arima.fit()
# make predictions
start_value = len(train)  # df_train_scaled
end_value = len(train) + len(valid) - 1  # df_train_scaled
#ar_pred = ar_model.predict(start=start_value, end=end_value, dynamic=False)
arima_predict = model_arima.predict(start=start_value, end=end_value, dynamic=False)

print(arima_predict)


# In[94]:


arima_mse = mean_squared_error(valid, arima_predict)
print(arima_mse**0.5)


# ### SARIMAX

# In[99]:


model_sarimax = SARIMAX(train, order=(5,1,3), seasonal_order=(2,0,0,7))
model_sarimax = model_sarimax.fit()
# make predictions
start_value = len(train)  # df_train_scaled
end_value = len(train) + len(valid) - 1  # df_train_scaled

sarimax_predict = model_sarimax.predict(start=start_value, end=end_value, dynamic=False)
print(sarimax_predict)


# In[100]:


sarimax_mse = mean_squared_error(valid, sarimax_predict)
print(sarimax_mse**0.5)


# All the models achieved a RMSE score below 48 on the validation set.

# ## Testing

# The RMSE metric on the test set should not be more than 48.

# In[75]:


feature_train_linreg = pd.concat([feature_train_lr, feature_valid_lr])

target_train_linreg = pd.concat([target_train_lr, target_valid_lr])
feature_train_linreg.shape
#target_train_linreg


# In[76]:


model_lr = LinearRegression()
model_lr.fit(feature_train_lr, target_train_lr)
predict_lr = model_lr.predict(feature_test_lr)
lr_mse_val = mean_squared_error(target_test_lr, predict_lr)
print(lr_mse_val**0.5)  # 


# In[77]:


# 53.9665039683576 @ 3 lags; 52.4117501263748 @ 24 roll; 52.180901032100245 @ 5, 24; 53.438835006494436 @ 12,12
# 52.19143565124224 @ 12,24; 46.32453535485391 @ 24,24


# In[90]:


# AutoReg Testing
# find optimal number of lags
mod = ar_select_order(endog=train_all, maxlag=24)  # train_ar_arima
ar_order = mod.ar_lags

# create AR model and fit it using the training set
ar_model = AutoReg(train_all, lags=ar_order, seasonal=True)  # train_ar_arima
ar_model = ar_model.fit()

# make predictions
start_value = len(train_all)  # train_ar_arima
end_value = len(train_all) + len(test) - 1  # train_ar_arima
ar_pred = ar_model.predict(start=start_value, end=end_value, dynamic=True)

# evaluate model
ar_mse_value = mean_squared_error(test, ar_pred) # calculate MSE
print(ar_mse_value**0.5)  
# 68.92728379869911 train_all


# In[95]:


model_arima = ARIMA(train_all, order=(5,1,3))
model_arima = model_arima.fit()
# make predictions
start_value = len(train_all)  # df_train_scaled
end_value = len(train_all) + len(test) - 1  # df_train_scaled
#ar_pred = ar_model.predict(start=start_value, end=end_value, dynamic=False)
arima_predict = model_arima.predict(start=start_value, end=end_value, dynamic=False)


# In[96]:


arima_mse = mean_squared_error(test, arima_predict)
print(arima_mse**0.5)


# In[101]:


model_sarimax = SARIMAX(train_all, order=(5,1,3), seasonal_order=(2,0,0,7))
model_sarimax = model_sarimax.fit()
# make predictions
start_value = len(train_all)  # df_train_scaled
end_value = len(train_all) + len(test) - 1  # df_train_scaled

sarimax_predict = model_sarimax.predict(start=start_value, end=end_value, dynamic=False)
#print(sarimax_predict)
sarimax_mse = mean_squared_error(test, sarimax_predict)
print(sarimax_mse**0.5)


# In[102]:


# auto_arima Testing
model = auto_arima(train_all, start_p=5, d=1, start_q=3, max_p=5, max_d=1, max_q=3,
                   seasonal=True, m=7, scoring='mse', error_action='ignore', trace=True,
                   suppress_warnings=True, maxiter=5)  

# make predictions
predictions = model.predict(len(test))

# evaluate the model predictions
auto_arima_mse = mean_squared_error(test, predictions)
print(auto_arima_mse**0.5)  # 60.697477945419294


# The only model to achieve the benchmark under a RMSE value of 48 was the LinearRegression model with the number of lags at 24 and a rolling mean of 24.

# <div class="alert alert-danger">
# <s><b>Reviewer's comment</b>
# 
# Great, you tried a couple of different models, but there are some problems here:
#     
# 1. It is not appropriate to tune the model's hyperparameters using the test set. You need to use either a validation set or cross-validation with a [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
#     
# 2. You're calculating MAE instead of RMSE
# 
# 3. You're doing a different problem than what the task asked for due to incorrect resampling :)
#     
# </div>

# <div class="alert alert-warning">
# <b>Reviewer's comment</b>
# 
# It would be nice if you tried doing some feature engineering from the time series yourself (e.g. calendar features, lag features, a shifted rolling mean) and training a linear regression model using those features
# 
# </div>

# <div class="alert alert-info">
# 
# 1. Split the training set into train and validation sets.
# 
# 2. Updated MAE to RMSE, got confused with all the acronyms, heh.
# 
# 3. Added features for the linear regression model.
#     
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Ok, all looks good! :)
# 
# </div>

# ## Conclusion

# Sweet Lift Taxi should use the LinearRegression model with the number of lags at 24 and a rolling mean of 24, even though the data is seasonal on a daily basis. When we looked at the data in a weeks time, we saw that the largest amount of taxi orders at the airport tended to spike at night around midnight.

# <div class="alert alert-danger">
# <s><b>Reviewer's comment</b>
# 
# Please check the results after fixing the problems above
# 
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Well done!
# 
# </div>

# # Review checklist

# - [x]  Jupyter Notebook is open
# - [ ]  The code is error-free
# - [ ]  The cells with the code have been arranged in order of execution
# - [ ]  The data has been downloaded and prepared
# - [ ]  The data has been analyzed
# - [ ]  The model has been trained and hyperparameters have been selected
# - [ ]  The models have been evaluated. Conclusion has been provided
# - [ ] *RMSE* for the test set is not more than 48
