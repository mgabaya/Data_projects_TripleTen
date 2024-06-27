#!/usr/bin/env python
# coding: utf-8

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
# First of all, thank you for turning in the project! You did an excellent job! The project is accepted. Keep up the good work on the next sprint!

# Rusty Bargain used car sales service is developing an app to attract new customers. In that app, you can quickly find out the market value of your car. You have access to historical data: technical specifications, trim versions, and prices. You need to build the model to determine the value. 
# 
# Rusty Bargain is interested in:
# 
# - the quality of the prediction;
# - the speed of the prediction;
# - the time required for training

# ## Data preparation
# 
# To begin, we will first perform the following on the data:
# 
# - Load the necessary libraries
# - Import the file into Dataframes
# - Display the data
# - View the info
# - Adjust column names
# - Describe the data
# - Check for nulls and fill in, if possible
# - Check the unique values
# - Check for duplicates
# - Adjust data set for unnecessary columns
# - Fix any data types
# - Encode the data
# - Scale the data

# In[1]:


# Load the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MaxAbsScaler
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error,roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# In[2]:


# Import the file into Dataframes
try:
    df_car = pd.read_csv('/datasets/car_data.csv', sep=',')

except FileNotFoundError:
    df_car = pd.read_csv('../datasets/car_data.csv', sep=',')


# **Features**
# 
# - DateCrawled — date profile was downloaded from the database
# - VehicleType — vehicle body type
# - RegistrationYear — vehicle registration year
# - Gearbox — gearbox type
# - Power — power (hp)
# - Model — vehicle model
# - Mileage — mileage (measured in km due to dataset's regional specifics)
# - RegistrationMonth — vehicle registration month
# - FuelType — fuel type
# - Brand — vehicle brand
# - NotRepaired — vehicle repaired or not
# - DateCreated — date of profile creation
# - NumberOfPictures — number of vehicle pictures
# - PostalCode — postal code of profile owner (user)
# - LastSeen — date of the last activity of the user
# 
# **Target**
# 
# - Price — price (Euro)

# In[3]:


display(df_car)


# In[4]:


df_car.info()


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The data was loaded and inspected
# 
# </div>

# In[5]:


# Change column names
df_car.columns = df_car.columns.str.lower()
df_car.info()


# In[6]:


# Descriptive Statistics
df_car.describe()


# In[7]:


#df_car[df_car['price'] == 0]  # 10772 rows
len(df_car[(df_car['power'] > 10000)])  # (df_car['power'] < 20) | 


# In[8]:


df_car['registrationyear'].describe()


# In[9]:


df_car['registrationyear'].value_counts(ascending=False, dropna=False)


# In[10]:


no_year = df_car[(df_car['registrationyear'] < 1900) | (df_car['registrationyear'] > 2024)]
display(no_year) #.index)


# The 171 values for the registrationyear column don't make sense, so it would be best to remove them.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Makes sense!
# 
# </div>

# In[11]:


df_car.drop(no_year.index, inplace=True)
df_car.reset_index(drop=True, inplace=True)
df_car.info()


# ### Nulls
# 
# We will need to decide what to do with the nulls.

# Our strategy to fill in nulls:
# - nan to 'other' for multiple columns
# - combine petrol and gasoline as the same category.

# In[12]:


df_car.isna().sum()


# In[13]:


df_car[df_car['model'].isna()]


# In[14]:


# Check Duplicates
print(df_car[df_car.duplicated()])


# We will choose not to delete the duplicates because it is possible that the same car is on sale multiple times.

# <div class="alert alert-warning">
# <b>Reviewer's comment</b>
# 
# In that case `datecreated` should be different, right?
# 
# </div>

# In[15]:


# Uniqueness
print(df_car['vehicletype'].unique())
print(df_car['gearbox'].unique())
print(df_car['model'].unique())
print(df_car['model'].nunique())
print(df_car['fueltype'].unique())
print(df_car['brand'].unique())
print(df_car['notrepaired'].unique())
print(df_car['numberofpictures'].unique())


# In[16]:


pd.Series(df_car['model'].unique()).sort_values()


# In[17]:


# Drop unnecesary columns since they shouldn't help predict the price


# In[18]:


col_drop = ['datecrawled', 'datecreated', 'numberofpictures', 'lastseen', 'postalcode']
df_car.drop(col_drop, axis=1, inplace=True)
df_car.info()


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Makes sense!
# 
# </div>

# In[19]:


df_car['vehicletype'].value_counts(ascending=False, dropna=False)


# In[20]:


df_car[['vehicletype', 'gearbox', 'model', 'fueltype', 'notrepaired']] = \
    df_car[['vehicletype', 'gearbox', 'model', 'fueltype', 'notrepaired']].fillna('other')


# In[21]:


df_car.isna().sum()


# In[22]:


df_car['fueltype'] = df_car['fueltype'].replace('gasoline', 'petrol')
df_car['fueltype'].value_counts(ascending=False, dropna=False)


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Missing values were dealt with reasonably
# 
# </div>

# ### Encoding and Scaling
# 
# - OHE for Linear Regression and XGBoost
# - Ordinal/Label for CatBoost, LGBM, DTR, and RFR
# 
# Since some models require different encoding, we will make a data set with One Hot Encoding (OHE) and another one with ordinal encoding for the categorical columns.
# 
# After splitting the data into its features and targets, we will scale the data for the OHE data set.

# In[23]:


# Encoding with OHE
ohe_cols = ['vehicletype', 'gearbox', 'model', 'fueltype', 'brand', 'notrepaired']


# In[24]:


get_ipython().run_cell_magic('timeit', '', 'df_car\n')


# In[25]:


df_car_ohe = pd.get_dummies(df_car, columns=ohe_cols, drop_first=True)
df_car_ohe.info()


# In[26]:


for i in df_car_ohe.columns:
    print(i)


# In[27]:


df_car_ohe.loc[20, 'price': 'mileage']


# In[28]:


features_ohe = df_car_ohe.drop(['price'], axis=1)
target_ohe = df_car_ohe['price']
print(features_ohe.shape)
print(target_ohe.shape)


# In[29]:


features_train_ohe, features_test_ohe, target_train_ohe, target_test_ohe = train_test_split(
    features_ohe, target_ohe, test_size=0.3, random_state=12345
)


# In[30]:


# Scaling OHE data set
scaler = MaxAbsScaler()
scaler.fit(features_train_ohe)
features_train = scaler.transform(features_train_ohe)
print(features_train_ohe)
features_test = scaler.transform(features_test_ohe)
print(features_test_ohe)


# In[31]:


# Encoding with OrdinalEncoder
enc = OrdinalEncoder()
df_ord = df_car.copy()
df_ord[['vehicletype', 'gearbox', 'model', 'fueltype', 'brand', 'notrepaired']] = \
    enc.fit_transform(df_car[['vehicletype', 'gearbox', 'model', 'fueltype', 'brand', 'notrepaired']])
df_ord.sample(10)


# In[32]:


features_ord = df_ord.drop(['price'], axis=1)
target_ord = df_ord['price']
#print(features)
#print(target)
features_train_ord, features_test_ord, target_train_ord, target_test_ord = train_test_split(
    features_ord, target_ord, test_size=0.3, random_state=12345
)
features_train_ord.shape


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Caterogical features were encoded, the data was split into train and test, scaling was applied correctly
# 
# </div>

# ## Model training
# 
# Train different models with various hyperparameters (You should make at least two different models, but more is better. Remember, various implementations of gradient boosting don't count as different models.) The main point of this step is to compare gradient boosting methods with random forest, decision tree, and linear regression.
# 
# - Use the RMSE metric to evaluate the models.
# - Linear regression is not very good for hyperparameter tuning, but it is perfect for doing a sanity check of other methods. If gradient boosting performs worse than linear regression, something definitely went wrong.
# - On your own, work with the LightGBM library and use its tools to build gradient boosting models.
# - Ideally, your project should include linear regression for a sanity check, a tree-based algorithm with hyperparameter tuning (preferably, random forrest), LightGBM with hyperparameter tuning (try a couple of sets), and CatBoost and XGBoost with hyperparameter tuning (optional).
# - Take note of the encoding of categorical features for simple algorithms. LightGBM and CatBoost have their implementation, but XGBoost requires OHE.
# - You can use a special command to find the cell code runtime in Jupyter Notebook. Find that command.
# - Since the training of a gradient boosting model can take a long time, change only a few model parameters.

# ### Sanity Check
# 
# Sanity check using LinearRegression

# In[34]:


get_ipython().run_cell_magic('timeit', '-n 3 -r 2', "# LinearRegression\nmodel_lr = LinearRegression()\nmodel_lr.fit(features_train_ohe, target_train_ohe)\npredictions_lr = model_lr.predict(features_test_ohe)\nrmse_lr = mean_squared_error(target_test_ohe, predictions_lr)**0.5\nprint('RMSE:', rmse_lr)\n# RMSE: 2949.2\n")


# ### DecisionTreeRegressor and RandomForestRegressor

# In[35]:


def model_optimizer(model, parameters, scoring_metric='neg_mean_squared_error', cv=5):

    grid_search = GridSearchCV(model, parameters, cv=cv, scoring=scoring_metric)
    grid_search.fit(features_train_ord, target_train_ord)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_params, best_score


# In[36]:


dt_params = {'max_depth': list(range(9, 18)),
             'random_state': [12345]
            }
dt_model = DecisionTreeRegressor()

best_params, best_score = model_optimizer(dt_model, dt_params, scoring_metric='neg_mean_squared_error')
print(best_params)
print(best_score)
# {'max_depth': 13, 'random_state': 12345}
# -4403431.3


# In[38]:


dt_params_rfr = {'max_depth': list(range(14, 17)),
                 'n_estimators': list(range(40, 61, 10)),
                 'random_state': [12345]
                }
dt_model_rfr = RandomForestRegressor()

best_params_rfr, best_score_rfr = model_optimizer(dt_model_rfr, dt_params_rfr, scoring_metric='neg_mean_squared_error')
print(best_params_rfr)
print(best_score_rfr)
# {'max_depth': 15, 'n_estimators': 50, 'random_state': 12345}
# -3233139.9298187154


# In[42]:


get_ipython().run_cell_magic('timeit', '-n 3 -r 2', "# DecisionTreeRegressor\nmodel_dtr = DecisionTreeRegressor(random_state=12345, max_depth=13)\nmodel_dtr.fit(features_train_ord, target_train_ord)\npredictions_dtr = model_dtr.predict(features_test_ord) \nresult_dtr = mean_squared_error(target_test_ord, predictions_dtr)**0.5\nprint('RMSE:', result_dtr) # with ohe: 2051.7; with ord: 2066.2\n")


# In[43]:


get_ipython().run_cell_magic('timeit', '-n 2 -r 2', "# RandomForestRegressor\nmodel_rfr = RandomForestRegressor(random_state=12345, n_estimators=60, max_depth=16) # initialize model constructor with parameters random_state=12345 and n_estimators=est\nmodel_rfr.fit(features_train_ord, target_train_ord) # train model on training set\npredictions_rfr = model_rfr.predict(features_test_ord) # get model predictions on validation set\nresult_rfr = mean_squared_error(target_test_ord, predictions_rfr)**0.5\nprint('RMSE:', result_rfr) # with ohe: 2039.2; with ord: 1777.8\n")


# ### XGBoost
# Requires OHE

# In[56]:


get_ipython().run_cell_magic('timeit', '-n 1 -r 1', '# XGBRegressor\nxgbr = XGBRegressor(objective=\'reg:squarederror\')\nxgbr.fit(features_train_ohe, target_train_ohe)\n \npredictions_xgbr = xgbr.predict(features_test_ohe)\nrmse_xgbr = mean_squared_error(target_test_ohe, predictions_xgbr)**0.5\nprint("RMSE:", rmse_xgbr)  # RMSE: 1814.5\n')


# ### LightGBM and CatBoost
# 
# Don't need OHE, but Label/Ordinal Code columns

# In[ ]:


df_ord.sample(8)


# In[61]:


get_ipython().run_cell_magic('timeit', '-n 2 -r 2', '# Catboost\nmodel_cbr = CatBoostRegressor(loss_function="RMSE", iterations=200, learning_rate=0.5, random_seed=12345) # learning_rate=0.4, \nmodel_cbr.fit(features_train_ord, target_train_ord, verbose=10)\npred_cbr = model_cbr.predict(features_test_ord)\nmse = mean_squared_error(target_test_ord, pred_cbr)**0.5\nprint(mse)  # 1901.3; 1839.1 @ 100 iter; 1810.4 @ 150 iter; 1795.8 @ 200 iter, 0.5 l_rate\n')


# In[59]:


get_ipython().run_cell_magic('timeit', '-n 3 -r 2', "# LGBMRegressor\nmodel_lgbm = LGBMRegressor(metric='rmse', n_estimators=200)\nmodel_lgbm.fit(features_train_ord, target_train_ord)\npred_lgbm = model_lgbm.predict(features_test_ord)\nmse_lgbm = mean_squared_error(target_test_ord, pred_lgbm)**0.5\nprint(mse_lgbm)  # 1863.1; 1830.3 @ 150 n_est; 1809 @ 200 n_est\n")


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Great, you tried a few different models, did some hyperparameter tuning using cross-validation and compared the final models using the test set
# 
# </div>

# ## Model analysis
# 
# Analyze the speed and quality of the models.

# RandomForestRegressor is accurate with an RMSE of 1777.8, but slow (41.7 s ± 972 ms).
# 
# XGBRegressor comes in second in terms of the RMSE with , but is slowest model by far (7min 37s ± 1.59 s).
# 
# In order of the RMSE values, Catboost (10.8 s ± 25.7 ms), LGBMRegressor (8.5 s ± 2.01 s), DecisionTreeRegressor (1.05 s ± 280 µs per loop), and lastly, LinearRegression (19.7 s ± 54 ms) round out the rest of the accuracies, but were much faster than the the RandomForest and XGBoost. All the models were able to beat our sanity check with the LinearRegression model in terms of RMSE. However, the speed at which RandomForest and XGBoost models performed was much slower than the LinearRegression's baseline speed.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Very good, the analysis takes into account both speed and quality of predictions of the models
# 
# </div>

# ## Conclusion

# The model that Rusty Bargain should use for quality, speed, and time is CatBoostRegressor with 200 iterations and a learning rate of 0.5. This model did well overall in the three qualities that Rusty Bargain required. It came in a close second in terms of the RMSE value behind the RandomForest model, which makes it's quality high. It also ranked as the third fastest model to train and make it's predictions, whereas the RandomForest model was not able to beat our benchmark time set by the LinearRegression model.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Alright!
# 
# </div>

# # Checklist

# Type 'x' to check. Then press Shift+Enter.

# - [x]  Jupyter Notebook is open
# - [ ]  Code is error free
# - [ ]  The cells with the code have been arranged in order of execution
# - [ ]  The data has been downloaded and prepared
# - [ ]  The models have been trained
# - [ ]  The analysis of speed and quality of the models has been performed

# In[ ]:




