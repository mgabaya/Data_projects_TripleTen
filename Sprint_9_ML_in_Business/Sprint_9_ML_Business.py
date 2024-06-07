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
# First of all, thank you for turning in the project! You did an excellent job! The project is accepted. Good luck on the next sprint!

# # Introduction

# We are tasked by the OilyGiant mining company to find the best place for new wells. We are given data on oil samples from three regions. Parameters of each oil well in the region are already known. 
# 
# ## Objective
# 
# Steps to choose the location:
# - Collect the oil well parameters in the selected region: oil quality and volume of reserves;
# - Build a model for predicting the volume of reserves in the new wells;
# - Pick the oil wells with the highest estimated values;
# - Pick the region with the highest total profit for the selected oil wells.
# 

# Conditions:
# - Only linear regression is suitable for model training (the rest are not sufficiently predictable).
# - When exploring the region, a study of 500 points is carried with picking the best 200 points for the profit calculation.
# - The budget for development of 200 oil wells is 100 USD million.
# - One barrel of raw materials brings 4.5 USD of revenue The revenue from one unit of product is 4,500 dollars (volume of reserves is in thousand barrels).
# - After the risk evaluation, keep only the regions with the risk of losses lower than 2.5%. From the ones that fit the criteria, the region with the highest average profit should be selected.
# 
# The data is synthetic: contract details and well characteristics are not disclosed.
# 

# ### Preprocess the Data
# To begin, we will first perform the following on the data:
# 
# - Load the necessary libraries
# - Import the file into Dataframes
# - Display the data
# - View the info
# - Check for nulls
# - Check the unique values
# - Check for duplicates

# In[1]:


# import libraries
import pandas as pd
import numpy as np
from scipy import stats as st
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score, r2_score
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
#from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import random
from matplotlib import pyplot as plt
import plotly.express as px


# In[2]:


# Load the data into dataframes
try:
    geo_data_0 = pd.read_csv('/datasets/geo_data_0.csv')
    geo_data_1 = pd.read_csv('/datasets/geo_data_1.csv')
    geo_data_2 = pd.read_csv('/datasets/geo_data_2.csv')

except FileNotFoundError:
    geo_data_0 = pd.read_csv('../datasets/geo_data_0.csv')
    geo_data_1 = pd.read_csv('../datasets/geo_data_1.csv')
    geo_data_2 = pd.read_csv('../datasets/geo_data_2.csv')


# Geological exploration data for the three regions are stored in files:
# - geo_data_0.csv. download dataset
# - geo_data_1.csv. download dataset
# - geo_data_2.csv. download dataset
# 
# * id — unique oil well identifier
# * f0, f1, f2 — three features of points (their specific meaning is unimportant, but the features themselves are significant)
# * product — volume of reserves in the oil well (thousand barrels).

# In[3]:


# Display the data
display(geo_data_0)


# In[4]:


display(geo_data_1)


# In[5]:


display(geo_data_2)


# In[6]:


# Inspect the data
print(geo_data_0.info())
print(geo_data_1.info())
print(geo_data_2.info())


# In[7]:


# Describe the data
print(geo_data_0.describe())
print(geo_data_1.describe())
print(geo_data_2.describe())


# In[8]:


# Check for Nulls
print(geo_data_0.isna().sum(),
geo_data_1.isna().sum(),
geo_data_2.isna().sum())


# In[9]:


# Chceck id for Uniqueness
print(geo_data_0['id'].nunique())
print(geo_data_1['id'].nunique())
print(geo_data_2['id'].nunique())


# In[10]:


# Check for Duplicates
print(geo_data_0[geo_data_0.duplicated()])
print(geo_data_1[geo_data_0.duplicated()])
print(geo_data_2[geo_data_0.duplicated()])


# No duplicate rows, but now we'll check the id column.

# In[11]:


# Check if any id's are duplicated
print(geo_data_0[geo_data_0['id'].duplicated()])


# In[12]:


print(geo_data_1[geo_data_1['id'].duplicated()])
print(geo_data_2[geo_data_2['id'].duplicated()])


# In[13]:


# Confirming if a single id makes sense to have multiple rows
print(geo_data_0[geo_data_0['id'] == 'HZww2'])


# Even though we have duplicate id's, we will keep them as is since it most likely means the same oil well was measured twice at different times.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Alright!
# 
# </div>

# ## Split the Data
# 
# On setting our feature and target variables, we will drop the id column since we don't want it to be used for modeling, and the product column will be our target.
# 
# We will split the data into a training set and validation set at a ratio of 75:25

# In[14]:


# Feature and target for each region
# geo_data_0 region
features_0 = geo_data_0.drop(['id', 'product'], axis=1)
target_0 = geo_data_0['product']


# In[15]:


# geo_data_1 region
features_1 = geo_data_1.drop(['id', 'product'], axis=1)
target_1 = geo_data_1['product']


# In[16]:


# geo_data_2 region
features_2 = geo_data_2.drop(['id', 'product'], axis=1)
target_2 = geo_data_2['product']


# In[17]:


# Verifying that the target and features were assigned correctly
print(features_0.shape)
print(target_0.shape)


# In[18]:


# 75:25 split for geo_data_0 region
features_train_0, features_valid_0, target_train_0, target_valid_0 = train_test_split(
    features_0, target_0, test_size=0.25, random_state=12345
)
print(features_valid_0.shape)


# In[19]:


# 75:25 split for geo_data_1 region
features_train_1, features_valid_1, target_train_1, target_valid_1 = train_test_split(
    features_1, target_1, test_size=0.25, random_state=12345
)
print(target_train_1.shape)


# In[20]:


# 75:25 split for geo_data_2 region
features_train_2, features_valid_2, target_train_2, target_valid_2 = train_test_split(
    features_2, target_2, test_size=0.25, random_state=12345
)
print(features_valid_2.shape)


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The data for each region was split into train and validation sets
# 
# </div>

# ## Model Testing
# 
# Train and test the model for each region:
# * 2.1 Split the data into a training set and validation set at a ratio of 75:25.
# * 2.2. Train the model and make predictions for the validation set.
# * 2.3. Save the predictions and correct answers for the validation set.
# * 2.4. Print the average volume of predicted reserves and model RMSE.
# *  2.5. Analyze the results.

# In[21]:


# LinearRegression
# Region 0
print('Region 0')
model_0 = LinearRegression() #, )n_jobs=-1, normalize=True
model_0.fit(features_train_0, target_train_0) # train model on training set
predictions_valid_0 = model_0.predict(features_valid_0) # get model predictions on validation set
rmse_0 = mean_squared_error(target_valid_0, predictions_valid_0)**0.5
print('RMSE:', rmse_0)
r2_0 = r2_score(target_valid_0, predictions_valid_0)
print('R-Squared:', r2_0)
predictions_valid_0 = pd.Series(predictions_valid_0)
print('Predictions:\n', predictions_valid_0)
print('Correct Answers:\n', target_valid_0)


# In[22]:


# Region 1
print('Region 1')
model_1 = LinearRegression() 
model_1.fit(features_train_1, target_train_1) # train model on training set
predictions_valid_1 = model_1.predict(features_valid_1) 
rmse_1 = mean_squared_error(target_valid_1, predictions_valid_1)**0.5
print('RMSE:', rmse_1)
r2_1 = r2_score(target_valid_1, predictions_valid_1)
print('R-Squared', r2_1)
predictions_valid_1 = pd.Series(predictions_valid_1)
print(predictions_valid_1)
print(target_valid_1)


# In[23]:


# Region 2
print('Region 2')
model_2 = LinearRegression() 
model_2.fit(features_train_2, target_train_2) 
predictions_valid_2 = model_2.predict(features_valid_2) 
rmse_2 = mean_squared_error(target_valid_2, predictions_valid_2)**0.5
print('RMSE:', rmse_2)
r2_2 = r2_score(target_valid_2, predictions_valid_2)
print('R-Squared', r2_2)
predictions_valid_2 = pd.Series(predictions_valid_2)
print(predictions_valid_2)
print(target_valid_2)


# In[24]:


# Print the average volume of predicted reserves and model RMSE
print('Region 0 Average Volume:', predictions_valid_0.mean())
print('Region 0 RMSE:', rmse_0)
print('Region 0 R-Squared', r2_0)
print()
print('Region 1 Average Volume:', predictions_valid_1.mean())
print('Region 1 RMSE:', rmse_1)
print('Region 1 R-Squared', r2_1)
print()
print('Region 2 Average Volume:', predictions_valid_2.mean())
print('Region 2 RMSE:', rmse_2)
print('Region 2 R-Squared', r2_2)


# *Analyze the results:*
# 
# To compare the models in each region, we used linear regression for our predictions and calculated the RMSE and R-Squared values for each.
# 
# - Region 0 had a high average volume of predicted reserves, but scored poorly in the RMSE and R-squared values.   
# - Region 1 had the lowest average volume of predicted reserves, but the strongest RMSE and R-Squared values by far.
# - Region 2 had the highest average volume of predicted reserves of all the regions, but had the worst scores of RMSE and R-squared.
# 
# The high variability shown by the RMSE value and the low R-squared values in Region 0 and 2 might indicate the the high end outliers might be affecting the scores.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The models were trained and evaluated correctly
# 
# </div>

# ## Prepare for Profit Calculation
# 
# *  3.1. Store all key values for calculations in separate variables.
# *  3.2. Calculate the volume of reserves sufficient for developing a new well without losses. Compare the obtained value with the average volume of reserves in each region.
# *  3.3. Provide the findings about the preparation for profit calculation step.

# In[25]:


# Will need to match each target reserve with it's prediction
print(target_valid_0)
print(predictions_valid_0)


# In[26]:


# Create and store the target values and predictions in a new variable for Region 0
reset_tv0 = target_valid_0.reset_index(drop=True) # reset index for target values
print(reset_tv0)


# In[27]:


# Match each prediction to their target value and create a variable to store in
target_predictions_0 = pd.DataFrame({'target': reset_tv0, 'predictions': predictions_valid_0})
print(target_predictions_0)


# In[28]:


print(target_valid_1)
print(predictions_valid_1)


# In[29]:


# Create and store the target values and predictions in a new variable for Region 1
reset_tv1 = target_valid_1.reset_index(drop=True)
print(reset_tv1)
target_predictions_1 = pd.DataFrame({'target': reset_tv1, 'predictions': predictions_valid_1})
print(target_predictions_1)


# In[30]:


print(target_valid_2)
print(predictions_valid_2)


# In[31]:


# Create and store the target values and predictions in a new variable for Region 2
reset_tv2 = target_valid_2.reset_index(drop=True)
print(reset_tv2)
target_predictions_2 = pd.DataFrame({'target': reset_tv2, 'predictions': predictions_valid_2})
print(target_predictions_2)


# In[32]:


# Calculate the volume of reserves sufficient for developing a new well without losses
100000000/4500 # Budget / revenue per unit


# A region would need about 22,223 units in reserve to operate without a loss.

# In[33]:


# Average volume of reserves in each region
print(target_predictions_0['predictions'].mean())
print(target_predictions_1['predictions'].mean())
print(target_predictions_2['predictions'].mean())


# In[34]:


print(22223/92.59)
print(22223/68.73)
print(22223/94.97)


# The averages in regions 0 and 2 are higher than region 1, so it would require less wells at their respective averages to break even the budget of $100 million. Region 2 would require almost one hundred more wells at the average to break even compared to the other two regions. Next, we will look at the top 200 wells in each region and see how profitable they can be.

# <div class="alert alert-warning">
# <b>Reviewer's comment</b>
# 
# So, looking at the results we can say that none of the regions have the average volume to break even if we choose the wells randomly (as our budget only allows development of 200 wells)
# 
# </div>

# ## Profit Function
# 
# Write a function to calculate profit from a set of selected oil wells and model predictions:
# *  4.1. Pick the wells with the highest values of predictions. 
# *  4.2. Summarize the target volume of reserves in accordance with these predictions
# *  4.3. Provide findings: suggest a region for oil wells' development and justify the choice. Calculate the profit for the obtained volume of reserves.

# In[35]:


# Profit function
def oil_profit(target_predicts, count):
    top_oil = target_predicts.sort_values(by=['predictions'], ascending=False).head(count)
    target_sum = top_oil['target'].sum()
    revenue = target_sum * 4500  # revenue per unit
    profit = revenue - 100000000  # budget for 200 wells
    return profit


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Profit calculation function is correct
# 
# </div>

# In[36]:


profit_0 = oil_profit(target_predictions_0, 200)
print(profit_0)


# In[37]:


profit_1 = oil_profit(target_predictions_1, 200)
print(profit_1)


# In[38]:


profit_2 = oil_profit(target_predictions_2, 200)
print(profit_2)


# While considering the top 200 wells of each region in terms of predicted reserves, Region 0 has a potential profit of \\$33.2 million, the highest of the regions. Region 2 has the next lowest potential profit of \\$27.1 million. Not much behind region 2 is region 1 at \\$24.15 million.
# 
# The regions that could be considered are all of them at this point. But, there is a caveat with regions 0 and 2. Since their models' predictions were not strong, it increases the risk of a loss. 
# 
# In the next section, we will look at possible profits if any 200 wells were chosen at random in each region through bootstrapping. This can further help narrow down the optimal region.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Makes sense!
# 
# </div>

# ## Risks and Profit
# 
# Calculate risks and profit for each region:
# *      5.1. Use the bootstrapping technique with 1000 samples to find the distribution of profit.
# *      5.2. Find average profit, 95% confidence interval and risk of losses. Loss is negative profit, calculate it as a probability and then express as a percentage.
# *      5.3. Provide findings: suggest a region for development of oil wells and justify the choice.

# In[39]:


# Bootstapping function
def profit_bootstrap(target_predictions):
    state = np.random.RandomState(12345)

    values = []

    for i in range(1000):
        subsample_target = target_predictions.sample(n=500, replace=True, random_state=state)
        subsample_profit = oil_profit(subsample_target, 200)
        values.append(subsample_profit)

    values = pd.Series(values)
    lower = values.quantile(0.025) 
    upper = values.quantile(0.975) 
    profit_mean = values.mean()
    #df_bootstrap = pd.DataFrame({'values': values, 'lower': lower, 'upper': upper, 'profit_mean': profit_mean})
    return values, lower, upper, profit_mean


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Bootstrapping is done correctly, all needed statistics are calculated
# 
# </div>

# In[40]:


# Region 0
values_0, lower_0, upper_0, profit_mean_0 = profit_bootstrap(target_predictions_0)
print(values_0, '\nLower:', lower_0, '\nUpper:', upper_0, '\nAverage:', profit_mean_0)


# In[41]:


#values_0.plot(
#    kind='hist',
#    bins=30,
#    title='Region 0 Profits',
#    figsize=(10,6)
#)
#plt.xlabel('$USD')
#plt.show()
fig_histogram_0 = px.histogram(values_0, nbins=30) # , color='model_year')
fig_histogram_0.update_layout(
    title_text='Region 0 Bootstrapped Profits', # title of plot
    xaxis_title_text='$USD Millions', # xaxis label
    yaxis_title_text='Count', # yaxis label
)
fig_histogram_0.show()


# In[42]:


# risk of losses Region 0
losses_0 = values_0[values_0 < 0]
print(losses_0)


# For region 0, the probability of loss is 0.069, or 6.9%, which is higher than our risk tolerance of 2.5%. We also see that the bottom 2.5th percentile is already at a loss.

# In[43]:


# Region 1
values_1, lower_1, upper_1, profit_mean_1 = profit_bootstrap(target_predictions_1)
print(values_1, '\nLower:', lower_1, '\nUpper:', upper_1, '\nAverage:', profit_mean_1)


# <!--values_1.plot(
# #    kind='hist',
# #    bins=30,
# #    title='Region 1 Profits',
# #    figsize=(10,6)
# #)
# #plt.xlabel('$USD')
# #plt.show()
# confidence_interval = st.t.interval(
#     alpha=.95, df=len(values_1)-1, loc=values_1.mean(), scale=values_1.sem()
# )
# print(confidence_interval)-->

# In[44]:


fig_histogram_1 = px.histogram(values_1, nbins=30) # , color='model_year')
fig_histogram_1.update_layout(
    title_text='Region 1 Bootstrapped Profits', # title of plot
    xaxis_title_text='$USD Millions', # xaxis label
    yaxis_title_text='Count', # yaxis label
)
fig_histogram_1.show()


# In[45]:


# risk of losses Region 1
losses_1 = values_1[values_1 < 0]
print(len(losses_1))


# For region 1, the probability of loss is 0.015, or 1.5%, which passes our risk tolerance of 2.5%. This is also shown in the histogram above with low frequency on the left side values.

# In[46]:


# Region 2
values_2, lower_2, upper_2, profit_mean_2 = profit_bootstrap(target_predictions_2)
print(values_2, '\nLower:', lower_2, '\nUpper:', upper_2, '\nAverage:', profit_mean_2)


# In[47]:


fig_histogram_2 = px.histogram(values_2, nbins=30) # , color='model_year')
fig_histogram_2.update_layout(
    title_text='Region 2 Bootstrapped Profits', # title of plot
    xaxis_title_text='$USD Millions', # xaxis label
    yaxis_title_text='Count', # yaxis label
)
fig_histogram_2.show()


# In[48]:


# risk of losses Region 2
losses_2 = values_2[values_2 < 0]
print(len(losses_2))


# For region 2, the probability of loss is 0.076, or 7.6%, which is higher than our risk tolerance of 2.5%. We also see that the bottom 2.5th percentile is already at a loss and the left tail of the histogram makes up a larger portion of the distribution. 

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Very well! Good idea to plot the bootstrapped profit distributions
# 
# </div>

# <!--# 95% Confidence Interval
# confidence_interval = st.t.interval(
#     alpha=.95, df=len(values)-1, loc=values.mean(), scale=values.sem()
# )
# print(confidence_interval)-->

# ## Conclusion

# <!--Here’s what the reviewers will look at when reviewing your project:
# - How did you prepare the data for training?
# - Have you followed all the steps of the instructions?
# - Have you taken into account all the business conditions?
# - What are your findings about the task study?
# - Have you applied the Bootstrapping technique correctly?
# - Have you suggested the best region for well development? Is the choice justified?
# - Did you avoid code duplication?
# - Have you kept to the project structure and kept the code neat?-->

# Through our investigation, we found the following for OilyGiant about which region to develop new wells in:
# 
# - Region 1 is the region in which our model predicted the oil reserves with strong correlation.
# - Region 1 had the lowest risk of losses, and the only one to meet the criteria of losses less than 2.5%. Additionally, the region's 95% confidence interval was the only interval to have the lower end still in the profit zone.
# - Regions 0 and 2 behaved similarly through the study. Although they had the higher average profit of their top 200 predicted oil wells, when choosing any 200 oil wells, their risk of losses was much higher than our 2.5% loss criteria. Furthermore, these two regions' models did not have strong predictions, which can possibly increase the chances of losses.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Conclusions make sense, region choice is correct and justified
# 
# </div>

# In[ ]:




