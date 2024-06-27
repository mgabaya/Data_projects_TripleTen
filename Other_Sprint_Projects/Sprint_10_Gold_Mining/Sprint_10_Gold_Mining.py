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
# First of all, thank you for turning in the project! You did a pretty good job overall, but there are some problems that need to be fixed before the project is accepted. Let me know if you have questions!

# # Introduction
# 
# Zyfra is a company in the mining sector that works to improve efficiency and safety for the mining industry.
# 
# We are tasked to predict the amount of gold recovered from gold ore. The data provided is on the extraction and the two-stage purification in a process called Floatation, Purificationm, and Recovery.
# 
# **Objective:**
# In order to predict the amount of gold recovered from ore, we need to predict two values:
# * rougher concentrate recovery [rougher.output.recovery]
# * final concentrate recovery [final.output.recovery]
# * calculate the final sMAPE value
# 

# **Data description**
# 
# Technological process
# * Rougher feed — raw material
# * Rougher additions (or reagent additions) — flotation reagents: Xanthate, Sulphate, Depressant
#     * Xanthate — promoter or flotation activator;
#     * Sulphate — sodium sulphide for this particular process;
#     * Depressant — sodium silicate.
# * Rougher process — flotation
# * Rougher tails — product residues
# * Float banks — flotation unit
# * Cleaner process — purification
# * Rougher Au — rougher gold concentrate
# * Final Au — final gold concentrate
# 
# Parameters of stages
# * air amount — volume of air
# * fluid levels
# * feed size — feed particle size
# * feed rate

# ## Prepare the Data
# 
# To begin, we will first perform the following on the data:
# 
# - Load the necessary libraries
# - Import the file into Dataframes
# - Display the data
# - View the info
# - Describe the data
# - Check for nulls
# - Check the unique values
# - Check for duplicates
# - Fix data types
# 
# 
# * 1.2 Check that recovery is calculated correctly. Using the training set, calculate recovery for the rougher.output.recovery feature. Find the MAE between your calculations and the feature values. Provide findings.
# * 1.3 Analyze the features not available in the test set. What are these parameters? What is their type?
# * 1.4. Perform data preprocessing.

# ### Import Libraries

# In[1]:


# import libraries
import pandas as pd
import numpy as np
from scipy import stats as st
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
from sklearn.dummy import DummyRegressor
from sklearn.metrics import make_scorer


# ### Load Dataframes

# In[2]:


# Load the data into dataframes
try:
    gold_train = pd.read_csv('/datasets/gold_recovery_train.csv')
    gold_test = pd.read_csv('/datasets/gold_recovery_test.csv')
    gold_full = pd.read_csv('/datasets/gold_recovery_full.csv')

except FileNotFoundError:
    gold_train = pd.read_csv('../datasets/gold_recovery_train.csv')
    gold_test = pd.read_csv('../datasets/gold_recovery_test.csv')
    gold_full = pd.read_csv('../datasets/gold_recovery_full.csv')


# ### Display the Data

# In[3]:


# Display dataframes
display(gold_train)


# In[4]:


display(gold_test)


# In[5]:


display(gold_full)


# ### Data Info

# In[6]:


# View the info
print(gold_train.info())


# In[7]:


print(gold_test.info())


# In[8]:


print(gold_full.info())


# ### Describe the Data

# In[9]:


# Describe the data
print(gold_train.describe())


# In[10]:


print(gold_test.describe())


# In[11]:


print(gold_full.describe())


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The data was loaded and inspected!
# 
# </div>

# ### Nulls

# In[12]:


# Check for nulls
print(gold_train.isna().sum())


# In[13]:


print(gold_test.isna().sum())


# In[14]:


print(gold_full.isna().sum())


# We will need to choose what to do with the null values.
# 
# - Fill with mean
# - Fill with median
# - Drop null rows

# In[15]:


gold_train_dropna = gold_train.dropna()  # [~gold_train.isna()].reset_index(drop=True)
gold_train_dropna.info()  # 16860 before, 11017 after


# In[16]:


11017/16860


# We just checked how much of the training set was dropped if we only used rows without any nulls. Seeing that it removed almost 35% of the data, we will choose not to drop the rows. Later we will choose to fill the null values in with the mean or median.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Alright!
# 
# </div>

# ### Duplicates

# In[17]:


# Check for duplicates
print(gold_train[gold_train.duplicated()])


# In[18]:


print(gold_test[gold_test.duplicated()])


# In[19]:


print(gold_full[gold_full.duplicated()])


# There are no duplicates in any of the three dataframes.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Good!
# 
# </div>

# ### Fix data types

# In[20]:


# Fix data types
gold_train['date'] = pd.to_datetime(gold_train['date'], format='%Y-%m-%d')
print(gold_train.info())


# In[21]:


gold_test['date'] = pd.to_datetime(gold_test['date'], format='%Y-%m-%d %H:%M:%S')
print(gold_test.info())


# In[22]:


# display(gold_test)
gold_test['date'].nunique()


# In[23]:


gold_full['date'] = pd.to_datetime(gold_full['date'], format='%Y-%m-%d %H:%M:%S')
print(gold_full.info())


# In[24]:


# gold_full[(gold_full['date'] > '2016-09-01') & (gold_full['date'] < '2018-01-01')]


# In[25]:


gold_full['date'].nunique()


# In[26]:


# Find matching rows in test and full dataframes
full_test_index = gold_full[gold_full['date'].isin(gold_test['date'])].index
print(full_test_index)


# In[27]:


gf = gold_full.loc[full_test_index]
print(gf[gf['final.output.recovery'].isna()])


# In[28]:


gold_test_merged = gold_test.merge(gold_full[['date', 'final.output.recovery', 'rougher.output.recovery']], on='date', how='left')
print(gold_test_merged.info())


# In[29]:


print(gold_test_merged.isna().sum())


# Created a new test dataframe that combined the two target columns from the full set.

# ### MAE Calculation

# In[30]:


#Create function to calculate Recovery.
def recovery(df, C, F, T):
    rec = 100 * ((df[C] * (df[F] - df[T])) / (df[F] * (df[C] - df[T])))
    return rec

#Carry out calculations for rows which don't have NaN in rougher.output.recovery.
vals_pred = recovery(gold_train.dropna(subset=['rougher.output.recovery']), 'rougher.output.concentrate_au', 'rougher.input.feed_au', 'rougher.output.tail_au')
vals_act = gold_train['rougher.output.recovery'].dropna().to_list()

print(f"Mean Absolute Error: {mean_absolute_error(vals_act, vals_pred)}")


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Recovery data was validated successfully!
# 
# </div>

# In[31]:


#vals_pred_final = recovery(gold_train_na, 'final.output.concentrate_au', 'primary_cleaner.output.concentrate_au', 'secondary_cleaner.output.tail_au')
#vals_act_final = gold_train_na['final.output.recovery'].to_list()

#print(f"Mean Absolute Error: {mean_absolute_error(vals_act_final, vals_pred_final)}")


# Analyze the features not available in the test set. What are these parameters? What is their type?
# 
# The following columns appear in the training set, but not in the test set:
# - 9 columns named as "final.output.[concentrate or tail with element]"
# - 8 columns named as "primary_cleaner.output.[concentrate or tail with element]"
# - 4 columns named as "rougher.calculation.[parameter_name]"
# - 9 columns named as "rougher.output.[concentrate or tail with element]"
# - 4 columns named as "secondary_cleaner.output.[tail with element]"
# 
# These 34 missing features are all either outputs or calculation types.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Right! The main point here is that all these are either outputs of the process or calculations based on outputs, and thus using them as features in our models would cause data leakage
# 
# </div>

# In[32]:


not_avail = []
cols = gold_train.columns.tolist()
#print(cols)
for i in cols:
    if ('output' in i) or ('calculation' in i):
        not_avail.append(i)
print(not_avail)


# ### Perform data preprocessing
# 
# We saw from above that the means and medians of each column were close together. For our purposes, we will use the mean of each column to fill the null values. But first, we will drop the rows of the missing values for the target columns, final.output.recovery and rougher.output.recovery 

# In[33]:


# gold_train.iloc[:, 1:]


# In[34]:


final_na = gold_train[gold_train['final.output.recovery'].isna()].index
print(final_na)


# In[35]:


gold_train.drop(final_na, inplace=True)
#gold_train.drop(rougher_na)
gold_train.info()


# In[36]:


rougher_na = gold_train[gold_train['rougher.output.recovery'].isna()].index
print(rougher_na)


# In[37]:


gold_train.drop(rougher_na, inplace=True)
gold_train.info()


# In[38]:


# Preprocess the training set
gold_train_na = gold_train.drop(['date'], axis=1)
gold_train_na.fillna(gold_train_na.mean(), inplace=True)
print(gold_train_na.info())


# In[39]:


print(gold_train_na.describe())


# In[40]:


print(gold_train_na.isna().sum())


# Preprocess the test set by first matching the rows of the test set with the columns final.output.recovery and rougher.output.recovery in the full set. Then drop the nulls in these columns and finally fill the nulls in the other columns with the mean.

# In[41]:


fin_na = gold_test_merged[gold_test_merged['final.output.recovery'].isna()].index
gold_test_merged.drop(fin_na, inplace=True)
rough_na = gold_test_merged[gold_test_merged['rougher.output.recovery'].isna()].index
gold_test_merged.drop(rough_na, inplace=True)


# In[42]:


gold_test_merged.info()


# In[43]:


gold_test_na = gold_test_merged.drop(['date'], axis=1)


# In[44]:


gold_test_na.fillna(gold_test_na.mean(), inplace=True)
print(gold_test_na.info())


# In[45]:


print(gold_test_na.describe())


# In[46]:


print(gold_test_na.isna().sum())


# In[47]:


#gold_full.info()
#gold_full[['final.output.recovery', 'rougher.output.recovery']]


# In[48]:


# Preprocess the full set
#fin_na = gold_full[gold_full['final.output.recovery'].isna()].index
#gold_full.drop(fin_na, inplace=True)
#rough_na = gold_full[gold_full['rougher.output.recovery'].isna()].index
#gold_full.drop(rough_na, inplace=True)


# In[49]:


#gold_full_na = gold_full.drop(['date'], axis=1)
#gold_full_na.fillna(gold_full_na.mean(), inplace=True)
#print(gold_full_na.info())


# <div class="alert alert-danger">
# <S><b>Reviewer's comment</b>
# 
# It is important to drop missing values in target columns before filling the other columns, otherwise we won't be able to accurately estimate the model's performance
# 
# </div>

# <div class="alert alert-info">
#     Dropped the missing values for the target columns
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Great!
# 
# </div>

# ## Analyze the data
# * 2.1. Take note of how the concentrations of metals (Au, Ag, Pb) change depending on the purification stage.
# * 2.2. Compare the feed particle size distributions in the training set and in the test set. If the distributions vary significantly, the model evaluation will be incorrect.
# * 2.3. Consider the total concentrations of all substances at different stages: raw feed, rougher concentrate, and final concentrate. Do you notice any abnormal values in the total distribution? If you do, is it worth removing such values from both samples? Describe the findings and eliminate anomalies. 
# 
# <!--
# - 2.1 you should make hostograms of each metal concentration at each stage and note how each metal changes across stages. 
# - 2.2 you dont need to do a t-test, again it can be histograms of feed size for the training and testing set separately and then noting the differences and if you think they are significant. then lastly for 
# - 2.3, sum up the metal concentrations (sum gold, silver, lead) for each stage and make a histogram of each of these stages sums, then if you see any values for the sum of these that are obvious outliers, go ahead and remove them -->

# ### Purification Concentrations of Metals 
# 
# Take note of how the concentrations of metals (Au, Ag, Pb) change depending on the purification stage.
# 
# #### Au = Gold

# <!--# Au
# print(gold_train_na['primary_cleaner.output.concentrate_au'].mean())
# print(gold_train_na['primary_cleaner.output.concentrate_au'].median())
# print(gold_train_na['primary_cleaner.output.concentrate_au'].isna().sum())
# 
# print(gold_train_na['final.output.concentrate_au'].mean())
# print(gold_train_na['final.output.concentrate_au'].median())
# print(gold_train_na['final.output.concentrate_au'].isna().sum())-->

# In[50]:


gold_train_na['rougher.input.feed_au'].plot(
    kind='hist',
    bins=30,
    alpha=0.7, 
    density=True
)
gold_train_na['rougher.output.concentrate_au'].plot(
    kind='hist',
    bins=30,
    alpha=0.7, 
    density=True
)
gold_train_na['primary_cleaner.output.concentrate_au'].plot(
    kind='hist',
    bins=30,
    title='Concentration of Gold Through Purification Stages',
    figsize=[10,8],
    alpha=0.7, 
    density=True
)
plt.legend(['Before Flotation', 'After Flotation', 'After Primary', 'After Secondary'], title='Stage')
plt.xlabel('Au Concentrate')
gold_train_na['final.output.concentrate_au'].plot(
    kind='hist',
    bins=30,
    alpha=0.7, 
    density=True
)
plt.show()


# The overall concentration of gold from the flotation input stage through the secondary cleaner process goes up with a large portion of the bell curve moving up by about 10 from each stage to the next.

# In[51]:


#final.output.concentrate_au
#final.output.concentrate_ag
#final.output.concentrate_pb


# #### Ag = Silver

# In[52]:


gold_train_na['rougher.input.feed_ag'].plot(
    kind='hist',
    bins=30,
    alpha=0.7, 
    density=True
)
gold_train_na['rougher.output.concentrate_ag'].plot(
    kind='hist',
    bins=30,
    alpha=0.7, 
    density=True
)
gold_train_na['primary_cleaner.output.concentrate_ag'].plot(
    kind='hist',
    bins=30,
    title='Concentration of Silver Through Purification Stages',
    figsize=[10,8],
    alpha=0.7, 
    density=True
)
plt.legend(['Before Flotation', 'After Flotation', 'After Primary', 'After Secondary'], title='Stage')
plt.xlabel('Ag Concentrate')
gold_train_na['final.output.concentrate_ag'].plot(
    kind='hist',
    bins=30,
    alpha=0.7, 
    density=True
)
plt.show()


# From the input stage to the rougher output stage, the silver concentration initially goes up. After the primary stage, though, the overall concentrations of silver decrease from the primary cleaner process and the secondary cleaner stages.

# #### Pb = Lead

# In[53]:


gold_train_na['rougher.input.feed_pb'].plot(
    kind='hist',
    bins=30,
    alpha=0.7, 
    density=True
)
gold_train_na['rougher.output.concentrate_pb'].plot(
    kind='hist',
    bins=30,
    alpha=0.7, 
    density=True
)
gold_train_na['primary_cleaner.output.concentrate_pb'].plot(
    kind='hist',
    bins=30,
    title='Concentration of Lead Through Purification Stages',
    figsize=[10,8],
    alpha=0.7, 
    density=True
)
plt.legend(['Before Flotation', 'After Flotation', 'After Primary', 'After Secondary'], title='Stage')
plt.xlabel('Pb Concentrate')
gold_train_na['final.output.concentrate_pb'].plot(
    kind='hist',
    bins=30,
    alpha=0.7, 
    density=True
)
plt.show()


# The distribution of concentration of lead increases from the input stage until the output of the primary cleaner process. From here to the secondary cleaner process, the concentration stays the same, the main difference being the bell curve gets more concentrated about the mean.

# <div class="alert alert-danger">
# <s><b>Reviewer's comment</b>
# 
# Please add the first two stages: `rougher.input.feed.au`, etc and `rougher.output.concentrate_au`, etc.
# 
# </div>

# <div class="alert alert-info">
#   Added rougher.input.feed_ and rougher.output.concentrate_ stages to the histograms and updated summaries beneath each plot.
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Awesome!
# 
# </div>

# ### Feed Particle Size
# 
# Compare the feed particle size distributions in the training set and in the test set. If the distributions vary significantly, the model evaluation will be incorrect.

# In[54]:


# Train vs Test Rougher Input Size
gold_train_na['rougher.input.feed_size'].plot(
    kind='hist',
    bins=30,
    title='Input Size in the Rougher Stage',
    figsize=[10,8],
    alpha=0.7, 
    density=True
)
plt.legend(['Train', 'Test'], title='Rougher Input Size')
plt.xlabel('Au')
gold_test_na['rougher.input.feed_size'].plot(
    kind='hist',
    bins=30,
    alpha=0.7, 
    density=True
)
plt.show()


# In[55]:


# Train vs Test Input Size
gold_train_na['primary_cleaner.input.feed_size'].plot(
    kind='hist',
    bins=30,
    title='Input Size in the Primary Cleaner Stage',
    figsize=[10,8],
    alpha=0.7, 
    density=True
)
plt.legend(['Train', 'Test'], title='Primary Cleaner Input Size')
plt.xlabel('Au')
gold_test_na['primary_cleaner.input.feed_size'].plot(
    kind='hist',
    bins=30,
    alpha=0.7, 
    density=True
)
plt.show()


# In[56]:


# Histogram of of games by genre
# color='genre', barmode="overlay", , hover_data=['type']) # color='make'
#pt_histo = px.histogram(gold_train['primary_cleaner.input.feed_size'], x='primary_cleaner.input.feed_size', nbins=20, histnorm='probability') 
#pt_histo.update_layout(
#    title_text='Input size', # title of plot
#    xaxis_title_text='Size', # xaxis label
#    yaxis_title_text='Frequency', # yaxis label
#)
#pt_histo2 = px.histogram(gold_test['primary_cleaner.input.feed_size'], x='primary_cleaner.input.feed_size', nbins=20, histnorm='probability')
#pt_histo.show()
#pt_histo2.show()


# Both the previous histograms show the input size at the rougher and primary cleaner stages. By overlaying the normalized graphs on each other, we see that the distributions are similar enough that we can use the test set for modeling.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Very well! Feed size distributions were compared successfully
# 
# </div>

# ### Total Concentrations
# 
# Consider the total concentrations of all substances at different stages: raw feed, rougher concentrate, and final concentrate. Do you notice any abnormal values in the total distribution? If you do, is it worth removing such values from both samples? Describe the findings and eliminate anomalies.

# In[70]:


# 2.3 sum up the metal concentrations (sum gold, silver, lead) 
# for each stage and make a histogram of each of these stages sums, 
# then if you see any values for the sum of these that are obvious outliers, 
# go ahead and remove them
sets = []
stages = ['rougher.input.feed_', 'rougher.output.concentrate_', 'primary_cleaner.output.concentrate_', 'final.output.concentrate_'] # 
metals = ['au', 'ag', 'pb']
for i in stages:
    rows = []
    for j in metals:
        rows.append(i+j)
    sets.append(rows)

#print(sets[0])        
#gold_train['concentrations'] = gold_train[cols].sum(axis=1)
concentrations = [[],[],[],[]]
concentrations[0].append(gold_train_na[sets[0]].sum(axis=1))  # rougher.input.feed
concentrations[1].append(gold_train_na[sets[1]].sum(axis=1))  # rougher.output.concentrate
concentrations[2].append(gold_train_na[sets[2]].sum(axis=1))  # primary_cleaner.output.concentrate
concentrations[3].append(gold_train_na[sets[3]].sum(axis=1))  # final.output.concentrate
#concentrations = gold_train_na[cols].sum(axis=1)

print(concentrations)


# <div class="alert alert-danger">
# <s><b>Reviewer's comment</b>
# 
# Note that you need to separately calculate total concentration at each stage (e.g. for initial concentrations `rougher.input.feed_au + rougher.input.feed_ag + rougher.input.feed_pb`). Also don't forget about `primary_cleaner.output.concentrate_*`
# 
# </div>

# <div class="alert alert-info">
#   Calculated concentrations seperately and added primary cleaner stage. 
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Well done!
# 
# </div>

# In[58]:


#print(gold_train.loc[0, cols].sum())
#print(gold_train.loc[16859, cols].sum())


# In[59]:


concentrations[0][0].plot(
    kind='hist',
    bins=30,
    title='Rougher Input Size',
    figsize=[8,6] 
)
plt.xlabel('Concentrations Size')
plt.show()


# In[60]:


concentrations[1][0].plot(
    kind='hist',
    bins=30,
    title='Rougher Output Size',
    figsize=[8,6] 
)
plt.xlabel('Concentrations Size')
plt.show()


# In[61]:


concentrations[2][0].plot(
    kind='hist',
    bins=30,
    title='Primary Cleaner Output Size',
    figsize=[8,6] 
)
plt.xlabel('Concentrations Size')
plt.show()


# In[62]:


concentrations[3][0].plot(
    kind='hist',
    bins=30,
    title='Final Output Size',
    figsize=[8,6] 
)
plt.xlabel('Concentrations Size')
plt.show()


# In each of the stages, we see outliers mostly on the lower end that may affect our models later. So, next we will calculate the upper and lower range to find our cutoffs for outliers.

# In[63]:


def outlier_index(vals):
    q1 = np.percentile(vals, 25)
    q3 = np.percentile(vals, 75)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr
    idx = vals[(vals < lower) | (vals > upper)].index
    return idx


# In[64]:


oi1 = outlier_index(concentrations[0][0])
oi2 = outlier_index(concentrations[1][0])
oi3 = outlier_index(concentrations[2][0])
oi4 = outlier_index(concentrations[3][0])
#print(oi1)


# In[65]:


#gold_train_na_out = gold_train_na.drop(oi1)
#gold_train_na_out.info()


# In[66]:


#oi4 = outlier_index(concentrations[3][0])
#print(oi4)
#gold_train_na_out = gold_train_na.drop(oi4)
#gold_train_na_out.info()


# In[67]:


out_ind = [oi1, oi2, oi3, oi4]
unq_ind = sorted(list(set(x for l in out_ind for x in l)))       
print(len(unq_ind))


# In[68]:


#gold_train_na.info()
gold_train_na_out = gold_train_na.drop(unq_ind)
gold_train_na_out.info()


# <!--14149 before, 13370 after, redone 12086-->

# This final dataframe has all the nulls filled in and outliers removed so we can begin modeling.

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Outliers were removed successfully!
# 
# </div>

# ## Build the model
# * 3.1. Write a function to calculate the final sMAPE value.
# * 3.2. Train different models. Evaluate them using cross-validation. Pick the best model and test it using the test sample. Provide findings.
# 
# Use these formulas for evaluation metrics:
# 

# ### sMAPE and Final sMAPE Functions
# sMAPE = 1/N * sum( |yi - yih| / (|yi| + |yih|)/2 * 100%
# 
# Final sMAPE = sMAPE(rougher) + 75% * sMAPE(final)

# In[71]:


# function for sMAPE
def smape(target, predictions):
    sigma = 0
    for i in range(len(predictions)):
        sigma += abs(target[i] - predictions[i]) / (abs(target[i]) + abs(predictions[i])) / 2
    return 100 / len(predictions) * sigma


# In[72]:


# function for Final sMAPE
def final_smape(rougher_smape, final_smape):
    total = (0.25*rougher_smape) + (0.75*final_smape)
    return total


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The functions for SMAPE calculation are correct
# 
# </div>

# In[73]:


# Testing sMAPE function
test_target =      [-1, 10, 10, 11, 1, 10, 1, 10, 11, 1, 10, 10, 0, 1]
test_predictions = [1, 10, 10, 10, 1, 11, -1, 10, 10, 1, 10, 11, 10, 1]
smape(test_target, test_predictions)


# In[74]:


# Testing Final sMAPE function
final_smape(4.111, 1.22222)


# #### Train Models

# In[75]:


gold_train_na_out.reset_index(drop=True, inplace=True)


# In[76]:


# Feature and target for each region
features = gold_train_na_out.drop(not_avail, axis=1)
target = gold_train_na_out['final.output.recovery']  
target_rougher = gold_train_na_out['rougher.output.recovery']


# In[77]:


features.info()


# In[78]:


target_list = target.tolist()
print(target_list)


# In[79]:


target_rougher


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Columns that would cause data leakage were dropped
# 
# </div>

# In[ ]:


# 75:25 split for train and valid sets
#features_train, features_valid, target_train, target_valid = train_test_split(
#    features, target, test_size=0.25, random_state=12345
#)
#print(features_valid.shape)
#print(target_valid.shape)


# In[ ]:


#features_train_rougher, features_valid_rougher, target_train_rougher, target_valid_rougher = train_test_split(
#    features, target_rougher, test_size=0.25, random_state=12345
#)
#print(features_train_rougher.shape)
#print(target_valid_rougher.shape)


# In[ ]:


#model = LinearRegression()
#model.fit(features_train, target_train)
#predictions_valid = model.predict(features_valid)
#print(predictions_valid)
#tv_list = target_valid.tolist()
#print(tv_list)
#smape_valid = smape(tv_list, predictions_valid)
#print(smape_valid)


# <div class="alert alert-danger">
# <s><b>Reviewer's comment</b>
# 
# There's no need to split the data into train and validation in this project, as you're supposed to use cross-validation to compare different models
# 
# </div>

# <div class="alert alert-info">
#   Commented the splits out. 
#     
#   New cross-validation function below.
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Alright!
# 
# </div>

# In[69]:


'''
def model_optimizer(model_name, dep=1, estim=10, strg=0):
    best_model = None
    best_est, best_depth, best_smape = 0, 0, 100000
    depth_or_strength = 'Depth'
    estimators = estim+1
    max_dep = dep+1
    if strg != 0:
        max_dep = strg
        depth_or_strength = 'Strength'
    for est in range(10, estimators, 10):
        for depth in range(1, max_dep):
            # if statements for DecTree, RandFor, LogReg
            if model_name == RandomForestRegressor:
                model = model_name(random_state=12345, max_depth=depth, n_estimators=est)
            elif model_name == LinearRegression:
                model = LinearRegression()
            else:
                model = DecisionTreeRegressor(random_state=12345, max_depth=depth)
            model.fit(features_train, target_train)
            predictions_valid = model.predict(features_valid)
            smape_valid = smape(target_valid, predictions_valid)
            #mae_valid = mean_absolute_error(target_valid, predictions_valid)
            #print('MAE', mae_valid)
            if smape_valid < best_smape:
                best_est = est
                best_depth = depth
                #best_valid_f1 = f1_valid
                #best_roc_auc = roc_auc_valid
                best_smape = smape_valid
                best_model = model
    print('Best', depth_or_strength, ':', best_depth)
    print('Best n-estimators', best_est)
    #print('Best F1 of Validation Set:', best_valid_f1)
    #print('AUC-ROC of Validation Set:', best_roc_auc)
    print('Best sMAPE', best_smape)
    print(best_model)
'''


# <div class="alert alert-danger">
# <s><b>Reviewer's comment</b>
# 
# 1. It is problematic to use global variables like `features_train` which can be easily overwritten in some cell
# 2. Please use cross-validation using our target metric SMAPE instead of validation set with MAE
# 
# </div>

# <div class="alert alert-info">
#     Commented out previous optimizer function and removed the features_train and like variables.
#     Moved optimizer below, still getting negative values
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Well, yeah, if you set `greater_is_better=False`, you're supposed to get negative values :)
#     
# Why is that? In scikit-learn there is a convention that the 'score' is always maximized (by GridSearchCV and similar objects). Maximizing the negation of an error function is the same thing as minimizing it, thus setting `greater_is_better=False` simply flips the sign
# 
# </div>

# In[90]:


# Declare make_scorer
smape_scorer = make_scorer(smape)  #, greater_is_better=False)


# In[88]:


#model = DecisionTreeRegressor(max_depth=16, random_state=12345)
#model = RandomForestRegressor(random_state=12345, max_depth=16, n_estimators=50)
#model = LinearRegression()


# In[91]:


#scores_try = cross_val_score(model, features, target_list, scoring=smape_scorer, cv=5)
#print(scores_try.mean())


# In[110]:


def optimizer(model_name, feat, targ, dep=1, estim=10):
    best_model = None
    best_est, best_depth, best_score = 0, 0, 100000
    max_dep = dep+1
    estimators = estim+1
    targ_list = targ.tolist()
    for est in range(10, estimators, 10):
        for depth in range(1, max_dep):
            # if statements for DecTree, RandFor, LogReg
            if model_name == RandomForestRegressor:
                model = model_name(random_state=12345, max_depth=depth, n_estimators=est)
            elif model_name == DecisionTreeRegressor:
                model = DecisionTreeRegressor(random_state=12345, max_depth=depth)
            else:
                model = LinearRegression()
            scores = cross_val_score(model, features, targ_list, scoring=smape_scorer, cv=5)
            #print(scores.mean(), est, depth)
            if scores.mean() < best_score:
                best_est = est
                best_depth = depth
                best_score = scores.mean()
                best_model = model
    print('Best depth', best_depth)
    print('Best n-estimators', best_est)
    print('Best score', best_score)
    print(best_model)


# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# All looks good!
# 
# </div>

# In[115]:


# RandomForestRegressor for final.output.recovery
optimizer(RandomForestRegressor, features, target, 13, 40)  # 4, 40, 2.0603438870787016


# In[114]:


# DecisionTreeRegressor for final.output.recovery
optimizer(DecisionTreeRegressor, features, target, dep=16)  # 1, 2.2019506255921546


# In[104]:


# LinearRegression for final.output.recovery
optimizer(LinearRegression, features, target)


# In[113]:


# RandomForestRegressor for rougher.output.recovery
optimizer(RandomForestRegressor, features, target_rougher, 13, 40)  # 5, 40, 1.5723987484400497


# In[112]:


# DecisionTreeRegressor for rougher.output.recovery
optimizer(DecisionTreeRegressor, features, target_rougher, dep=16)  # 4, 1.706894280099346


# In[119]:


# LinearRegression for rougher.output.recovery
optimizer(LinearRegression, features, target_rougher)  


# The best sMAPE value for final.output.recovery was with the RandomForestRegressor model with a depth of 4 and n_estimators of 40.
# 
# The best sMAPE value for rougher.output.recovery was with the RandomForestRegressor model with a depth of 5 and n_estimators of 40.
# 

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Excellent!
# 
# </div>

# ## Cross-Validation on Test Set

# In[ ]:


#def cross_val_final_score(model):
#    scores = cross_val_score(model, features, target, cv=5)
#    final_score = sum(scores) / len(scores)
#    return final_score


# <div class="alert alert-danger">
# <s><b>Reviewer's comment</b>
# 
# 1. It is problematic to use global variables like `features` and `target`
#     
# 2. Right now you're using unidentified metric in cross-validation (it's not even clear whether greater values of this metric are better). Please use SMAPE metric instead. You can create a scoring object using [make_scorer](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring) and then pass it as the `scoring` parameter to `cross_val_score`
# 
# </div>

# <div class="alert alert-info">
#     Commented out to remove the cross_val_final_score function.
#     Updated the cross_val_score to use the smape function in the make_scorer metric.
#     
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Great!
# 
# </div>

# In[ ]:


#print('DTR', cross_val_final_score(DecisionTreeRegressor(max_depth=12, random_state=12345)))


# In[ ]:


#print('RFR', cross_val_final_score(RandomForestRegressor(random_state=12345, max_depth=13, n_estimators=30)))


# In[ ]:


#print('LR', cross_val_final_score(LinearRegression()))


# The linear regression model had the highest cross-validation score, so this will be used for the test set.

# In[123]:


# Use the test set to match the predictions from the test set
#target_full = gold_full_na.loc[full_test_index]
#print(target_full.info())


# In[122]:


# This dataframe is the combination of the test set with the 
# final.output.recovery and rougher.output.recovery columns from the full set
# previously matched by their dates.
gold_test_na.info()


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Test set targets were iudentified correctly
# 
# </div>

# <div class="alert alert-info">
#   gold_test_na now has the two target columns matched from the full set.
# </div>

# In[116]:


# create the features variables for modeling
gold_test_na.reset_index(drop=True, inplace=True)
g_test_features = gold_test_na.drop(['final.output.recovery', 'rougher.output.recovery'], axis=1)
g_test_features.info()


# In[133]:


# Final.output.recovery
model_final = RandomForestRegressor(max_depth=4, n_estimators=40, random_state=12345) 
model_final.fit(features, target) 
predictions_test = model_final.predict(g_test_features)
print(predictions_test)


# In[134]:


# sMAPE for final.output.recovery
fin_smape = smape(gold_test_na['final.output.recovery'].tolist(), predictions_test)
print(fin_smape)


# In[136]:


# rougher.output.recovery
model_rougher = RandomForestRegressor(max_depth=5, n_estimators=40, random_state=12345) 
model_rougher.fit(features, target_rougher) 
predictions_test_rougher = model_rougher.predict(g_test_features)
print(predictions_test_rougher)


# In[138]:


# sMAPE for rougher.output.recovery
rougher_smape = smape(gold_test_na['rougher.output.recovery'].tolist(), predictions_test_rougher)
print(rougher_smape)


# In[140]:


# Final sMape
final_smape(rougher_smape, fin_smape)


# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# The final model was evaluated on the test set
# 
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The final model was evaluated on the test set using SMAPE metric
# 
# </div>

# <div class="alert alert-danger">
# <s><b>Reviewer's comment</b>
# 
# Please add a comparison with a simple baseline, e.g. a constant model always predicting mean or median of train set targets (you can use [DummyRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html))
# 
# </div>

# <div class="alert alert-info">
#   Added the dummy regressor below.
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Awesome!
# 
# </div>

# ### Dummy Regressor Comparison

# In[142]:


# final.output.recovery
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(features, target)


dummy_predict = dummy_regr.predict(g_test_features)
#print(len(dummy_predict))
#print(len(gold_test_na['final.output.recovery'].tolist()))
dummy_final_smape = smape(gold_test_na['final.output.recovery'].tolist(), dummy_predict)
print(dummy_final_smape)


# In[143]:


# rougher.output.recovery
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(features, target_rougher)


dummy_predict = dummy_regr.predict(g_test_features)
#print(len(dummy_predict))
#print(len(gold_test_na['rougher.output.recovery'].tolist()))
dummy_rougher_smape = smape(gold_test_na['rougher.output.recovery'].tolist(), dummy_predict)
print(dummy_rougher_smape)


# In[144]:


# Final sMape
final_smape(dummy_rougher_smape, dummy_final_smape)


# Both our models were able to produce sMAPE values smaller than the DummyRegressor model as well as with the Final sMAPE value. Albeit, the values calculated in the our model compared to the dummy are relatively close.

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Alright!
# 
# </div>

# ## Conclusion
# 

# Using the data sets provided, we found the following:
# - There are features in the training and full datasets that don't appear in the test set.
# - As we examined the concentrations of gold, silver and lead go through the stages of before flotation, after flotation, after the primary cleaner and after the secondary cleaner, their behaviors are as follows:
#     - The concentrations of gold (Au) went up from the beginning to end stages.
#     - The concentrations of silver (Ag) went up from before to after flotation, but decreased in the subsequent stages.
#     - The concentrations of lead (Pb) increased until the after primary cleaning stage and then stayed about the same between the primary and secondary cleaning stages.
# 
# 
# - The Final sMAPE value we got using the RandomForestRegressor model is about 2.313, which was smaller compared to the DummyRegressor value that was more than 0.1 above.

# <div class="alert alert-warning">
# <b>Reviewer's comment</b>
# 
# Don't forget to change conclusions if necessary :)
# 
# </div>

# <div class="alert alert-info">
#   Updated the conclusions. Thanks for reviewing! 
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# You're welcome! Good job!
#     
# The project is now accepted. Keep up the good work on the next sprint!
# 
# </div>
