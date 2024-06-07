#!/usr/bin/env python
# coding: utf-8

# **Review**
# 
# Hello Mark!
# 
# I'm happy to review your code today.
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
# First of all, thank you for turning in the project! You did an excellent job! The code is accepted. Based on your best model's ROC-AUC on the test set, you've got 4.5 SP. Congratulations!

# <div class="alert alert-success">
#     Hi Mark! I have approved your initial report for the final project. Great work! Just wanted to add some helpful pointers:
#     <br/><br/>
#     I believe the correct data files are in this directory: 
#     /datasets/final_provider/internet.csv
#     <br/><br/>
# If end date is empty, assume that the customer is still with the company (no churn)
#     <br/><br/>
#     When merging dataframes make sure that you do not miss any customers since some ID’s may not exist in certain tables. You can treat these missing cases as that customer not signing up for the specific service
# <br/><br/>
#     Your target variable should be Churn / No Churn, this will be a binary classification task
# <br/><br/>
#     Make sure that the validation set is used to tune hyperparameters, good idea to use gridsearch
# <br/><br/>
#     There does indeed exist a class imbalance in the data, make sure this is accounted for
# <br/><br/>
#     Careful with data leakage, make sure to not include features that may be collinear (i.e if using one-hot encoding for categorical variables, drop one of the columns) or reference the target variable in some way (see topic on data leakage)
# <br/><br/>
#     Good choice of models to try, make sure to also try the gradient boosting type classifiers
# <br/><br/>
#     Very good EDA plan
# <br/><br/>
#     The goal of this project is to understand why customers churned or stopped their service. And we can use machine learning to identify features of the data that had high predictive power when determining which customers will churn. And if we know that, what can we do as a company to prevent ongoing churn (this part is more of a discussion question for your end
# <br/><br/>
# -Yervand, Data Science Tutor
# </div>

# # Introduction
# 
# INTERCONNECT, a telecom company, has collected the following 4 Datasets:
# 
# Contract - Billing info and length of service<br> 
# Personal - Customer's personal info <br>
# Internet - Breakdown of user's internet usage<br>
# Phone - If user has multiple phone lines
# 
# <!-- 
# df_contract
# df_personal 
# df_internet
# df_phone 
# -->

# ## Objective

# Ultimately, using the data from Interconnect, we aim to forecast the churn of their customers. Doing so can help the company offer special promotions to prevent these customers from canceling their plans.
# 
# In the following section, we will do a preliminary look at the data as well as cleaning it up. This will give us insight on how to proceed with our Project Plan.

# ## Prepare the Data
# 
# - Load the necessary libraries
# - Import the file into Dataframes
# - Display the data
# - View the info
# - Check for nulls
# - Check for duplicates
# - Check the unique values
# - Convert column names and types
# - Merge data sets and investigate it
#    

# In[1]:


# Load the necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt
import plotly.express as px
from xgboost import XGBClassifier
import seaborn as sns


# In[2]:


# Import the files into Dataframes
try:
    df_contract = pd.read_csv('/datasets/final_provider/contract.csv', sep=',')
except FileNotFoundError:
    df_contract = pd.read_csv('../datasets/final_provider/contract.csv', sep=',')
    
display(df_contract)


# In[3]:


try:
    df_personal = pd.read_csv('/datasets/final_provider/personal.csv', sep=',')
except FileNotFoundError:
    df_personal = pd.read_csv('../datasets/final_provider/personal.csv', sep=',')
    
display(df_personal)


# In[4]:


try:
    df_internet = pd.read_csv('/datasets/final_provider/internet.csv', sep=',')
except FileNotFoundError:
    df_internet = pd.read_csv('../datasets/final_provider/internet.csv', sep=',')
    
display(df_internet)


# In[5]:


try:
    df_phone = pd.read_csv('/datasets/final_provider/phone.csv', sep=',')
except FileNotFoundError:
    df_phone = pd.read_csv('../datasets/final_provider/phone.csv', sep=',')

display(df_phone)


# In[6]:


# Info on each data set
print('CONTRACT')
print(df_contract.info())
print('PERSONAL')
print(df_personal.info())
print('INTERNET')
print(df_internet.info())
print('PHONE')
print(df_phone.info())


# In[7]:


# Check for nulls
print('CONTRACT')
print(df_contract.isna().sum())
print('PERSONAL')
print(df_personal.isna().sum())
print('INTERNET')
print(df_internet.isna().sum())
print('PHONE')
print(df_phone.isna().sum())


# In[8]:


# Check for duplicates
print(df_contract[df_contract.duplicated()])
print(df_personal[df_personal.duplicated()])
print(df_internet[df_internet.duplicated()])
print(df_phone[df_phone.duplicated()])


# There are no nulls and duplicates in each of the original data sets.

# In[9]:


# Checking uniqueness of each customer
print(df_contract['customerID'].nunique())
print(df_personal['customerID'].nunique())
print(df_internet['customerID'].nunique())
print(df_phone['customerID'].nunique())


# Contract, personal and phone data sets all have unique customerID's, but internet and phone have fewer than contract and personal. 

# In[10]:


# Unique values in contact set
for i in range(1, 8):
    print(df_contract.columns.values[i], ':', df_contract[df_contract.columns.values[i]].sort_values().unique())


# In[11]:


display(df_contract[df_contract['TotalCharges'] == ' '])
display(df_contract[df_contract['BeginDate'] == '2020-02-01'])


# In[12]:


df_contract[df_contract['Type'] == 'Two year']


# In the contract data set, it looks like there's only 4 months that users ended their service mixed with the string "No" as the values. We will have to handle this column in a later section. 
# 
# Also, there are empty values for total_charges that relate to the February 2020 date. We will fill these in at a later section where we adjust any column types. 
# 
# Other than that, all other column's unique values don't need to handle any implicit duplicates.

# In[13]:


# Unique values in personal
for i in range(1, 5):
    print(df_personal.columns.values[i], ':', df_personal[df_personal.columns.values[i]].sort_values().unique())


# We will assume that for SeniorCitizen that 0 is not a senior citizen and 1 is they are.

# In[14]:


# Unique values in internet
for i in range(1, 8):
    print(df_internet.columns.values[i], ':', df_internet[df_internet.columns.values[i]].sort_values().unique())


# No abnormal values from the internet set.

# In[15]:


# Unique values in phone
df_phone['MultipleLines'].unique()


# No abnormal values from the phone set.

# ### Adjust Column Names and Types

# In[16]:


# Renaming columns
df_contract.columns = df_contract.columns.str.lower()
df_contract = df_contract.rename(columns={
    'customerid': 'customer_id', 
    'begindate': 'begin_date',
    'enddate': 'end_date',
    'paperlessbilling': 'paperless_billing', 
    'paymentmethod': 'payment_method',
    'monthlycharges': 'monthly_charges',
    'totalcharges': 'total_charges'
})
df_contract.info()


# In[17]:


df_personal.columns = df_personal.columns.str.lower()
df_personal = df_personal.rename(columns={
    'customerid': 'customer_id', 
    'seniorcitizen': 'senior_citizen'
})
df_personal.info()


# In[18]:


df_phone = df_phone.rename(columns={
    'customerID': 'customer_id', 
    'MultipleLines': 'multiple_lines'
})
df_phone.info()


# In[19]:


df_internet = df_internet.rename(columns={
    'customerID': 'customer_id', 
    'InternetService': 'internet_service',
    'OnlineSecurity': 'online_security',
    'OnlineBackup': 'online_backup', 
    'DeviceProtection': 'device_protection',
    'TechSupport': 'tech_support',
    'StreamingTV': 'streaming_tv',
    'StreamingMovies': 'streaming_movies'
})
df_internet.info()


# In[20]:


# Fix column types


# In[21]:


# Convert to datetime
df_contract['begin_date'] = pd.to_datetime(df_contract['begin_date'], format='%Y-%m-%d')
#df_contract['end_date'] = pd.to_datetime(df_contract['end_date'], format='%Y-%m-%d')


# We found out earlier that the end_date column is a mix between string and datetime values, so using to_datetime won't work. We will use feature engineering to explore what to do with the end_date column.

# In[22]:


# Change total_charges column to float
df_contract['total_charges'] = pd.to_numeric(df_contract['total_charges'], errors='coerce')
print(df_contract.info())


# In[23]:


# Check the null values for total_charges column
df_contract[df_contract['total_charges'].isna()] # .index


# In[24]:


# Fill null values with their monthly charge since February 2020 is the last month of the collected data
df_contract['total_charges'] = df_contract['total_charges'].fillna(df_contract['monthly_charges'])
df_contract.iloc[488, :]  # Check one of the null rows to make sure it was filled correctly


# Since most of the other columns are categorical, we will leave them as objects that will later be encoded.

# ### Merge data sets

# In[25]:


# Combine datasets with customerID
df_combine = df_contract.merge(df_personal, on='customer_id')  # , how='outer'
df_combine.info()


# In[26]:


# Combine datasets with customerID, again
df_merged = df_combine.merge(df_phone, on='customer_id', how='outer')
df_merged.info()


# In[27]:


df_merged  # [df_merged.isna()]


# In[28]:


# df_merged.isna().sum()


# In[29]:


# df_merged[df_merged['multiple_lines'].isna()]


# In[30]:


df_all = df_merged.merge(df_internet, on='customer_id', how='outer')
df_all


# In[31]:


df_all.isna().sum()


# We see null values in our all merged data set. Since

# In[32]:


# df_all[df_all['online_security'].isna()]


# In[33]:


df_all.info()


# In[34]:


# Fill the nulls
df_all['multiple_lines'] = df_all['multiple_lines'].fillna('No')


# In[35]:


df_all['multiple_lines'].unique()


# In[36]:


df_all['internet_service'] = df_all['internet_service'].fillna('None')


# In[37]:


df_all['internet_service'].value_counts(dropna=False) # .unique()


# In[38]:


df_all[df_all.columns.values[19]]


# In[39]:


# df_all['multiple_lines'].value_counts(dropna=False)
for i in range(14, 20):
    df_all[df_all.columns.values[i]] = df_all[df_all.columns.values[i]].fillna('No')
    print(df_all[df_all.columns.values[i]].value_counts())


# There are null values in our merged data set since the internet and phone data did not match all the customer_id's. We will investigate the class imbalances next in order to decide what to do with these null values.

# In[40]:


# Check class imbalances
for col in df_all.columns:
    print(col)
    print(df_all[col].value_counts(dropna=False), '\n')


# Classes with imbalances: Type, PaperlessBilling, PaymentMethod, SeniorCitizen, Dependents, and all the columns related to the internet.
# 
# These will be noted when exploring our models to see if class balancing should occur.

# In[41]:


# Descriptive statistics for the merged data
print(df_all.describe())


# In[42]:


print(df_all.describe(exclude=[np.number]))


# ## Project Plan

# Data Preprocessing (Above sections):
# 1. Load the necessary libraries
# 2. Import the files into Dataframes
#     - Use try/except blocks
# 3. Display the dataframes
# 4. View the info of each dataframe
# 5. Check for nulls for each dataframe
# 6. Check for row duplicates for each dataframe
# 7. Check the unique values for each dataframe
#     - Check individual columns for any implicit duplicates or other strange values
# 8. Adjust column names
#     - Lower case and standard underscore for multiple words
# 9. Convert column types, if necessary
# 10. Merge data sets
#     - Investigate the merged data set
#     - Check class imbalances
#     - Descriptive statistics for the data
# 11. Replace the null values of the merged data set

# After Preprocessing:
# 
# 12. Determine which columns should be removed/won't help model predictions from the merged data<br/>
# 13. Feature engineering
#     - Create a length_of_account column from begin_date to end_date
#     - Create an Churn / No Churn column from EndDate to signify if a customer is active or not
# 14. Choose a target column, most likely the Churn column
# 15. Encode the data
#     - OHE for categorical data
#     - MaxAbsScalar for quantitative
# 16. Create data visualizations 
#     - histograms for numerical columns
#     - scatter plots 
#     - boxplots with outliers
#     - pairplots
# 17. Use a dummy classifier to set a baseline
# 18. Split the data into train, valid, and test sets
#     - Use a 60/20/20 split
# 19. Test different models for classification 
#     - DecisionTreeClassifier
#     - RandomForrestClassifier
#     - LogisticRegression
#     - Gradient Boosting
# 20. Achieve AUC-ROC score >= 0.75
#     - Hypertune parameters to get better scores by using gridsearch
#     - Test to see if balancing some of the classes will get better scores

# ## Feature Engineering

# In[43]:


df_all.info()


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Good job on data preprocessing! Dataframes were merged correctly
# 
# </div>

# In[44]:


df_all.head()


# In[45]:


df_all['end_date'].value_counts(dropna=False)


# In[46]:


def churn(end_date):
    """
    Function decides if a customer is churned or not
    0 is the customer did not churn
    1 is the customer churned
    """
    if end_date == 'No':
        return 0
    else: 
        return 1
    
churn(df_all.loc[2, 'end_date'])


# In[47]:


df_all['churn'] = df_all['end_date'].apply(churn)
df_all.head()


# In[48]:


df_all['churn'].value_counts()


# There is definitely a class imbalance with the churn rate.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Good point!
# 
# </div>

# ## Encode the Data
# 
# - OHE for categorical data
# - MaxAbsScalar for quantitative

# In[49]:


data = df_all.drop(['customer_id', 'begin_date', 'end_date'], axis=1)
display(data)
data.info()


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Both dates were dropped, so there shouldn't be any data leakage
# 
# </div>

# In[50]:


# OHE
data_ohe = pd.get_dummies(data, drop_first=True)
display(data_ohe)
data_ohe.info()


# The data has been one-hot-encoded on the categorical columns. The two float columns will have a scalar applied to it after the data is split.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Categorical features were encoded
# 
# </div>

# ## Visualizations
# 
# - histograms for numerical columns
# - scatter plots 
# - boxplots with outliers
# - pairplots

# In[51]:


# Histogram of Monthly Charges
plt.title('Churn by Monthly Charges')
plt.xlabel('Monthly Charge ($)')
data_ohe[data_ohe['churn'] == 0]['monthly_charges'].plot(
    kind='hist',
    bins=20,
    alpha=0.7, 
    density=True
)
data_ohe[data_ohe['churn'] == 1]['monthly_charges'].plot(
    kind='hist',
    bins=20,
    alpha=0.7, 
    density=True
)
plt.legend(['Did not churn', 'Churned'], title='Customer Status')
plt.show()


# In[52]:


# Histogram of Total Charges
plt.title('Churn by Total Charges')
plt.xlabel('Total Charge ($)')
data_ohe[data_ohe['churn'] == 0]['total_charges'].plot(
    kind='hist',
    bins=20,
    alpha=0.7, 
    density=True
)
data_ohe[data_ohe['churn'] == 1]['total_charges'].plot(
    kind='hist',
    bins=20,
    alpha=0.7, 
    density=True
)
plt.legend(['Did not churn', 'Churned'], title='Customer Status')
plt.show()


# We see that in the distributions have similar shape between the customers that churned and did not churn even though the churn class is imbalanced.

# In[53]:


fig = px.scatter(data_ohe, x='monthly_charges', y='total_charges', color='churn')
fig.update_layout(
    title="Charges for Churned vs. Non-Churn Customers"
)
fig.show()


# In the scatter plot above, we see a concentration of churned customers in yellow in the higher monthly charges.

# In[54]:


# boxplot = plt.boxplot(df['Profit'].values)
# outliers = list(boxplot['fliers'][0].get_data()[1])
# df_outliers = df[df["Profit"].isin(outliers)]
# print('Number of anomalies: ', len(df_outliers))
fig = px.box(data_ohe, x="churn", y="total_charges", notched=True, title='Distribution of Total Charges') # color="smoker", 
fig.show()


# The churn customers (1) would naturally have an overall lower distriibution of total charges since their payments stop earlier than non-churned customers.

# In[55]:


fig = px.box(data_ohe, x="churn", y="monthly_charges", notched=True, title='Distribution of Monthly Charges') # color="smoker", 
fig.show()


# In the above boxplot, we see that the monthly charges were overall higher, shown by the IQR box, in the churn customers. This may be a cause for churning.

# In[56]:


g = sns.pairplot(data, kind='hist')
g.fig.set_size_inches(12, 12)


# The pairplots don't give as much info into the data as the previous charts.

# ### Churned Customers

# In[57]:


# Focus on what the churn customers are like
churned = df_all.copy()
churned = churned[churned['churn'] == 1]
churned


# In[58]:


churned.info()


# In[59]:


churned.reset_index(drop=True)


# In[60]:


churned['end_date'] = pd.to_datetime(churned['end_date'], format='%Y-%m-%d')
churned.info()


# In[61]:


# Feature Engineer a new column for the contract length for those who churned
churned['contract_length'] = (churned['end_date'] - churned['begin_date']).dt.days // 30
churned.info()


# In[62]:


fig_hist = px.histogram(churned['contract_length'])  # , title='Conctract Length by Months') #, x='Months')
fig_hist.update_layout(
    title="Conctract Length by Months",
    xaxis_title="Number of Months"
)
fig_hist.show()


# We see in the distribution above that most of the churned customers stayed for less than a year, with many churning within the first six months.

# In[63]:


colm = list(range(3, 6)) + list(range(8,20))
j=1
plt.figure(figsize=(15, 35))
for i in colm:
    #df_all[df_all.columns.values[i]] = df_all[df_all.columns.values[i]].fillna('No')
    #print(churned[churned.columns.values[i]].value_counts())
    plt.subplot(len(colm), 3, j)
    bargraph = churned[churned.columns.values[i]].value_counts().reset_index()
    plt.bar(bargraph['index'], bargraph.iloc[:, 1])
    plt.title(churned.columns.values[i])
    j += 1
plt.suptitle('Categorical Column Charts')
plt.tight_layout(pad=3.0)
plt.show()


# Of the categorical bar charts shown above, we notice the churned customers did not sign up for a lot of the online/internet options like online security and tech support. Also, many of the churned customers pay month-to-month and pay by Electronic check.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Nice EDA!
# 
# </div>

# ## Features and Target

# In[64]:


features = data_ohe.drop(['churn'], axis=1)
target = data_ohe['churn']
display(features)
print(target)


# ## Split the Data
# 
# 60/20/20 split

# In[65]:


# First 80/20 train/test
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=12345
)


# In[66]:


# Then 75/25 train/validation
features_train, features_valid, target_train, target_valid = train_test_split(
    features_train, target_train, test_size=0.25, random_state=12345
)


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The data split is reasonable!
# 
# </div>

# In[67]:


print(features_valid)
print(features_test)


# In[68]:


# Scaling the data set
scaler = MaxAbsScaler()
scaler.fit(features_train)
features_train = scaler.transform(features_train)
print(features_train)
features_valid = scaler.transform(features_valid)
print(features_valid)
features_test = scaler.transform(features_test)
print(features_test)


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Scaling was applied correctly!
# 
# </div>

# Now our data is encoded, split and scaled for our model testing.

# ## Dummy Classifier

# In[69]:


from sklearn.dummy import DummyClassifier


# In[70]:


dummy_clf = DummyClassifier(strategy="most_frequent", random_state=0)

# Fit the dummy classifier on the training data
dummy_clf.fit(features_train, target_train)

# Predict the labels for the test set
dummy_predictions = dummy_clf.predict(features_test)

# Calculate the accuracy of the dummy classifier
#dummy_accuracy = accuracy_score(target_test, dummy_predictions)
#print(dummy_predictions)
#print(f"Baseline Dummy Classifier Accuracy: {dummy_accuracy:.2f}")
dummy_pp = dummy_clf.predict_proba(features_test)
#print(dummy_pp)
dummy_proba = dummy_pp[:, 1]
#print(dummy_proba)
print(roc_auc_score(target_test, dummy_proba))


# Our dummy model set the standard to beat for our models of 0.5.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Good idea to make a simple baseline!
# 
# </div>

# ## Model Training
# 
#     - DecisionTreeClassifier
#     - RandomForrestClassifier
#     - LogisticRegression
#     - Gradient Boosting (XGBoost)

# In[71]:


model = DecisionTreeClassifier(max_depth=6, random_state=12345) 

model.fit(features_train, target_train)
predictions_train = model.predict(features_train)
predictions_valid = model.predict(features_valid)
pred_proba_train = model.predict_proba(features_train)
pred_proba_valid = model.predict_proba(features_valid)
proba_one_train = pred_proba_train[:, 1]
proba_one_valid = pred_proba_valid[:, 1]

print('Train Set:')
#print('F1:', f1_score(target_train, predictions_train))
print('AUC-ROC:', roc_auc_score(target_train, proba_one_train))
print()
print('Validation Set:')
#print('F1:', f1_score(target_valid, predictions_valid))
print('AUC-ROC:', roc_auc_score(target_valid, proba_one_valid))


# In[72]:


model = RandomForestClassifier(n_estimators=60, max_depth=14, random_state=12345) 

model.fit(features_train, target_train)
predictions_train = model.predict(features_train)
predictions_valid = model.predict(features_valid)
pred_proba_train = model.predict_proba(features_train)
pred_proba_valid = model.predict_proba(features_valid)
proba_one_train = pred_proba_train[:, 1]
proba_one_valid = pred_proba_valid[:, 1]

print('Train Set:')
#print('F1:', f1_score(target_train, predictions_train))
print('AUC-ROC:', roc_auc_score(target_train, proba_one_train))
print()
print('Validation Set:')
#print('F1:', f1_score(target_valid, predictions_valid))
print('AUC-ROC:', roc_auc_score(target_valid, proba_one_valid))


# In[73]:


model_lgr = LogisticRegression(solver='liblinear', C=4, random_state=12345)
model_lgr.fit(features_train, target_train)

pred_proba_train = model_lgr.predict_proba(features_train)
pred_proba_valid = model_lgr.predict_proba(features_valid)
proba_one_train = pred_proba_train[:, 1]
proba_one_valid = pred_proba_valid[:, 1]
print('Train Set:')
print('AUC-ROC:', roc_auc_score(target_train, proba_one_train))
print()
print('Validation Set:')
print('AUC-ROC:', roc_auc_score(target_valid, proba_one_valid))


# In[74]:


# %%timeit -n 3 -r 2
# XGBoost
xgbr = XGBClassifier(objective='binary:logistic', n_estimators=50, max_depth=2) #learning_rate=1)
xgbr.fit(features_train, target_train)
pred_prob_xgbr_train = xgbr.predict_proba(features_train)
pred_prob_xgbr_valid = xgbr.predict_proba(features_valid)
proba_train = pred_prob_xgbr_train[:, 1]
proba_valid = pred_prob_xgbr_valid[:, 1]
print("ROC_AUC Train:", roc_auc_score(target_train, proba_train))
print("ROC_AUC Valid:", roc_auc_score(target_valid, proba_valid))


# All four of the models on a preliminary training all managed to exceed the dummy model and achieved over 0.8 on the validation set.

# ## GridSearchCV

# In[75]:


def model_optimizer(model, parameters, scoring_metric='roc_auc', cv=5):

    grid_search = GridSearchCV(model, parameters, cv=cv, scoring=scoring_metric)
    grid_search.fit(features_valid, target_valid)  # features_train, target_train

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_params, best_score


# In[76]:


# DecisionTreeClassifier
dt_params_dtc = {'max_depth': list(range(1, 18)),
             'random_state': [12345]
            }
dt_model_dtc = DecisionTreeClassifier()

best_params, best_score = model_optimizer(dt_model_dtc, dt_params_dtc, scoring_metric='roc_auc')
print(best_params)
print(best_score)


# In[77]:


# RandomForestClassifier
dt_params_rfr = {'max_depth': list(range(11, 19)),
                 'n_estimators': list(range(30, 71, 10)),
                 'random_state': [12345]
                }
dt_model_rfr = RandomForestClassifier()

best_params_rfr, best_score_rfr = model_optimizer(dt_model_rfr, dt_params_rfr, scoring_metric='roc_auc')
print(best_params_rfr)
print(best_score_rfr)


# In[78]:


# LogisticRegression
dt_params_lr = {'C': list(range(1, 11)),
                 'random_state': [12345]  
                }
dt_model_lr = LogisticRegression(solver='liblinear')

best_params_lr, best_score_lr = model_optimizer(dt_model_lr, dt_params_lr, scoring_metric='roc_auc')
print(best_params_lr)
print(best_score_lr)


# In[80]:


# XGBClassifier
dt_params_xgbc = {'max_depth': list(range(1, 5)),
                  'n_estimators': list(range(50, 81, 10))
                }
dt_model_xgbc = XGBClassifier(objective='binary:logistic')

best_params_xgbc, best_score_xgbc = model_optimizer(dt_model_xgbc, dt_params_xgbc, scoring_metric='roc_auc')
print(best_params_xgbc)
print(best_score_xgbc)


# We used GrudSearchCV to hypertune our parameters for the four models.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Great, you tried a couple of different models and tunedf their hyperparameters using cross-validation
# 
# </div>

# ## Model Testing

# In[79]:


# Combine the train and valid sets
features_all = np.concatenate((features_train, features_valid))
features_all


# In[80]:


target_all = np.concatenate((target_train, target_valid))
target_all


# In[81]:


# DecisionTreeClassifier
model_dtc = DecisionTreeClassifier(max_depth=4, random_state=12345)
model_dtc.fit(features_all, target_all)   #   features_train, target_train)
predict_proba_test_dtc = model_dtc.predict_proba(features_test)
predict_test_dtc = predict_proba_test_dtc[:, 1]
print(roc_auc_score(target_test, predict_test_dtc))  # train: 0.8164164766759741; all: 0.8251402515562158


# In[82]:


# RandomForestClassifier
model_rfc = RandomForestClassifier(n_estimators=70, max_depth=11, random_state=12345)
model_rfc.fit(features_all, target_all)   # features_train, target_train)
predict_proba_test = model_rfc.predict_proba(features_test)
predict_test = predict_proba_test[:, 1]
print(roc_auc_score(target_test, predict_test))  # train 0.8378704818505522 ; all 0.8386748469400825


# In[83]:


# LogisticRegression
model_lr = LogisticRegression(solver='liblinear', C=4, random_state=12345)
model_lr.fit(features_all, target_all)   # features_train, target_train)
predict_proba_test_lr = model_lr.predict_proba(features_test)
predict_test_lr = predict_proba_test_lr[:, 1]
print(roc_auc_score(target_test, predict_test_lr))  # train 0.8404718600302278; all 0.8402669262494556


# In[84]:


# XGBClassifier
model_xgbc = XGBClassifier(objective='binary:logistic', n_estimators=50, max_depth=1)
model_xgbc.fit(features_all, target_all)   # features_train, target_train)
predict_proba_test_xgbc = model_xgbc.predict_proba(features_test)
predict_test_xgbc = predict_proba_test_xgbc[:, 1]
print(roc_auc_score(target_test, predict_test_xgbc))  # train 0.8453057355841893 ; all 0.8427261316187207


# When using our four models on the test set, we were able to acheive AUC-ROC scores well above our dummy model standard and all above 0.82.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The tuned models were compared using the test set
#     
# </div>

# <!--## Class Balancing -->

# <!--
# # Upsampling
# def upsample(features, target, repeat):
#     features_zeros = features[target == 0]
#     features_ones = features[target == 1]
#     target_zeros = target[target == 0]
#     target_ones = target[target == 1]
# 
#     features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
#     target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
# 
#     features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=12345)
#     return features_upsampled, target_upsampled
# 
# features_upsampled, target_upsampled = upsample(
#     features_train, target_train, 4
# )
# 
# print(features_upsampled.shape)
# print(target_upsampled.shape)
# print(target_upsampled.value_counts() / target_upsampled.shape[0])
# -->

# <!--
# # Downsampling
# def downsample(features, target, fraction):
#     features_zeros = features[target == 0]
#     features_ones = features[target == 1]
#     target_zeros = target[target == 0]
#     target_ones = target[target == 1]
# 
#     features_downsampled = pd.concat(
#         [features_zeros.sample(frac=fraction, random_state=12345)]
#         + [features_ones]
#     )
#     target_downsampled = pd.concat(
#         [target_zeros.sample(frac=fraction, random_state=12345)]
#         + [target_ones]
#     )
# 
#     features_downsampled, target_downsampled = shuffle(
#         features_downsampled, target_downsampled, random_state=12345
#     )
# 
#     return features_downsampled, target_downsampled
# 
# 
# features_downsampled, target_downsampled = downsample(
#     features_train, target_train, 0.25
# )
# 
# print(features_downsampled.shape)
# print(target_downsampled.shape)
# print(target_downsampled.value_counts() / target_downsampled.shape[0])
# -->

# ## Conclusions

# From the data provided by Interconnect, we saw the following:
# 
# - The churn customers tend to leave in the first half year
# - There was a high churn rate for those with a higher monthly total
# - The data should use one-hot encoding and MaxAbsScalar
# - When predicting the customers that will churn, the four optimized models (XGBClassifier, LogisticRegression, RandomForestClassifier, and DecisionTreeClassifier) work well for predicting with at least a AUC-ROC score of 0.82
#     - Although XGBClassifier had the highest score, the other three models came close with less than a 0.02 difference
# - When looking at the categorical columns, churned customers tended to pay month-to-month, pay with electronic check, and did not use the internet options like online backup.
# 
# With this information, Interconnect can forecast who is most likely to churn before they churn and offer some kind of promotions or incentives. For example, since we saw a large amount of customers stay for less than half a year, the company can offer a multiple month package for a standard rate, regardless of what features they sign up for. This can help curb some of the customers who have a high monthly total to stay with the company since the per month cost would be set.

# ## Project Report

# **What steps of the plan were performed and what steps were skipped (explain why)?**
# 
# All the data preprocessing was done in the project plan section. This included checking each individual data set and cleaning up the data before they were merged into a single dataframe. Once the data sets were merged, the null values had to be handled.
# 
# When continuing on with the project plan, I noticed that some parts of the plan were in the wrong order when it came time to enact the original plan. For example, when encoding the data, I ended up applying the MaxAbsScalar after splitting the data into its train, valid, and test sets rather than when the one-hot encoding was performed.
# 
# One part of the original plan that I chose to drop was the class balancing. When I made the visualizations I noticed that the customers that churned and those that did not had similar density distributions in terms of the monthly and total charges. This I felt was indicative that class balancing would not be necessary since the weight of these columns would be almost equal in the model predictions.

# **What difficulties did you encounter and how did you manage to solve them?**
# 
# During model training with optimal parameters, I had set some of the parameters for GridSearchCV too wide and caused the notebook to time out on a couple occasions. After looking on various forums on standard parameters that others start out with, I narrowed them down. This was especially helpful on hypertuning the XGBClassifier model.

# **What were some of the key steps to solving the task?**
# 
# Merging the four data sets by customer_id was essential to getting a holistic picture of the customers' behaviours. Then, encoding the data using one-hot encoding and MaxAbsScalar helped balance the data out in terms of the weights of each column, making the models better in this process. Another key step was using GridSearchCV to hypertune the parameters of the four models used. It would have been far more efficient than guessing and checking or writing my own function to hypertune the models. 

# **What is your final model and what quality score does it have?**
# 
# The XGBClassifier model with n_estimators=50 and max_depth=1 would be the final model that I would suggest for Interconnect to help predict what customers might churn in the future. With a AUC-ROC score of about 0.84, the quality should be high enough for Interconnect's purpose. Since they want to predict possible customers that churn, they would cover a high amount of those thinking of churning with the business strategy that they want to employ like offering a discount or special rate.

# In[ ]:




