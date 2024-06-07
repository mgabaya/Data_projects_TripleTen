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
# First of all, thank you for turning in the project! You did a great job overall! There are only a couple of small issues that need to be addressed before the project is accepted. Let me know if you have questions!

# # Introduction
# 
# Beta Bank has provided data on their clients' past behavior and termination of contracts with the bank. The bank notices that their customers are leaving a little more every month. Because of this, the bankers want to predict whether a customer will be leaving.
# 
# 
# ## Objective
# 
# Using the data collected by Beta Bank, we will perform the following: 
# - Make a prediction model on whether or not a client will terminate their contract with Beta Bank.
# - The model should achieve the maximum possible F1 score greater than 0.59 on the test set.
# - Compare the AUC-ROC metric and with the F1 score.

# <!--<div class="alert alert-info"; style="border-left: 7px solid blue">
# <b>Student's comment</b>
#     I'll use this blue box for my comments. Thank you for reviewing my project! 
# </div>-->

# ## Prepare the Data
# 
# To begin, we will first perform the following on the data:
# 
# - Load the necessary libraries
# - Import the file into Dataframes
# - Display the data
# - View the info
# - Adjust data set for unnecessary columns
# - Check for nulls and fill in, if possible
# - Check the unique values
# - Check for duplicates
# - Encode the data
# - Split the data

# In[1]:


# Load the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error,roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression, LinearRegression


# In[2]:


# Import the file into Dataframes
try:
    df = pd.read_csv('/datasets/Churn.csv', sep=',')

except FileNotFoundError:
    df = pd.read_csv('../datasets/Churn.csv', sep=',')
# data = pd.read_csv('/datasets/Churn.csv')


# In[3]:


display(df)


# In[4]:


print(df.info())


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The data was loaded and inspected
# 
# </div>

# ### Data description
# 
# Features
# - RowNumber — data string index
# - CustomerId — unique customer identifier
# - Surname — surname
# - CreditScore — credit score
# - Geography — country of residence
# - Gender — gender
# - Age — age
# - Tenure — period of maturation for a customer’s fixed deposit (years)
# - Balance — account balance
# - NumOfProducts — number of banking products used by the customer
# - HasCrCard — customer has a credit card
# - IsActiveMember — customer’s activeness
# - EstimatedSalary — estimated salary
# 
# Target
# - Exited — сustomer has left (0 = stayed; 1 = exited)

# For the purposes of improving our models later on, we see that the columns RowNumber, CustomerId, and Surname should be dropped. These identifiers won't help predict if the clients exit or not.

# In[5]:


data = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Makes sense!
# 
# </div>

# In[6]:


print(data.head(10))


# In[7]:


data.info()


# Since Geography and Gender are string objects, we will have to encode them to be usable for our models.

# In[8]:


print(data.isna().sum())


# In[9]:


print(data[data['Tenure'].isna()])


# There's null values in the Tenure column. We will inspect the records with nulls and determine what to do with them since they can possibly effect the models later on. 
# 
# Our options can be to:
# - use the mean or median to fill in
# - drop the rows moving forward

# In[10]:


data['Tenure'].sort_values().unique()


# In[11]:


print(data['Tenure'].mean())
print(data['Tenure'].median())


# Since the mean and median are so close, we will fill the null values in with the value of 5.

# In[12]:


data['Tenure'].sort_values().value_counts()


# In[13]:


# Fill null values
data['Tenure'] = data['Tenure'].fillna(data['Tenure'].median())


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Missing values were dealt with reasonably
# 
# </div>

# In[14]:


# Check uniqueness
print(data.nunique())


# In[15]:


print(data[data.duplicated()])
#print(data[data['CustomerId'].duplicated()])


# ### Encode the Data
# We saw earlier that Geography and Gender were object string columns. So, we will have to encode them for our models. Similarly, we will change some of the int and float columns into scalars.

# In[16]:


data_ohe = pd.get_dummies(data, drop_first=True)
print(data_ohe)


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Categorical features were encoded
# 
# </div>

# In[17]:


features = data_ohe.drop(['Exited'], axis=1)
target = data_ohe['Exited']


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Features and targets were separated
# 
# </div>

# ### Split the Data
# 
# We will use the split of 60% training, 20% validation and 20% test sets.

# In[18]:


# First, extract 20% for test set
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=12345
)
# Next, split the remaining 80% into 60% training and 20% validation sets
features_train, features_valid, target_train, target_valid = train_test_split(
    features_train, target_train, test_size=0.25, random_state=12345
)
print(features_train.shape)
print(features_valid.shape)
print(features_test.shape)


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The data split is reasonable
# 
# </div>

# In[19]:


numeric = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']

scaler = StandardScaler()
scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])
features_test[numeric] = scaler.transform(features_test[numeric])
print(features_train.head(5))


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Scaling was applied correctly
# 
# </div>

# ## Balance of Classes
# 
# Examine the balance of classes. Train the model without taking into account the imbalance. Briefly describe your findings.

# In[20]:


# print(data['Exited'].value_counts() / data.shape[0])
print(target.value_counts() / data.shape[0])


# The amount of customers that exited is about a quarter of the amount who stayed. We will first train the models without taking the imbalance in mind. Then we will incorporate the imbalance later on.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Distribution of targets was examined
# 
# </div>

# ## Build and Optimize Models
# 
# Improve the quality of the model. Make sure you use at least two approaches to fixing class imbalance. Use the training set to pick the best parameters. Train different models on training and validation sets. Find the best one. Briefly describe your findings.

# In[21]:


# Function to optimize the models

def model_optimizer(model_name, dep=0, estim=10, strg=0):
    best_model = None
    best_est, best_depth, best_valid_f1, best_roc_auc = 0, 0, 0, 0
    depth_or_strength = 'Depth'
    estimators = estim+1
    max_dep = dep+1
    if strg != 0:
        max_dep = strg
        depth_or_strength = 'Strength'
    for est in range(10, estimators, 5):
        for depth in range(1, max_dep):
            # if statements for DecTree, RandFor, LogReg
            if model_name == RandomForestClassifier:
                model = model_name(random_state=12345, max_depth=depth, n_estimators=est)
            elif model_name == LogisticRegression:
                model = LogisticRegression(solver='liblinear', C=depth, random_state=12345)
            else:
                model = DecisionTreeClassifier(random_state=12345, max_depth=depth)
            model.fit(features_train, target_train)
            predictions_valid = model.predict(features_valid)
            f1_valid = f1_score(target_valid, predictions_valid)
            pred_proba_valid = model.predict_proba(features_valid)
            proba_one_valid = pred_proba_valid[:, 1]
            roc_auc_valid = roc_auc_score(target_valid, proba_one_valid)
            if f1_valid > best_valid_f1:
                best_est = est
                best_depth = depth
                best_valid_f1 = f1_valid
                best_roc_auc = roc_auc_valid
                best_model = model
    print('Best', depth_or_strength, ':', best_depth)
    print('Best n-estimators', best_est)
    print('Best F1 of Validation Set:', best_valid_f1)
    print('AUC-ROC of Validation Set:', best_roc_auc)
    print(best_model)


# <div class="alert alert-danger">
# <s><b>Reviewer's comment</b>
# 
# 1. The test set's purpose is to get an unbiased estimate of the final model's generalization performance after we're done with model selection and hyperparameter tuning using the validation set. That is only possible if the test set is used only once: to evaluate that final model. Please make sure that you only use the validation set to evaluate the models prior to the final testing
#     
# 2. To calculate ROC-AUC we need slightly different inputs than other metrics. Remember that the ROC curve is constructed by varying the threshold of assigning positive class between 0 and 1. For binary predictions the threshold is predefined, so we need to use 'probabilities' (method `predict_proba` instead of `predict`)
# 
# </div>

# <div class="alert alert-info">
#   Adjusted the model optimizer function to find the best results from the validation set and hopefully did the AUC-ROC correctly this time.
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Awesome, all looks good!
# 
# </div>

# ### RandomForestClassifier Model

# In[22]:


model_optimizer(RandomForestClassifier, dep=12, estim=60)  # 10, 40


# In[23]:


model = RandomForestClassifier(n_estimators=35, max_depth=11, random_state=12345) 

model.fit(features_train, target_train)
predictions_train = model.predict(features_train)
predictions_valid = model.predict(features_valid)
pred_proba_train = model.predict_proba(features_train)
pred_proba_valid = model.predict_proba(features_valid)
proba_one_train = pred_proba_train[:, 1]
proba_one_valid = pred_proba_valid[:, 1]

print('Train Set:')
print('F1:', f1_score(target_train, predictions_train))
print('AUC-ROC:', roc_auc_score(target_train, proba_one_train))
print()
print('Validation Set:')
print('F1:', f1_score(target_valid, predictions_valid))
print('AUC-ROC:', roc_auc_score(target_valid, proba_one_valid))


# ### DecisionTreeClassifier Model

# In[24]:


model_optimizer(DecisionTreeClassifier, dep=12)  # 5


# In[25]:


model = DecisionTreeClassifier(max_depth=7, random_state=12345) 

model.fit(features_train, target_train)
predictions_train = model.predict(features_train)
predictions_valid = model.predict(features_valid)
pred_proba_train = model.predict_proba(features_train)
pred_proba_valid = model.predict_proba(features_valid)
proba_one_train = pred_proba_train[:, 1]
proba_one_valid = pred_proba_valid[:, 1]

print('Train Set:')
print('F1:', f1_score(target_train, predictions_train))
print('AUC-ROC:', roc_auc_score(target_train, proba_one_train))
print()
print('Validation Set:')
print('F1:', f1_score(target_valid, predictions_valid))
print('AUC-ROC:', roc_auc_score(target_valid, proba_one_valid))


# In[26]:


for depth in range(1, 13): 
    print(depth)
    model_dtc = DecisionTreeClassifier(random_state=12345, max_depth=depth) # min_samples_split, max_depth
    model_dtc.fit(features_train, target_train) # train model on training set
    predictions_valid_dtc = model_dtc.predict(features_valid) # get model predictions on validation set
    f1_valid = f1_score(target_valid, predictions_valid_dtc)
    print('F1 of Validation Set:', f1_valid)
    pred_proba_valid = model_dtc.predict_proba(features_valid)
    proba_one_valid = pred_proba_valid[:, 1]
    r_a_score = roc_auc_score(target_valid, proba_one_valid)
    print('AUC-Roc of Test Set:', r_a_score)
    print()


# ### Logistic Regression

# In[27]:


model_optimizer(LogisticRegression, strg=10)


# In[28]:


# LogisticRegression 
# Double checking the optimizer
best_model = None
best_strength, best_f1_valid, best_r_a_valid = 0, 0, 0 # best_rmse, 
print('LogisticRegression')
# for est in range(10, 51, 10):
for strength in range (1, 11):
    model_lgr = LogisticRegression(solver='liblinear', C=strength, random_state=12345)
    model_lgr.fit(features_train, target_train)
    predictions_valid_lgr = model_lgr.predict(features_valid)
    f1_valid_lgr = f1_score(target_valid, predictions_valid_lgr)
    
    pred_proba_valid = model_lgr.predict_proba(features_valid)
    proba_one_valid = pred_proba_valid[:, 1]
    roc_auc_valid = roc_auc_score(target_valid, proba_one_valid)
    if f1_valid_lgr > best_f1_valid:
        best_f1_valid = f1_valid_lgr
        best_strength = strength
        best_model = model_lgr
        best_r_a_valid = roc_auc_valid
print('Strength:', best_strength)
print('Best F1 of Validation Set:', best_f1_valid)
print(best_model)
print('AUC-ROC:', best_r_a_valid)


# <div class="alert alert-warning">
# <b>Reviewer's comment</b>
# 
# RMSE is a regression metric and is not very useful in a classification context
# 
# </div>

# <div class="alert alert-info">
#     Removed the RMSE metric and updated the AUC-ROC code.
# </div>

# Without balancing the classes, the RandomForestClassifier had the best F1 and AUC-ROC scores, followed closely by the DecisionTreeClassifier model in both scores. The Logistic Regression model performed the weakest. However, none of the models were able to reach our F1 target of 0.59, which leads us to try balancing the classes.
# 
# Moving forward, we will balance the classes with upsampling and downsampling. Then, we will rerun our optimized RandomForestClassifier with the parameters of max_depth of 11 and n_estimators at 35, and DecisionTreeClassifier with the max_depth of 7.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Alright, you trained a couple of models without taking the imbalance into account firfst
# 
# </div>

# <div class="alert alert-info">
#     Updated the findings in this section.
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Great!
# 
# </div>

# ## Upsampling and Downsampling
# 
# 

# Since the number of values for customers who stayed is about four times the amount of customers that exited, we will employ upsampling and downsampling to try and improve the chosen models from the previous section. Our results will be compared after both balanced sets have been modeled.

# In[29]:


# Upsampling
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=12345)
    return features_upsampled, target_upsampled

features_upsampled, target_upsampled = upsample(
    features_train, target_train, 4
)

print(features_upsampled.shape)
print(target_upsampled.shape)
print(target_upsampled.value_counts() / target_upsampled.shape[0])


# In[30]:


# Downsampling
def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)]
        + [features_ones]
    )
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)]
        + [target_ones]
    )

    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345
    )

    return features_downsampled, target_downsampled


features_downsampled, target_downsampled = downsample(
    features_train, target_train, 0.25
)

print(features_downsampled.shape)
print(target_downsampled.shape)
print(target_downsampled.value_counts() / target_downsampled.shape[0])


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Upsampling and downsampling were correctly applied only to the train set
# 
# </div>

# ## Final Testing 
# 
# With our upsampled and downsampled data, we will use the models found earlier to be the most optimized on these new data sets and compare results to narrow down which model we will employ in the final test. 
# 
# The first four sets of code will help us choose the final model for our test set. We will use the RandomForestClassifier with the parameters of max_depth of 11 and n_estimators at 35, and DecisionTreeClassifier with the max_depth of 7

# ### Upsampled Set Testing

# In[31]:


# DecisionTree with the upsampled set
model = DecisionTreeClassifier(max_depth=7, random_state=12345) 

model.fit(features_upsampled, target_upsampled) # upsampled set
predictions_valid = model.predict(features_valid)
f1_valid = f1_score(target_valid, predictions_valid)

pred_proba_valid = model.predict_proba(features_valid)
proba_one_valid = pred_proba_valid[:, 1]
roc_auc_valid = roc_auc_score(target_valid, proba_one_valid)

print('Validation Set:')
print('F1:', f1_valid)
print('AUC-ROC:', roc_auc_valid)


# In[32]:


# RandomForest upsample
model = RandomForestClassifier(n_estimators=35, max_depth=11, random_state=12345) 

model.fit(features_upsampled, target_upsampled) # upsampled set
predictions_valid = model.predict(features_valid)
f1_valid = f1_score(target_valid, predictions_valid)
pred_proba_valid = model.predict_proba(features_valid)
proba_one_valid = pred_proba_valid[:, 1]
roc_auc_valid = roc_auc_score(target_valid, proba_one_valid)

print('Validation Set:')
print('F1:', f1_valid)
print('AUC-ROC:', roc_auc_valid)


# ### Downsampled Set Testing

# In[33]:


# Train with the downsampled set
model = DecisionTreeClassifier(max_depth=7, random_state=12345) 

model.fit(features_downsampled, target_downsampled) # downsampled set
predictions_valid = model.predict(features_valid)
f1_valid = f1_score(target_valid, predictions_valid)

pred_proba_valid = model.predict_proba(features_valid)
proba_one_valid = pred_proba_valid[:, 1]
roc_auc_valid = roc_auc_score(target_valid, proba_one_valid)

print('Validation Set:')
print('F1:', f1_valid)
print('AUC-ROC:', roc_auc_valid)


# In[34]:


# RandomForest with the downsampled set
model = RandomForestClassifier(n_estimators=35, max_depth=11, random_state=12345) 

model.fit(features_downsampled, target_downsampled) # downsampled set
predictions_valid = model.predict(features_valid)
f1_valid = f1_score(target_valid, predictions_valid)
pred_proba_valid = model.predict_proba(features_valid)
proba_one_valid = pred_proba_valid[:, 1]
roc_auc_valid = roc_auc_score(target_valid, proba_one_valid)

print('Validation Set:')
print('F1:', f1_valid)
print('AUC-ROC:', roc_auc_valid)


# <div class="alert alert-warning">
# <b>Reviewer's comment</b>
# 
# I would suggest first selecting the balancing technique using the validation set, and only then evaluating the best model using the test set to avoid any bias
# 
# </div>

# <div class="alert alert-info">
#     Updated the previous code to use only the validation sets and made a seperate section below to run the best model on the test set.
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Very good!
# 
# </div>

# ### Final Model on Test Set
# 
# Now that we found that RandomForestClassifier, with the parameters of max_depth of 11 and n_estimators at 35, gave the best F1 and AUC-ROC scores on the upsampled set, we will run the model on our test set.

# In[36]:


# RandomForestClassifier with the upsampled set for final testing
model = RandomForestClassifier(n_estimators=35, max_depth=11, random_state=12345) 

model.fit(features_upsampled, target_upsampled) # upsampled set
predictions_test = model.predict(features_test) # Test sets
f1_test = f1_score(target_test, predictions_test)

pred_proba_test = model.predict_proba(features_test)
proba_one_test = pred_proba_test[:, 1]
roc_auc_test = roc_auc_score(target_test, proba_one_test)

print('Test Set:')
print('F1:', f1_test)
print('AUC-ROC:', roc_auc_test)


# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Excellent, the model beats the F1 score threshold when evaluated on unseen data :)
# 
# </div>

# ## Conclusion
# 
# <!-- - How did you prepare the data for training? Have you processed all of the feature types?
# - Have you explained the preprocessing steps well enough?
# - How did you investigate the balance of classes?
# - Did you study the model without taking into account the imbalance of classes?
# - What are your findings about the task research?
# - Have you correctly split the data into sets?
# - How have you worked with the imbalance of classes?
# - Did you use at least two techniques for imbalance fixing?
# - Have you performed the model training, validation, and final testing correctly?
# - How high is your F1 score?
# - Did you examine the AUC-ROC values?
# - Have you kept to the project structure and kept the code neat?
# -->

# Beta Bank should use the upsampled data to prevent an imbalance of classes. Before using either upsampled or downsampled data, none of our optimized models on the original data had beaten our minimum F1 score of 0.59. 
# 
# The F1 score of 0.636, which beat our original benchmark of 0.59, was achieved by upsampling our data and using the RandomForestClassifier with the n_estimators=35 and max_depth=11 as attributes. This model also produced an AUC-ROC score of 0.860.
# 
# Across the different models, Beta Bank should use the RandomForestClassifier model on the upsampled data to determine if a customer is going to terminate their contract or stay with Beta Bank.
# 
# <!--The AUC-ROC scores among all models, balanced and imbalanced, did beat the base ROC score of 0.5, which was a sanity check in itself. Both the F1 and AUC-ROC scores went up when using both balanced sets. -->

# <div class="alert alert-danger">
# <s><b>Reviewer's comment</b>
# 
# Please check the conclusions after fixing the problems above
# 
# </div>

# <div class="alert alert-info">
#     Updated the conclusion with new results.
#     Thank you for taking the time to review and give helpful feedback!
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# You're welcome! :)
#     
# The project is now accepted. Keep up the good work on the next sprint!
# 
# </div>

# In[ ]:




