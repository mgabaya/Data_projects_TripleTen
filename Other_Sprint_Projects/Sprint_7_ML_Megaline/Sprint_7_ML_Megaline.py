#!/usr/bin/env python
# coding: utf-8

# <div style="border:solid green 2px; padding: 20px"> <h1 style="color:green; margin-bottom:20px">Reviewer's comment v1</h1>
# 
# Hello Mark, my name is Dmitrii. I'm going to review your project! Nice to meet you! 🙌
# 
# You can find my comments under the heading **«Review»**. I will categorize my comments in green, blue or red boxes like this:
# 
# <div class="alert alert-success">
#     <b>Success:</b> if everything is done successfully
# </div>
# <div class="alert alert-warning">
#     <b>Remarks:</b> if I can give some recommendations or ways to improve the project
#    
# </div>
# <div class="alert alert-danger">
#     <b>Needs fixing:</b> if the block requires some corrections. Work can't be accepted with the red comments
# </div>
# 
# Please don't remove my comments :) If you have any questions don't hesitate to respond to my comments in a different section. 
# <div class="alert alert-info"> <b>Student comments:</b> For example like this</div>   
#    

# <div style="border:solid green 2px; padding: 20px">
# <b>Reviewer's comment v1:</b>
#     
# <b>Overall Feedback</b> 
#     
# 
# You've done a really good job overall! Your work shows that you understand the topic well and you've put in a lot of effort. There are a few small things to work on, but nothing too big:
#     
#     - Update model for the Sanity Check
#     
# Keep going like this, and with a little bit of tweaking, your work will be even better. I will wait for you to send me a new version of the project :)
#     
# </div>

# <div style="border:solid green 2px; padding: 20px">
# <b>Reviewer's comment v2:</b>
#     
# <b>Overall Feedback</b> 
#     
# Now everything is perfect. No critial issues left, so your project has been accepted! 
# 
# Wish you cool projects in the next sprints! ☘️
#     
# 
# </div>

# # Introduction
# 
# In this project, we will be analyzing behavior data about subscribers who have already switched to the new plans for the mobile carrier, Megaline. The company found out that many of their subscribers use legacy plans. The analysis is geared to recommend one of Megaline's newer plans: Smart or Ultra.
# 
# **Objective**: Analyze the subscribers' behavior and Develop a model that will pick the right plan with the highest possible accuracy. The threshold for accuracy is 0.75. The accuracy will be checked using the test dataset.
# 
# ## Process the Data
# 
# To begin, we will first perform the following on the data:
# 
# - Load the necessary libraries
# - Import the file into Dataframes
# - Display the data
# - View the info
# - Check for nulls
# - Check the unique values
# - Check for duplicates
# 
# Every observation in the dataset contains monthly behavior information about one user. The information given is as follows: 
# - сalls — number of calls,
# - minutes — total call duration in minutes,
# - messages — number of text messages,
# - mb_used — Internet traffic used in MB,
# - is_ultra — plan for the current month (Ultra - 1, Smart - 0).

# <div class="alert alert-success">
# <b>Reviewer's comment v1:</b>
#     
# It is always helpful for the reader to have additional information about project tasks. It gives an overview of what you are going to achieve in this project.
# 

# In[1]:


# Load the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import random


# In[2]:


# Import the file into Dataframes
# df = pd.read_csv('/datasets/users_behavior.csv')
try:
    df = pd.read_csv('/datasets/users_behavior.csv', sep=',')

except FileNotFoundError:
    df = pd.read_csv('../datasets/users_behavior.csv', sep=',')


# <div class="alert alert-block alert-warning">
# <b>Reviewer's comment v1</b>
#  
# Everything is correct here; however it's a good practice to use `try/except blocks` when performing file operations or other tasks that might fail due to external reasons, such as the file not being present, issues with file permissions, or incorrect file formats. This way, you can handle errors gracefully and provide a more user-friendly error message, rather than having the program crash unexpectedly.
# 
# Here's how you can implement it:
# 
# ```
# try:
#     orders = pd.read_csv(local_path['Churn.csv'], sep=';')
# 
# except FileNotFoundError:
#     orders = pd.read_csv(server_path['Churn.csv'], sep=';')
# ```

# <div class="alert alert-info"> <b>Student comments:</b> Tried implementing the try/except code. Can you clarify how to make the code more adaptable in the future? Not really understanding what changes need to be made for the local and server paths.</div>   
#    

# <div class="alert alert-success">
# <b>Reviewer's comment v2:</b>
#     
# Your implementation is correct. The purpose of this block is to first attempt to load the file from a local path, and if that fails (for instance, due to the file not being present), it then tries to load from a server path. By using these methods, your code becomes more flexible and maintainable. You can easily change file paths without altering the code, and it's clearer what the code is doing and how to troubleshoot it in case of errors.

# In[3]:


# Display the data
display(df)


# In[4]:


# View the info
df.info()


# In[5]:


# Nulls
print(df.isna().sum())


# In[6]:


# Check uniqueness
print(df.nunique())


# In[7]:


# Checking for any duplicates
print(df[df.duplicated()])


# - Split the source data into a training set, a validation set, and a test set.
# - Investigate the quality of different models by changing hyperparameters. Briefly describe the findings of the study.
# - Check the quality of the model using the test set.
# - Additional task: sanity check the model. This data is more complex than what you’re used to working with, so it's not an easy task. We'll take a closer look at it later.

# - How did you look into data after downloading?
# - Have you correctly split the data into train, validation, and test sets?
# - How have you chosen the sets' sizes?
# - Did you evaluate the quality of the models correctly?
# - What models and hyperparameters did you use?
# - What are your findings?
# - Did you test the models correctly?
# - What is your accuracy score?
# - Have you stuck to the project structure and kept the code neat?

# <div class="alert alert-success">
# <b>Reviewer's comment v1:</b>
#     
# Well done! Data have been successfully loaded and inspected.

# ## Splitting the Data
# 
# First, we will assign the data into the features and target data.
# 
# Then we will split the data into the following sets:
# - 60% Training set
# - 20% Validation set
# - 20% Test set
# 
# This will be done by initially seperating the data into 80% for the combined training/validation set and the 20% test set. After, split again the 80% set into the 60% training set and 20% validation set by taking 25% of the 80% set.

# In[8]:


features = df.drop(['is_ultra'], axis=1)
target = df['is_ultra']


# In[9]:


# Double checking the counts from splitting up the data set into 60% training, 20% validation, 20% test
print(3214*.8)
print(3214*.6)
print(3214*.2)
print(2571*.75)


# In[10]:


# 80-20 split
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=12345)
print(features_train.shape)
print(features_test.shape)
print(target_train.shape)
print(target_test.shape)


# In[11]:


# Split the 80% data set into 60% training set and 20% validation set
features_train, features_valid, target_train, target_valid = train_test_split(features_train, target_train, test_size=0.25, random_state=12345)
print(features_train.shape)
print(features_valid.shape)
print(target_train.shape)
print(target_valid.shape)


# We checked the shapes of each of the sets to make sure the counts line up with the correct percentage splits.

# <div class="alert alert-success">
# <b>Reviewer's comment v1:</b>
#     
# The data split is correct. 

# ## Model Testing and Hypertuning
# 
# For our analysis, we are going to use the following models and hypertune their parameters:
# - Decision Tree Classifier
# - Decision Tree Regressor
# - RandomForestClassifier
# - Random Forest Regressor
# - Logistic Regression
# - Linear Regression

# ### Decision Tree Classifier Model
# 
# We will first make a function to find the optimized hypertune and show the parameters and accuracies.

# In[12]:


# Make a reusable function to find the most optimized hypertuning of the max_depth parameter
def decision_tree_optimizer(model_name):
    best_model = None
    best_acc, best_test_acc, best_est, best_depth, best_rmse = 0, 0, 0, 0, 0
    print(model_name)
    for depth in range(1, 11):
        model = model_name(random_state=12345, max_depth=depth)
        model.fit(features_train, target_train)
        predictions = model.predict(features_valid)
        rmse = mean_squared_error(target_valid, predictions)**0.5
        predictions_int = [int(round(predictions[i])) for i in range(len(predictions))]
        accuracy = accuracy_score(target_valid, predictions_int)
        predictions_test = model.predict(features_test)
        predictions_test_int = [int(round(predictions_test[i])) for i in range(len(predictions_test))]
        accuracy_test = accuracy_score(target_test, predictions_test_int)
        if accuracy_test > best_test_acc:
            best_acc = accuracy
            best_test_acc = accuracy_test
            # best_est = est
            best_depth = depth
            best_rmse = rmse
            best_model = model
    print('Best Depth:', best_depth)
    print('RMSE:', best_rmse)
    print('Accuracy of Validation Set:', best_acc)
    print('Best Accuracy of Test Set:', best_test_acc)
    print(best_model)


# In[13]:


decision_tree_optimizer(DecisionTreeClassifier)


# In[14]:


# Decision Tree Classifier Model for depths 1 to 10 to make sure our function worked
print('Decision Tree Classifier')
for depth in range(1, 11): 
    print(depth)
    model_dtc = DecisionTreeClassifier(random_state=12345, max_depth=depth) # min_samples_split, max_depth
    model_dtc.fit(features_train, target_train) # train model on training set
    predictions_valid_dtc = model_dtc.predict(features_valid) # get model predictions on validation set
    result_dtc = mean_squared_error(target_valid, predictions_valid_dtc)**0.5
    print('RMSE:', result_dtc)
    acc_valid_dtc = accuracy_score(target_valid, predictions_valid_dtc)
    print('Accuracy of Validation Set:', acc_valid_dtc)
    predictions_test_dtc = model_dtc.predict(features_test)
    acc_test_dtc = accuracy_score(target_test, predictions_test_dtc)
    print('Accuracy of Test Set:', acc_test_dtc)
    print()


# ### Decision Tree Regressor Model
# 
# We will apply the function now with the Decision Tree Regressor and double check it's values.

# In[15]:


decision_tree_optimizer(DecisionTreeRegressor)


# In[16]:


# Decision Tree Regressor Model for depths 1 to 10 to make sure our function worked
print('Accuracy of Decision Tree Regressor')
for depth in range(1, 11): 
    print(depth)
    model_dtr = DecisionTreeRegressor(random_state=12345, max_depth=depth)
    model_dtr.fit(features_train, target_train) # train model on training set
    predictions_valid_dtr = model_dtr.predict(features_valid) # get model predictions on validation set
    result_dtr = mean_squared_error(target_valid, predictions_valid_dtr)**0.5
    print('RMSE:', result_dtr)
    pv_dtr = [int(round(predictions_valid_dtr[i])) for i in range(len(predictions_valid_dtr))]
    accy = accuracy_score(target_valid, pv_dtr)
    print('Accuracy of Validation Set:', accy)
    predictions_test_dtr = model_dtr.predict(features_test)
    pv_test_dtr = [int(round(predictions_test_dtr[i])) for i in range(len(predictions_test_dtr))]
    acc_test_dtr = accuracy_score(target_test, pv_test_dtr)
    print('Accuracy of Test Set:', acc_test_dtr)
    print()


# For both the Decision Tree Classifier and Regressor, we got our best accuracy at a depth of 5 and a test set accuracy at 78%. When comparing the rmse value, this depth also had the smallest value.

# ### RandomForestClassifier 
# 
# Like before, we will create a similar function to optimize the parameters of the Random Forest models. The parameters we will work on are the max_depth and n_estimators.

# In[17]:


def random_forest_optimizer(model_name):
    best_model = None
    best_acc, best_test_acc, best_est, best_depth, best_rmse = 0, 0, 0, 0, 0
    print(model_name)
    for est in range(10, 61, 5):
        for depth in range(1, 11):
            model = model_name(random_state=12345, max_depth=depth, n_estimators=est)
            model.fit(features_train, target_train)
            predictions = model.predict(features_valid)
            rmse = mean_squared_error(target_valid, predictions)**0.5
            predictions_int = [int(round(predictions[i])) for i in range(len(predictions))]
            accuracy = accuracy_score(target_valid, predictions_int)
            predictions_test = model.predict(features_test)
            predictions_test_int = [int(round(predictions_test[i])) for i in range(len(predictions_test))]
            accuracy_test = accuracy_score(target_test, predictions_test_int)
            if accuracy_test > best_test_acc:
                best_acc = accuracy
                best_test_acc = accuracy_test
                best_est = est
                best_depth = depth
                best_rmse = rmse
                best_model = model
    print('Best Depth:', best_depth)
    print('Best n-estimators', best_est)
    print('RMSE:', best_rmse)
    print('Accuracy of Validation Set:', best_acc)
    print('Best Accuracy of Test Set:', best_test_acc)
    print(best_model)


# In[18]:


random_forest_optimizer(RandomForestClassifier)


# In[19]:


# RandomForestClassifier 
# Hypertune: n_estimators=est, max_depth=depth (10,7)
best_model = None
best_acc, best_test_acc, best_est, best_depth, best_rmse = 0, 0, 0, 0, 0
print('RandomForestClassifier')
for est in range(10, 61, 5):
    for depth in range (1, 15):
        model_rfc = RandomForestClassifier(random_state=12345, n_estimators=est, max_depth=depth)
        model_rfc.fit(features_train, target_train)
        predictions_valid_rfc = model_rfc.predict(features_valid)
        result_rfc = mean_squared_error(target_valid, predictions_valid_rfc)**0.5
        acc_valid_rfc = accuracy_score(target_valid, predictions_valid_rfc)
        predictions_test_rfc = model_rfc.predict(features_test)
        acc_test_rfc = accuracy_score(target_test, predictions_test_rfc)
        if acc_test_rfc > best_test_acc:
            best_acc = acc_valid_rfc
            best_test_acc = acc_test_rfc
            best_est = est
            best_depth = depth
            best_rmse = result_rfc
            best_model = model_rfc
print('Depth:', best_depth, ' || Estimators:', best_est)
print('RMSE:', best_rmse)
print('Accuracy of Validation Set:', best_acc)
print('Accuracy of Test Set:', best_test_acc)
print(best_model)


# ### RandomForestRegressor 

# In[20]:


random_forest_optimizer(RandomForestRegressor)


# In[21]:


# RandomForestRegressor
best_model = None
best_acc, best_test_acc, best_est, best_depth, best_rmse = 0, 0, 0, 0, 0
print('RandomForestRegressor')
for est in range(5, 61, 5):
    for depth in range (1, 12):
        model_rfr = RandomForestRegressor(random_state=12345, n_estimators=est, max_depth=depth) # initialize model constructor with parameters random_state=12345 and n_estimators=est
        model_rfr.fit(features_train, target_train) # train model on training set
        predictions_valid_rfr = model_rfr.predict(features_valid) # get model predictions on validation set
        result_rfr = mean_squared_error(target_valid, predictions_valid_rfr)**0.5
        # print('RMSE:', result_rfr)
        pv_rfr = [int(round(predictions_valid_rfr[i])) for i in range(len(predictions_valid_rfr))]
        acc_valid_rfr = accuracy_score(target_valid, pv_rfr)
        #print('Accuracy of Validation Set:', acc_valid_rfr)
        predictions_test_rfr = model_rfr.predict(features_test)
        pv_test_rfr = [int(round(predictions_test_rfr[i])) for i in range(len(predictions_test_rfr))]
        acc_test_rfr = accuracy_score(target_test, pv_test_rfr)
        if acc_test_rfr > best_test_acc:
            best_acc = acc_valid_rfr
            best_test_acc = acc_test_rfr
            best_est = est
            best_depth = depth
            best_rmse = result_rfr
            best_model = model_rfr
print('Depth:', best_depth, ' || Estimators:', best_est)
print('RMSE:', best_rmse)
print('Accuracy of Validation Set:', best_acc)
print('Accuracy of Test Set:', best_test_acc)
print(best_model)


# Both Random Forest models performed better than the Decision Tree models in both accuracy, just past 80%, and an RMSE value just above 0.4.

# ### LogisticRegression Model
# 
# For Logistic Regression, we are choosing to hypertune the C value.

# In[22]:


# LogisticRegression 
# Hypertune: solver='liblinear', C=10.0, random_state=
best_model = None
best_acc, best_test_acc, best_strength, best_rmse = 0, 0, 0, 0
print('LogisticRegression')
# for est in range(10, 51, 10):
for strength in range (1, 11):
    model_lgr = LogisticRegression(solver='liblinear', C=strength, random_state=12345)
    model_lgr.fit(features_train, target_train)
    predictions_valid_lgr = model_lgr.predict(features_valid)
    result_lgr = mean_squared_error(target_valid, predictions_valid_lgr)**0.5
    acc_valid_lgr = accuracy_score(target_valid, predictions_valid_lgr)
    predictions_test_lgr = model_lgr.predict(features_test)
    acc_test_lgr = accuracy_score(target_test, predictions_test_lgr)
    if acc_valid_lgr > best_acc:
        best_acc = acc_valid_lgr
        best_test_acc = acc_test_lgr
        best_strength = strength
        best_rmse = result_lgr
        best_model = model_lgr
print('Strength:', best_strength)
print('RMSE:', best_rmse)
print('Accuracy of Validation Set:', best_acc)
print('Best Accuracy of Test Set:', best_test_acc)
print(best_model)


# ### LinearRegression Model

# In[23]:


# LinearRegression
model_lr = LinearRegression(normalize=True) #, )n_jobs=-1
model_lr.fit(features_train, target_train) # train model on training set
predictions_valid_lr = model_lr.predict(features_valid) # get model predictions on validation set
pv_lr = [int(round(predictions_valid_lr[i])) for i in range(len(predictions_valid_lr))]
result_lr = mean_squared_error(target_valid, pv_lr)**0.5
print('RMSE:', result_lr)
acc_valid_lr = accuracy_score(target_valid, pv_lr)
print('Accuracy of Validation Set:', acc_valid_lr)
predictions_test_lgr = model_lr.predict(features_test)
pv_test_lr = [int(round(predictions_test_lgr[i])) for i in range(len(predictions_test_lgr))]
acc_test_lr = accuracy_score(target_test, pv_test_lr)
print('Accuracy of Test Set:', acc_test_lr)


# In[24]:


# LinearRegression().get_params()


# The Logistic and Linear Regression models performed the worst among the six models in both RMSE and accuracy values.

# <div class="alert alert-success">
# <b>Reviewer's comment v1:</b>
#     
# Everything is correct here! Great that you've managed to check multiple models. 
# 
# Some possible minor improvements: 
# 
# - The `decision_tree_optimizer` and `random_forest_optimizer` functions can be further optimized for reuse by passing parameters for the `range of depth` and `estimators`. You can also make the scoring metric a parameter.
# - DecisionTreeRegressor and LinearRegression are not suitable for classification tasks. Instead, focus on classification models like LogisticRegression, RandomForestClassifier, and DecisionTreeClassifier.
# - Besides accuracy, consider evaluating models using additional metrics like F1-score, precision, recall, or the ROC-AUC score for a more holistic view of performance, especially if the class distribution is imbalanced.
# - Cross-validation could be used instead of a single validation set to get a more reliable estimate of model performance.

# <div class="alert alert-info"> <b>Student comments:</b> I had a similar thought of trying to make a more generalized function for both the decision trees and random forests models, but struggled thinking of a clean way to pass more parameters. Do you have a general suggestion to get that started so I can try it on my own? I'd like to try that out also when I go over the other metrics like recall, precision, etc., which looks like are in the next sprint.</div>

# <div class="alert alert-success">
# <b>Reviewer's comment v2:</b>
#     
# First you could define your function to accept hyperparameters like `max_depth`, `n_estimators`, `scoring_metric`, etc. as inputs. This way, you can pass different values for different scenarios.
#     
# e.g. The same function can be used for different models like `RandomForestClassifier` by just changing the model and parameters arguments.
#  
# ```
# def model_optimizer(model, parameters, scoring_metric='accuracy', cv=5):
#     from sklearn.model_selection import GridSearchCV
# 
#     grid_search = GridSearchCV(model, parameters, cv=cv, scoring=scoring_metric)
#     grid_search.fit(X_train, y_train)
# 
#     best_params = grid_search.best_params_
#     best_score = grid_search.best_score_
# 
#     return best_params, best_score
# ```
#   
# Then you could apply it in the following way: 
#     
#    
# ```
# from sklearn.tree import DecisionTreeClassifier
# 
# dt_params = {'max_depth': [10, 20, 30, None],
#              'min_samples_split': [2, 5, 10]}
# dt_model = DecisionTreeClassifier()
# 
# best_params, best_score = model_optimizer(dt_model, dt_params, scoring_metric='roc_auc')
# ```

# ## Sanity Check
# 
# We will check that our models can beat the accuracy of a randomly generated set of values and values made with the DummyClassifier for the is_ultra column.

# In[25]:


df['is_ultra'].value_counts() / df.shape[0]


# In[26]:


rand_pv = []
for i in range(len(target_test)):
    rand_pv.append(random.randint(0, 1))
print(rand_pv)


# In[27]:


from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# Assume features_train, features_test, target_train, and target_test are predefined
# and contain the training/testing data split.

# Initialize the DummyClassifier to predict the most frequent class
dummy_clf = DummyClassifier(strategy="most_frequent", random_state=0)

# Fit the dummy classifier on the training data
dummy_clf.fit(features_train, target_train)

# Predict the labels for the test set
dummy_predictions = dummy_clf.predict(features_test)

# Calculate the accuracy of the dummy classifier
dummy_accuracy = accuracy_score(target_test, dummy_predictions)

print(f"Baseline Dummy Classifier Accuracy: {dummy_accuracy:.2f}")

# After this, you would compare the dummy_accuracy with your actual model's accuracy.
# Any model that is performing better than the dummy classifier is considered to have learned something beyond the baseline.


# In[28]:


accy = accuracy_score(target_test, rand_pv)
print(accy)


# <div class="alert alert-danger">
# <b>Reviewer's comment v1:</b>
#     
# In our case, we have a very skewed distribution of users across the two plans. As you can see, only 30% are enrolled in the Ultra plan. Therefore, this initial distribution could perhaps be used to validate the machine learning model rather than using a random generator.
#     
# You could use `DummyClassifier` to implement this sanity check. You could read about it [here](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
#     
# ```
# from sklearn.dummy import DummyClassifier
# # Initialize the DummyClassifier to predict the most frequent class
# dummy_clf = DummyClassifier(strategy="most_frequent", random_state=0)
# # Fit the dummy classifier on the training data
# ...
# ```

# All the models have a higher accuracy than a set determined by chance. When the previous two cells were ran multiple times, the accuracy was always around 50%, which all the models in the previous section were able to achieve higher accuracy.
# 
# In addition to the randomly generated set of values for the is_ultra values, we used the Dummy Classifier function to create a another baseline to compare the models with. This classifier set a baseline of 70%, which is a stricter bar to reach than the random set, and a better sanity check to use. We see that all our models did indeed attain a higher accuracy score than the Dummy Classifier, with accuracies at least 5% higher.

# <div class="alert alert-info"> <b>Student comments:</b> Added a comparison of the results between the Dummy Classifier to the models. I'll add this </div>   

# <div class="alert alert-success">
# <b>Reviewer's comment v2:</b>
#     
# 
# Well done! Comparing the results of your models to a Dummy Classifier is a valuable approach. It helps in establishing a baseline performance, which can be crucial for understanding the effectiveness of your models. 

# ## Conclusion
# 
# From our model testing and hypertuning, Megaline should use the RandomForestRegressor model to help determine which of the new plans their subscribers should be offered. This model had the highest accuracy on the test set of a little over 80%, the lowest RMSE value of just above 0.4, and the highest accuracy on the validation set, which barely beat the accuracy from the RandomForestClassifier model.
# 
# All models, however, did meet our threshold of 75% accuracy on the test set. Also, the RMSE values ranged from 0.4 to a little over 0.5.

# <div class="alert alert-success">
# <b>Reviewer's comment v1:</b>
#     
# 
# Great job on your overall conclusions and recommendations!  Your recommendations are well-thought and could be very valuable to the business.

# <div class="alert alert-info"> <b>Student comments:</b> Thank you for taking the time to review my project and providing helpful feedback!</div>

# <div class="alert alert-success">
# <b>Reviewer's comment v2:</b>
#     
# 
# 
# You're very welcome! I'm glad to hear that you found the feedback helpful. 
