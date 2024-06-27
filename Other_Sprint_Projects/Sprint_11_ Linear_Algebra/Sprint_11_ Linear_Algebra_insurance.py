#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Thank you for taking the time to improve the project! It is accepted now. Good luck on the next sprint! 
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
# First of all, thank you for turning in the project! You did a pretty good job overall, but there are some problems that need to be fixed before the project is accepted. Let me know if you have questions!

# # Statement

# The Sure Tomorrow insurance company wants to solve several tasks with the help of Machine Learning and you are asked to evaluate that possibility.
# 
# - Task 1: Find customers who are similar to a given customer. This will help the company's agents with marketing.
# - Task 2: Predict whether a new customer is likely to receive an insurance benefit. Can a prediction model do better than a dummy model?
# - Task 3: Predict the number of insurance benefits a new customer is likely to receive using a linear regression model.
# - Task 4: Protect clients' personal data without breaking the model from the previous task. It's necessary to develop a data transformation algorithm that would make it hard to recover personal information if the data fell into the wrong hands. This is called data masking, or data obfuscation. But the data should be protected in such a way that the quality of machine learning models doesn't suffer. You don't need to pick the best model, just prove that the algorithm works correctly.

# # Data Preprocessing & Exploration
# 
# ## Initialization

# In[1]:


pip install scikit-learn --upgrade


# In[2]:


import numpy as np
import pandas as pd

import seaborn as sns

import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing

from sklearn.model_selection import train_test_split

from IPython.display import display

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier


# ## Load Data

# Load data and conduct a basic check that it's free from obvious issues.

# In[3]:


df = pd.read_csv('/datasets/insurance_us.csv')


# We rename the colums to make the code look more consistent with its style.

# In[4]:


df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})


# In[5]:


df.sample(10)


# In[6]:


df.info()


# In[7]:


# we may want to fix the age type (from float to int) though this is not critical

# write your conversion here if you choose:
df['age'] = df['age'].astype('Int64')


# In[8]:


# check to see that the conversion was successful
df.info()


# In[9]:


# now have a look at the data's descriptive statistics. 
# Does everything look okay?
df.describe()


# There appears to be nothing wrong in the descriptive statistics, but we will check on nulls, uniqueness and duplicates in the data.

# In[10]:


df['gender'].unique()


# In[11]:


df['gender'].value_counts()


# In[12]:


df.isna().sum()


# In[13]:


df[df.duplicated()]


# There is about 3% duplicated rows. It is possible that customer attributes are similar.
# 
# We will need to decide whether to keep or remove these records.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Alright!
# 
# </div>

# ## EDA

# Let's quickly check whether there are certain groups of customers by looking at the pair plot.

# In[14]:


g = sns.pairplot(df, kind='hist')
g.fig.set_size_inches(12, 12)


# Ok, it is a bit difficult to spot obvious groups (clusters) as it is difficult to combine several variables simultaneously (to analyze multivariate distributions). That's where LA and ML can be quite handy.

# # Task 1. Similar Customers

# In the language of ML, it is necessary to develop a procedure that returns k nearest neighbors (objects) for a given object based on the distance between the objects.
# 
# You may want to review the following lessons (chapter -> lesson)
# - Distance Between Vectors -> Euclidean Distance
# - Distance Between Vectors -> Manhattan Distance
# 
# To solve the task, we can try different distance metrics.

# Write a function that returns k nearest neighbors for an $n^{th}$ object based on a specified distance metric. The number of received insurance benefits should not be taken into account for this task. 
# 
# You can use a ready implementation of the kNN algorithm from scikit-learn (check [the link](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors)) or use your own.
# 
# Test it for four combination of two cases
# - Scaling
#   - the data is not scaled
#   - the data is scaled with the [MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html) scaler
# - Distance Metrics
#   - Euclidean
#   - Manhattan
# 
# Answer these questions:
# - Does the data being not scaled affect the kNN algorithm? If so, how does that appear?
# - How similar are the results using the Manhattan distance metric (regardless of the scaling)?

# In[15]:


feature_names = ['gender', 'age', 'income', 'family_members']


# In[16]:


def get_knn(df, n, k, metric):
    
    """
    Returns k nearest neighbors

    :param df: pandas DataFrame used to find similar objects within
    :param n: object no for which the nearest neighbours are looked for
    :param k: the number of the nearest neighbours to return
    :param metric: name of distance metric
    """

    nbrs = NearestNeighbors(n_neighbors=k, metric=metric) # <your code here> 
    nbrs.fit(df.loc[:, feature_names])
    nbrs_distances, nbrs_indices = nbrs.kneighbors([df.iloc[n][feature_names]], k, return_distance=True)
    
    df_res = pd.concat([
        df.iloc[nbrs_indices[0]], 
        pd.DataFrame(nbrs_distances.T, index=nbrs_indices[0], columns=['distance'])
        ], axis=1)
    
    return df_res


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The function for finding nearest neighbors is correct
# 
# </div>

# Scaling the data.

# In[17]:


feature_names = ['gender', 'age', 'income', 'family_members']

transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(df[feature_names].to_numpy())

df_scaled = df.copy()
df_scaled.loc[:, feature_names] = transformer_mas.transform(df[feature_names].to_numpy())


# In[18]:


df_scaled.sample(5)


# Now, let's get similar records for a given one for every combination

# In[19]:


#df_scaled.loc[0].values
get_knn(df_scaled, 5, 6, 'euclidean')


# In[20]:


get_knn(df_scaled, 5, 6, 'cityblock')


# In[21]:


df_unscaled = df.loc[:, feature_names]
print(df_unscaled.sample(5))
print(df_unscaled.info())


# In[22]:


get_knn(df_unscaled, 5, 6, 'euclidean')


# In[23]:


get_knn(df_unscaled, 5, 6, 'cityblock')


# Answers to the questions

# **Does the data being not scaled affect the kNN algorithm? If so, how does that appear?** 
# 
# The unscaled results were not the same as the scaled results for the chosen n. This is shown through the distance column as the distances of the unscaled results were much higher than the scaled results.

# **How similar are the results using the Manhattan distance metric (regardless of the scaling)?** 
# 
# For our chosen k value, the closest k results in the Manhattan results were the same records as the Euclidean results, but some rows were ranked closer or in different orders in terms of distance. For instance, the second result in the unscaled Euclidean set was the fifth result in the unscaled Manhattan set, but still ranked within the k number of results. Furthermore, the Manhattan distances were relatively close to the distances of the Euclidean distances for both the scaled and unscaled data sets.

# <div class="alert alert-warning">
# <b>Reviewer's comment</b>
# 
# An important point here is that without scaling if one feature has much bigger values than others (like income), it will dominate the distance calculation, and other features will be treated as less important (as you can see with unscaled data, neighbors are basically selected just based on income)
# 
# </div>

# <div class="alert alert-info">
# Didn't notice that the first time, makes so much sense!
# </div>

# # Task 2. Is Customer Likely to Receive Insurance Benefit?

# In terms of machine learning we can look at this like a binary classification task.

# With `insurance_benefits` being more than zero as the target, evaluate whether the kNN classification approach can do better than a dummy model.
# 
# Instructions:
# - Build a KNN-based classifier and measure its quality with the F1 metric for k=1..10 for both the original data and the scaled one. That'd be interesting to see how k may influece the evaluation metric, and whether scaling the data makes any difference. You can use a ready implemention of the kNN classification algorithm from scikit-learn (check [the link](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)) or use your own.
# - Build the dummy model which is just random for this case. It should return "1" with some probability. Let's test the model with four probability values: 0, the probability of paying any insurance benefit, 0.5, 1.
# 
# The probability of paying any insurance benefit can be defined as
# 
# $$
# P\{\text{insurance benefit received}\}=\frac{\text{number of clients received any insurance benefit}}{\text{total number of clients}}.
# $$
# 
# Split the whole data in the 70:30 proportion for the training/testing parts.

# In[24]:


# calculate the target

df['insurance_benefits_received'] = df['insurance_benefits'].apply(lambda x: 1 if x > 0 else 0)  #<your code here>


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Target is calculated correctly
# 
# </div>

# In[25]:


# check for the class imbalance with value_counts()

# <your code here>
df['insurance_benefits_received'].value_counts()


# 0 stands for no benefits given and 1 means there were benefits given. This shows that there is a class imbalance.

# In[80]:


def eval_classifier(y_true, y_pred):
    
    f1_score = sklearn.metrics.f1_score(y_true, y_pred) 
    print(f'F1: {f1_score:.2f}')
    
# if you have an issue with the following line, restart the kernel and run the notebook again
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='all')
    print('Confusion Matrix')
    print(cm)


# <div class="alert alert-danger">
# <s><b>Reviewer's comment</b>
# 
# The F1 score is calculated incorrectly: no `average` parameter should be used for binary classification
# 
# </div>

# <div class="alert alert-info">
#     Removed the average parameter.
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Alright!
# 
# </div>

# In[27]:


# generating output of a random model

def rnd_model_predict(P, size, seed=42):

    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)


# In[70]:


#rng = np.random.default_rng(seed=42)
#rng.binomial(n=1, p=.5, size=15)  # len(df))
y_pred_rnd
df['insurance_benefits_received']


# In[81]:


for P in [0, df['insurance_benefits_received'].sum() / len(df), 0.5, 1]:

    print(f'The probability: {P:.2f}')
    y_pred_rnd = rnd_model_predict(P, len(df), seed=42) # <your code here> 
        
    eval_classifier(df['insurance_benefits_received'], y_pred_rnd)
    
    print()


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Alright, there's our random baseline
# 
# </div>

# ### Scaled and Unscaled

# In[91]:


#display(df_scaled)
#display(df_unscaled)
#df


# In[86]:


X_scaled = df[['age', 'gender', 'income', 'family_members']].to_numpy()
y_scaled = df['insurance_benefits_received'].to_numpy()

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=12345)


# <div class="alert alert-warning">
# <b>Reviewer's comment</b>
# 
# To avoid data leakage scaling in supervised learning should be applied after splitting the data. The scaler is fit using the train set, and then both train and test are transformed.
# 
# </div>

# <div class="alert alert-info">
#   Hopefully fixed the the scaled and unscaled part of this task below, at least it feels like I'm in the right direction with your suggestion.
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Yep!
# 
# </div>

# In[87]:


print(X_train_scaled.shape)
print(y_test_scaled.shape)
print(X_test_scaled.shape)


# In[88]:


#feature_names = ['gender', 'age', 'income', 'family_members']

transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(X_train_scaled) #df[feature_names].to_numpy())

#df_scaled = df.copy()
#df_scaled.loc[:, feature_names] 
X_train_scaled = transformer_mas.transform(X_train_scaled)
print(X_train_scaled)
X_test_scaled = transformer_mas.transform(X_test_scaled)
print(X_test_scaled)


# In[89]:


# Scaled 
for k in range(1, 11):
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(X_train_scaled, y_train_scaled)
    y_test_pred = knc.predict(X_test_scaled)
    #print(y_test_pred.shape)
    print('n_neighbors:', k)
    eval_classifier(y_test_scaled, y_test_pred)
    
    print()


# In[90]:


X_unscaled = df_unscaled[['age', 'gender', 'income', 'family_members']].to_numpy()
y_unscaled = df['insurance_benefits_received'].to_numpy()

X_train_unscaled, X_test_unscaled, y_train_unscaled, y_test_unscaled = train_test_split(X_unscaled, y_unscaled, test_size=0.3, random_state=12345)

for k in range(1, 11):
    knc_un = KNeighborsClassifier(n_neighbors=k)
    knc_un.fit(X_train_unscaled, y_train_unscaled)
    y_test_pred_un = knc_un.predict(X_test_unscaled)
    #print(y_test_pred_un)
    print('n_neighbors:', k)
    eval_classifier(y_test_unscaled, y_test_pred_un)
    print()


# In[92]:


#y_test_scaled
#y_test_pred_un


# The F1 scores for all k-neighbors, 1 to 10, of the scaled data beat both the scores for the randomized data and the unscaled data. Only the 1 to 5 k-neighbors produced better F1 scores than the randomized data.

# <div class="alert alert-danger">
# <S><b>Reviewer's comment</b>
# 
# The target should be `insurance_benefits_received`: we're working on binary classification
# 
# </div>

# <div class="alert alert-info">
#   Replaced insurance_benefits with insurance_benefits_received.
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Ok, all good now! So as you can see scaling is very important for distance-based models like kNN
# 
# </div>

# # Task 3. Regression (with Linear Regression)

# With `insurance_benefits` as the target, evaluate what RMSE would be for a Linear Regression model.

# Build your own implementation of LR. For that, recall how the linear regression task's solution is formulated in terms of LA. Check RMSE for both the original data and the scaled one. Can you see any difference in RMSE between these two cases?
# 
# Let's denote
# - $X$ — feature matrix, each row is a case, each column is a feature, the first column consists of unities
# - $y$ — target (a vector)
# - $\hat{y}$ — estimated tagret (a vector)
# - $w$ — weight vector
# 
# The task of linear regression in the language of matrices can be formulated as
# 
# $$
# y = Xw
# $$
# 
# The training objective then is to find such $w$ that it would minimize the L2-distance (MSE) between $Xw$ and $y$:
# 
# $$
# \min_w d_2(Xw, y) \quad \text{or} \quad \min_w \text{MSE}(Xw, y)
# $$
# 
# It appears that there is analytical solution for the above:
# 
# $$
# w = (X^T X)^{-1} X^T y
# $$
# 
# The formula above can be used to find the weights $w$ and the latter can be used to calculate predicted values
# 
# $$
# \hat{y} = X_{val}w
# $$

# Split the whole data in the 70:30 proportion for the training/validation parts. Use the RMSE metric for the model evaluation.

# In[105]:


class MyLinearRegression:
    
    def __init__(self):
        
        self.weights = None
    
    def fit(self, X, y):
        
        # adding the unities
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        self.weights = np.linalg.inv(X2.T @ X2) @ X2.T @ y # <your code here> np.linalg.inv(X2.T.dot(X2)).dot(X2.T).dot(y)
        #self.weights = np.float64(self.weights)

    def predict(self, X):
        
        # adding the unities
        X2 = np.append(np.ones([len(X), 1]), X, axis=1) # <your code here>
        y_pred = X2 @ self.weights # <your code here>
        #y_pred = np.float64(y_pred)
        
        return y_pred


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Linear regression was implemented correctly
# 
# </div>

# In[106]:


X_test = np.array([[2, 9], [8, 10]])
X2_test = np.append(np.ones([len(X_test), 1]), X_test, axis=1)
print(X2_test)
print(X2_test.T)
print(np.linalg.inv(X2_test.T @ X2_test) @ X2_test.T @ [1, 2])


# In[107]:


def eval_regressor(y_true, y_pred):
    
    rmse = (sklearn.metrics.mean_squared_error(y_true, y_pred))**0.5  # math.sqrt
    print(f'RMSE: {rmse:.2f}')
    
    r2_score = (sklearn.metrics.r2_score(y_true, y_pred))**0.5  # math.sqrt
    print(f'R2: {r2_score:.2f}')    


# In[108]:


X = df[['age', 'gender', 'income', 'family_members']].to_numpy()
y = df['insurance_benefits'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(np.float64(X_train), np.float64(y_train))
print(lr.weights)

y_test_pred = lr.predict(X_test)
print(y_test_pred)
eval_regressor(y_test, y_test_pred)


# In[109]:


X_scaled = df_scaled[['age', 'gender', 'income', 'family_members']].to_numpy()
y_scaled = df_scaled['insurance_benefits'].to_numpy()

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=12345)


# In[110]:


lr = MyLinearRegression()

lr.fit(np.float64(X_train_scaled), np.float64(y_train_scaled))
print(lr.weights)

y_test_pred_scaled = lr.predict(X_test_scaled)
print(y_test_pred_scaled)
eval_regressor(y_test_scaled, y_test_pred_scaled)


# Our linear regression model worked the same between the scaled and unscaled data as shown with the same RMSE and R2 values.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Yep! Note that scaling can make a difference if we use L1 or L2 regularization or fit the model using stochastic gradient descent
# 
# </div>

# # Task 4. Obfuscating Data

# It best to obfuscate data by multiplying the numerical features (remember, they can be seen as the matrix $X$) by an invertible matrix $P$. 
# 
# $$
# X' = X \times P
# $$
# 
# Try to do that and check how the features' values will look like after the transformation. By the way, the intertible property is important here so make sure that $P$ is indeed invertible.
# 
# You may want to review the 'Matrices and Matrix Operations -> Matrix Multiplication' lesson to recall the rule of matrix multiplication and its implementation with NumPy.

# In[98]:


personal_info_column_list = ['gender', 'age', 'income', 'family_members']
df_pn = df[personal_info_column_list]


# In[99]:


X = df_pn.to_numpy()


# In[100]:


print(X)


# Generating a random matrix $P$.

# In[101]:


rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))
print(P)
#print(P.shape)


# Checking the matrix $P$ is invertible

# In[102]:


p_inv = np.linalg.inv(P)
print(p_inv)


# Can you guess the customers' ages or income after the transformation?

# In[103]:


x_prime = X @ P
print(x_prime.shape)


# Can you recover the original data from $X'$ if you know $P$? Try to check that with calculations by moving $P$ from the right side of the formula above to the left one. The rules of matrix multiplcation are really helpful here.

# In[104]:


original_x = x_prime @ p_inv
print(original_x)
#print(np.round(original_x))


# We were able to get back to the original matrix X, but with values several decimal places off due to the nature of matrix P's inverse most likely being rounded float values.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Right!
# 
# </div>

# Print all three cases for a few customers
# - The original data
# - The transformed one
# - The reversed (recovered) one

# In[ ]:


print('X:\n', X[0:3])
print('X\' (transformed):\n', x_prime[0:3])
print('Recovered:\n', original_x[0:3])


# You can probably see that some values are not exactly the same as they are in the original data. What might be the reason for that?

# Through the process of calculating P inverse, there is most likely rounding done to the values provided. And, once multiplied back to our transformed matrix, X', the rounding throws off the values slightly.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Yep!
# 
# </div>

# ## Proof That Data Obfuscation Can Work with LR

# The regression task has been solved with linear regression in this project. Your next task is to prove _analytically_ that the given obfuscation method won't affect linear regression in terms of predicted values i.e. their values will remain the same. Can you believe that? Well, you don't have to, you should prove it!

# So, the data is obfuscated and there is $X \times P$ instead of just $X$ now. Consequently, there are other weights $w_P$ as
# $$
# w = (X^T X)^{-1} X^T y \quad \Rightarrow \quad w_P = [(XP)^T XP]^{-1} (XP)^T y
# $$
# 
# How would $w$ and $w_P$ be linked if you simplify the formula for $w_P$ above? 
# 
# What would be predicted values with $w_P$? 
# 
# What does that mean for the quality of linear regression if you measure it with RMSE?
# 
# Check Appendix B Properties of Matrices in the end of the notebook. There are useful formulas in there!
# 
# No code is necessary in this section, only analytical explanation!

# **Answer**

# Since $(XP)$ creates a new matrix, let's say $Z$, with the same dimensions, it would work exactly like the $w$ formula with Z in place of X.
# 
# The predicted values with $w_P$ would be the target values.
# 
# The quality of the RMSE should be the same between the obfuscated and non-obfuscated data.

# **Analytical proof**

# <!--Let $Z$ substitute for $(XP)$ in \$w_P = [(XP)^T XP]^{-1} (XP)^T y$
# 
# Then, $w_P = [Z^T Z]^{-1} Z^T y$.
# 
# Thus, $w_P$ works exactly like $w$.
# - - - -->
# $w_P$ and $w$ are linked by multiplying an obfuscating matrix by $w$ as shown below:
# 
# $w_P = [(XP)^T XP]^{-1} (XP)^T y$
# 
# $w_P = [P^T X^T XP]^{-1} P^T X^T y$ by the Reversivity of the transpose of a product of matrices
# 
# $w_P = P^{-1} X^{-1} (X^T)^{-1} (P^T)^{-1} P^T X^T y$ by the property of $(AB)^{-1} = B^{-1}A^{-1}$
# 
# $w_P = P^{-1} X^{-1} (X^T)^{-1} I X^T y$ by the identity property on $(P^T)^{-1} P^T$
# 
# $w_P = P^{-1} [X^{-1} (X^T)^{-1}] X^T y$ by the Associative property of multiplication
# 
# $w_P = P^{-1} (X^T X)^{-1} X^T y$ by the property of $(AB)^{-1} = B^{-1}A^{-1}$
# 
# So, $w_P = P^{-1} w$ 

# <div class="alert alert-danger">
# <b>Reviewer's comment</b>
# 
# Well, it does work like $w$, but we need to find out the exact relationship between $w$ and $w_P$ to actually prove that RMSE won't change.
#     
# Can you try simplifying the formula $w_P = [(XP)^T XP]^{-1} (XP)^T y$ using matrix algebra rules at the bottom of this notebook?
# 
# </div>

# <div class="alert alert-info">
#   Updated analytical proof.
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Awesome! The proof is correct, indeed $w_P = P^{-1} w$. And the last step we need is to prove that predictions will be the same. Let's calculate them:
#     
# $$\hat{y_P} = X' w_P = XPP^{-1} w = Xw = \hat{y}$$
#     
# Indeed they are equal to the old model's predictions, thus RMSE will be the same.
# 
# </div>

# ## Test Linear Regression With Data Obfuscation

# Now, let's prove Linear Regression can work computationally with the chosen obfuscation transformation.
# 
# Build a procedure or a class that runs Linear Regression optionally with the obfuscation. You can use either a ready implementation of Linear Regression from sciki-learn or your own.
# 
# Run Linear Regression for the original data and the obfuscated one, compare the predicted values and the RMSE, $R^2$ metric values. Is there any difference?

# **Procedure**
# 
# - Create a square matrix $P$ of random numbers.
# - Check that it is invertible. If not, repeat the first point until we get an invertible matrix.
# - Obfuscate the data by multiplying the original data with $P$, which will be used in our Linear Regression modeling.
# - Use $XP$ as the new feature matrix

# In[93]:


P = np.random.randint(0, 11, size=(4,4))
print(P)


# In[94]:


np.linalg.inv(P)


# In[95]:


X_obf = df[['age', 'gender', 'income', 'family_members']].to_numpy()
y_obf = df['insurance_benefits'].to_numpy()


# In[96]:


x_p = X_obf @ P
print(x_p)


# In[111]:


X_obf_train, X_obf_test, y_obf_train, y_obf_test = train_test_split(X, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(np.float64(X_obf_train), np.float64(y_obf_train))
print(lr.weights)

y_test_pred_obf = lr.predict(X_obf_test)
print(y_test_pred_obf)
eval_regressor(y_obf_test, y_test_pred_obf)


# We see that even with data obfuscation, we are able to get the same values for RMSE and R2 as we did earlier in Task 3 with just the original data.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Indeed! The experiment was a success!
# 
# </div>

# # Conclusions

# Through our analysis, we found the following that can be of use to the Sure Tomorrow insurance company:
# 
# - Scaled data can drastically close the distance of the nearest neighbor compared to unscaled data, which can help group similar customers together. 
# - Euclidean and Manhattan distances come up with similar results and similar distances regardless of using scaled or unscaled data.

# - Similarly, scaled and unscaled data provide the same RMSE and R2 values using our linear regression model.

# - With data obfuscation, the linear regression models can be used the same as if the data was not obfuscated. This can lend well to encrypting customer data that the insurance company would want to protect.

# # Checklist

# Type 'x' to check. Then press Shift+Enter.

# - [x]  Jupyter Notebook is open
# - [ ]  Code is error free
# - [ ]  The cells are arranged in order of logic and execution
# - [ ]  Task 1 has been performed
#     - [ ]  There is the procedure that can return k similar customers for a given one
#     - [ ]  The procedure is tested for all four proposed combinations
#     - [ ]  The questions re the scaling/distances are answered
# - [ ]  Task 2 has been performed
#     - [ ]  The random classification model is built and tested for all for probability levels
#     - [ ]  The kNN classification model is built and tested for both the original data and the scaled one, the F1 metric is calculated.
# - [ ]  Task 3 has been performed
#     - [ ]  The linear tegression solution is implemented with matrix operations.
#     - [ ]  RMSE is calculated for the implemented solution.
# - [ ]  Task 4 has been performed
#     - [ ]  The data is obfuscated with a random and invertible matrix P
#     - [ ]  The obfuscated data is recoved, few examples are printed out
#     - [ ]  The analytical proof that the transformation does not affect RMSE is provided 
#     - [ ]  The computational proof that the transformation does not affect RMSE is provided
# - [ ]  Conclusions have been made

# # Appendices 
# 
# ## Appendix A: Writing Formulas in Jupyter Notebooks

# You can write formulas in your Jupyter Notebook in a markup language provided by a high-quality publishing system called $\LaTeX$ (pronounced "Lah-tech"), and they will look like formulas in textbooks.
# 
# To put a formula in a text, put the dollar sign (\\$) before and after the formula's text e.g. $\frac{1}{2} \times \frac{3}{2} = \frac{3}{4}$ or $y = x^2, x \ge 1$.
# 
# If a formula should be in its own paragraph, put the double dollar sign (\\$\\$) before and after the formula text e.g.
# 
# $$
# \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.
# $$
# 
# The markup language of [LaTeX](https://en.wikipedia.org/wiki/LaTeX) is very popular among people who use formulas in their articles, books and texts. It can be complex but its basics are easy. Check this two page [cheatsheet](http://tug.ctan.org/info/undergradmath/undergradmath.pdf) for learning how to compose the most common formulas.

# ## Appendix B: Properties of Matrices

# Matrices have many properties in Linear Algebra. A few of them are listed here which can help with the analytical proof in this project.

# <table>
# <tr>
# <td>Distributivity</td><td>$A(B+C)=AB+AC$</td>
# </tr>
# <tr>
# <td>Non-commutativity</td><td>$AB \neq BA$</td>
# </tr>
# <tr>
# <td>Associative property of multiplication</td><td>$(AB)C = A(BC)$</td>
# </tr>
# <tr>
# <td>Multiplicative identity property</td><td>$IA = AI = A$</td>
# </tr>
# <tr>
# <td></td><td>$A^{-1}A = AA^{-1} = I$
# </td>
# </tr>    
# <tr>
# <td></td><td>$(AB)^{-1} = B^{-1}A^{-1}$</td>
# </tr>    
# <tr>
# <td>Reversivity of the transpose of a product of matrices,</td><td>$(AB)^T = B^TA^T$</td>
# </tr>    
# </table>

# <div class="alert alert-info">
#   Thanks again, Dmitry, for the review and great suggestions! I think you've been my code reviewer now for the last few projects.
# </div>
# 
