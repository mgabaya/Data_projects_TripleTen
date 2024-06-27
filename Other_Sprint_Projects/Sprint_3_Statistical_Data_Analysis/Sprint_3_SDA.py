#!/usr/bin/env python
# coding: utf-8

# Hello Mark!
# 
# My name is Dmitry.  I'm glad to review your work today.
# I will mark your mistakes and give you some hints how it is possible to fix them. We are getting ready for real job, where your team leader/senior colleague will do exactly the same. Don't worry and study with pleasure! 
# 
# Below you will find my comments - **please do not move, modify or delete them**.
# 
# You can find my comments in green, yellow or red boxes like this:
# 
# <div class="alert alert-block alert-success">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# 
# Success. Everything is done succesfully.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# 
# Remarks. Some recommendations.
# </div>
# 
# <div class="alert alert-block alert-danger">
# 
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# 
# Needs fixing. The block requires some corrections. Work can't be accepted with the red comments.
# </div>
# 
# You can answer me by using this:
# 
# <div class="alert alert-block alert-info">
# <b>Student answer.</b> <a class="tocSkip"></a>
# 
# Text here.
# </div>

# # Which one is a better plan?
# 
# You work as an analyst for the telecom operator Megaline. The company offers its clients two prepaid plans, Surf and Ultimate. The commercial department wants to know which of the plans brings in more revenue in order to adjust the advertising budget.
# 
# You are going to carry out a preliminary analysis of the plans based on a relatively small client selection. You'll have the data on 500 Megaline clients: who the clients are, where they're from, which plan they use, and the number of calls they made and text messages they sent in 2018. Your job is to analyze the clients' behavior and determine which prepaid plan brings in more revenue.

# <!--[We've provided you with some commentary to guide your thinking as you complete this project. However, make sure to remove all the bracketed comments before submitting your project.]
# 
# [Before you dive into analyzing your data, explain for yourself the purpose of the project and actions you plan to take.]
# 
# [Please bear in mind that studying, amending, and analyzing data is an iterative process. It is normal to return to previous steps and correct/expand them to allow for further steps.]
# -->

# ## Initialization

# <b><u>Objective</u></b>
# 
# We are trying to ascertain which prepaid plan, Surf or Ultimate brings in more revenue in minutes, texts, and volume of data through the following steps:
# 
# 1. Preprocessing the data
# 2. Analyzing the cleaned up data (statistical measures and visualizations)
# 3. Hypothesis testing (average revenue is the same between plans; average revenue in the NY-NJ area is the same as other regions)

# <div class="alert alert-block alert-success">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# 
# Great start with an introduction!
# </div>

# In[1]:


# Loading all the libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math as mt
from math import factorial
from scipy import stats as st


# ## Load data

# Load in the five data sets from Megaline.

# In[2]:


# Load the data files into different DataFrames
df_calls = pd.read_csv('/datasets/megaline_calls.csv')
df_internet = pd.read_csv('/datasets/megaline_internet.csv')
df_messages = pd.read_csv('/datasets/megaline_messages.csv')
df_plans = pd.read_csv('/datasets/megaline_plans.csv')
df_users = pd.read_csv('/datasets/megaline_users.csv')


# ## Prepare the data

# <!--[The data for this project is split into several tables. Explore each one to get an initial understanding of the data. Do necessary corrections to each table if necessary.]
# -->
# We are going to preprocess the data by checking for null values, checking the columns for their data types, and looking for duplicates.

# In[3]:


# Looking into all the dataframes
print('CALLS')
print(df_calls.info())
print('\nINERNET')
print(df_internet.info())
print('\nMESSAGES')
print(df_messages.info())
print('\nPLANS')
print(df_plans.info())
print('\nUSERS')
print(df_users.info())


# In[4]:


# Showing a sample of each of the dataframes
print('CALLS\n', df_calls.sample(5))
print('\nINERNET\n', df_internet.sample(5))
print('\nMESSAGES\n', df_messages.sample(5))
print('\nPLANS\n', df_plans)
print('\nUSERS\n', df_users.sample(5))


# In[5]:


# Checking if all the plans match
print(df_users['plan'].unique())


# In[6]:


# Getting an idea of how a user appears in each of the data sets
print('CALLS\n', df_calls[df_calls['user_id'] == 1196])
print('\nINERNET\n', df_internet[df_internet['user_id'] == 1196])
print('\nMESSAGES\n', df_messages[df_messages['user_id'] == 1196])
print('\nUSERS\n', df_users[df_users['user_id'] == 1196])
# print(df_users[~df_users['churn_date'].isna()])


# In[7]:


# Checking for any duplicates and the number of unique users
print('CALLS\n', df_calls[df_calls.duplicated()])
print('\nINERNET\n', df_internet[df_internet.duplicated()])
print('\nMESSAGES\n', df_messages[df_messages.duplicated()])
print('\nUSERS\n', df_users[df_users.duplicated()])
print()
print(df_calls['user_id'].nunique())
print(df_internet['user_id'].nunique())
print(df_messages['user_id'].nunique())


# We checked that there aren't any obvious duplicates for each of the data sets and that the number of users that appeared in each part of the plan (calls, texts, and internet usage). 

# ## Plans

# In[8]:


# Print the general/summary information about the plans' DataFrame
print('PLANS')
print(df_plans.info())


# In[9]:


# Print a sample of data for plans
print('\nPLANS\n', df_plans)


# <!--[Describe what you see and notice in the general information and the printed data sample for the above price of data. Are there any issues (inappropriate data types, missing data etc) that may need further investigation and changes? How that can be fixed?]
# -->
# In the plans data set, there are no nulls, duplicates or data types needing to be changed. No adjustments were made to this data set yet.

# <div class="alert alert-block alert-success">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# 
# Nice initial review.
# </div>

#  

# ## Fix data

# <!--[Fix obvious issues with the data given the initial observations.]-->
# 
# We are going to add a gigabyte per month included column for a later calculation since we charge extra per gigabyte.

# In[10]:


df_plans['gb_per_month_included'] = df_plans['mb_per_month_included']/1024


# ## Enrich data

# <!--[Add additional factors to the data if you believe they might be useful.]
# -->
# We will check that the gb_per_month_included column was added correctly and ready for the calculations later.

# In[11]:


df_plans.info()
print(df_plans)


# ## Users

# In[12]:


# Print the general/summary information about the users' DataFrame
df_users.info()
print(df_users['plan'].unique())  # Making sure all users have a ultimate or surf plan


# In[13]:


# Print a sample of data for users
df_users.sample(5)
# print(df_users[df_users['user_id']==1025])


# <!--[Describe what you see and notice in the general information and the printed data sample for the above price of data. Are there any issues (inappropriate data types, missing data etc) that may need further investigation and changes? How that can be fixed?]
# -->
# We will leave the churn_date column with null values since these are the customers that are still active.

#  

# ### Fix Data

# <!--[Fix obvious issues with the data given the initial observations.]
# -->
# For the users data, we check to see if a user might have multiple accounts under their name, so we check for duplicated first and last names or duplicated IDs. We will also make sure the city column is a string type for an analysis we will perform later on.

# In[14]:


# Double checking just in case a user had multiple accounts
print('\nUSER NAMES\n', df_users[df_users[['first_name', 'last_name']].duplicated()].sort_values(by='last_name'))  
print('\nUSER IDS\n', df_users[df_users['user_id'].duplicated()])
# print(len(df_users['city'].unique()))


# In[15]:


# Converting city into a string data type
np.array_equal(df_users['city'], df_users['city'].astype('str'))  
df_users['city'] = df_users['city'].astype(str)
# Saving users from the NY-NJ area
ny_nj_users = df_users[(df_users['city'].str.contains('NY')) | (df_users['city'].str.contains('NJ'))]
print(ny_nj_users)
# NY86, NJ97, NY_NJ80 
# df_users.iloc[377]  df['col'].str.contains('partial_string')


# ### Enrich Data

# <!--[Add additional factors to the data if you believe they might be useful.]
# -->
# For comparisons and calculations later, we will create a column denoting which users are from the NY-NJ regions.

# In[16]:


# add region column for NY-NJ users
def ny_nj_region(city):
    if ('NY'in city) | ('NJ'in city): #city.str.contains('NJ'): (city.str.contains('NY')) | (city.str.contains('NJ')):
        return 'ny-nj'

# (df_users['city'].str.contains('NY')) | (df_users['city'].str.contains('NJ'))
    
df_users['region'] = df_users['city'].apply(ny_nj_region)
# print(ny_nj_region('New York-Newark-Jersey City, NY-PA MSA'))
df_users.info()


# ## Calls

# In[17]:


# Print the general/summary information about the calls' DataFrame
# df_calls['call_date'] = pd.to_datetime(df_calls['call_date'], format='%Y-%m-%d')
df_calls.info()


# In[18]:


# Print a sample of data for calls
df_calls.sample(5)


# <!--[Describe what you see and notice in the general information and the printed data sample for the above price of data. Are there any issues (inappropriate data types, missing data etc) that may need further investigation and changes? How that can be fixed?]
# -->
# We are going to change the date columns to the appropriate datetime type to be able to work with it later on.

#  

# ### Fix data

# <!--[Fix obvious issues with the data given the initial observations.]-->
# First, we have to make sure the date is formatted correctly.

# In[19]:


df_calls['call_date'] = pd.to_datetime(df_calls['call_date'], format='%Y-%m-%d')
df_calls.info()
# print(df_calls[df_calls.duplicated()])


# ### Enrich data

# <!--[Add additional factors to the data if you believe they might be useful.]
# -->
# In order to do some of our calculations and comparisons, a month column will be created to the calls data in which the number of the month will be extracted from the call_date column.

# In[20]:


df_calls['month'] = df_calls['call_date'].dt.month
print(df_calls.sample(10))
print(df_calls[df_calls['user_id']==1025])


# ## Messages

# In[21]:


# Print the general/summary information about the messages' DataFrame
df_messages.info()


# In[22]:


# Print a sample of data for messages
df_messages.sample(4)


# <!--[Describe what you see and notice in the general information and the printed data sample for the above price of data. Are there any issues (inappropriate data types, missing data etc) that may need further investigation and changes? How that can be fixed?]
# -->
# In order to do some of our calculations and comparisons, a month column will be created to the messages data in which the number of the month will be extracted from the message_date column.

#  

# ### Fix data

# <!--[Fix obvious issues with the data given the initial observations.]
# -->
# We first need to convert the message_date column into a datetime data type.

# In[23]:


df_messages['message_date'] = pd.to_datetime(df_messages['message_date'], format='%Y-%m-%d')
df_messages.info()


# ### Enrich data

# <!--[Add additional factors to the data if you believe they might be useful.]
# -->
# We add a month column to denote the month in which the message was sent. This will help for some calculations and comparisons later.

# In[24]:


df_messages['month'] = df_messages['message_date'].dt.month
print(df_messages.sample(10))


# ## Internet

# In[25]:


# Print the general/summary information about the internet DataFrame
df_internet.info()


# In[26]:


# Print a sample of data for the internet traffic
df_internet.sample(4)


# <!--[Describe what you see and notice in the general information and the printed data sample for the above price of data. Are there any issues (inappropriate data types, missing data etc) that may need further investigation and changes? How that can be fixed?]
# -->
# As of now, the session_date column is just and object type. We will convert that into a usable data type shortly.

#  

# ### Fix data

# <!--[Fix obvious issues with the data given the initial observations.]
# -->
# We want to reformat the session_date column into a datetime data type.

# In[27]:


df_internet['session_date'] = pd.to_datetime(df_internet['session_date'], format='%Y-%m-%d')
df_internet.info()


# ### Enrich data

# <!--[Add additional factors to the data if you believe they might be useful.]
# -->
# In order to do some comparisons later, we will add the month column to the internet data set.

# In[28]:


df_internet['month'] = df_internet['session_date'].dt.month
print(df_internet.sample(5))


# ## Study plan conditions

# <!--[It is critical to understand how the plans work, how users are charged based on their plan subscription. So, we suggest printing out the plan information to view their conditions once again.]-->
# We print out each of the plan's info to know how much is charged for each attribute of the plans.

# In[29]:


# Print out the plan conditions and make sure they are clear for you
print(df_plans)
print(df_calls.sample())


# <div class="alert alert-block alert-success">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# 
# Good job.
# </div>

# ## Aggregate data per user
# 
# <!--[Now, as the data is clean, aggregate data per user per period in order to have just one record per user per period. It should ease the further analysis a lot.]
# -->
# We will create new dataframes out of the pivot tables created by the user IDs, the month, and the counts or sums of each type of telecom piece of data.

# In[30]:


# Calculate the number of calls made by each user per month. Save the result.
calls_per_user = df_calls.groupby(['user_id', 'month'])['duration'].count().reset_index()
calls_per_user.columns = ['user_id', 'month', 'num_calls']  # inplace=True)
print(calls_per_user)
print(calls_per_user.head(12))
print(df_calls.groupby(['user_id', 'month'])['duration'].count().head(12))


# In[31]:


# Calculate the amount of minutes spent by each user per month. Save the result.
mins_per_user = df_calls.groupby(['user_id', 'month'])['duration'].sum().reset_index()
mins_per_user['duration'] = np.ceil(mins_per_user['duration']) # Rounding up the duration column since the plans charge even if there is a decimal amount
print(mins_per_user)
# type(mins_per_user)


# In[32]:


# Calculate the number of messages sent by each user per month. Save the result.
messages_per_user = df_messages.groupby(['user_id', 'month'])['id'].count().reset_index()
messages_per_user.columns = ['user_id', 'month', 'num_msgs']
print(messages_per_user)
# print(messages_per_user[messages_per_user['user_id']==1025])


# In[33]:


# Calculate the volume of internet traffic used by each user per month. Save the result.
gb_per_user = df_internet.groupby(['user_id', 'month'])['mb_used'].sum().reset_index()

gb_per_user['gb_used'] = gb_per_user['mb_used']/1024
gb_per_user['gb_used'] = np.ceil(gb_per_user['gb_used']) # Rounding up the gb_used column since the plans charge even if there is a decimal amount
print(gb_per_user)


# <div class="alert alert-block alert-success">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# 
# Well done!
# </div>

# <!--[Put the aggregate data together into one DataFrame so that one record in it would represent what an unique user consumed in a given month.]
# -->
# For each user and each of their months, we will make a new dataframe using an outer merge. Nulls should fill months that might have one or more pieces of info missing since not every customer uses each part of their plan. We will fill the nulls in with appropriate info what is needed for calculation later on.

# In[34]:


# Merge the data for calls, minutes, messages, internet based on user_id and month
# user_month_merge = calls_per_user, mins_per_user, messages_per_user, gb_per_user
calls_merge = calls_per_user.merge(mins_per_user, on=['user_id', 'month'], how='outer')
print(calls_merge)
calls_msg_merge = calls_merge.merge(messages_per_user, on=['user_id', 'month'], how='outer')
print(calls_msg_merge.info())
# print(calls_msg_merge[calls_msg_merge['user_id']==1499])
calls_msg_gb_merge = calls_msg_merge.merge(gb_per_user, on=['user_id', 'month'], how='outer')
print(calls_msg_gb_merge)


# In[35]:


# Add the plan information
user_month_merge = df_users.merge(calls_msg_gb_merge, on='user_id', how='outer')
# print(user_month_merge.info())
user_month_merged = user_month_merge[~user_month_merge['month'].isna()].reset_index(drop=True)  # removing the 10 user_ids that don't have any data
print(user_month_merged.info())
user_month_merged = user_month_merged.drop(['first_name', 'last_name', 'age', 'city', 'reg_date', 'churn_date'], axis='columns')  # Removing unnecessary columns
user_month_merged['region'] = user_month_merged['region'].fillna('')  # Filling in null values with empty string
user_month_merged[['num_calls', 'duration', 'num_msgs', 'mb_used', 'gb_used']] = user_month_merged[['num_calls', 'duration', 'num_msgs', 'mb_used', 'gb_used']].fillna(0)  # Filling in null values with 0
print(user_month_merged.info())
print(user_month_merged)


# <div class="alert alert-block alert-success">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# 
# Correct.
# </div>

# <!--[Calculate the monthly revenue from each user (subtract the free package limit from the total number of calls, text messages, and data; multiply the result by the calling plan value; add the monthly charge depending on the calling plan). N.B. This might not be as trivial as just a couple of lines given the plan conditions! So, it's okay to spend some time on it.]
# -->
# Now, we will create a function that will calculate the revenue for each user and each of their months. This will be added as another row to the end of the user_month_merged data set.

# In[36]:


# Calculate the monthly revenue for each user
# Creating a function to call into the apply() method to create a new revenue column
def compute_monthly_bill(row):
    plan = row['plan']
    mins = row['duration']
    texts = row['num_msgs']
    gb = row['gb_used']
    cost = 0
    if plan == 'surf':
        cost += 20
        if mins > 500:
            cost += (mins - 500)*.03
            if texts > 50:
                cost += (texts - 50) * .03
                if gb > 15:
                    cost += (gb - 15) * 10
            else:
                if gb > 15:
                    cost += (gb - 15) * 10
        else:
            if texts > 50:
                cost += (texts - 50) * .03
                if gb > 15:
                    cost += (gb - 15) * 10
            else:
                if gb > 15:
                    cost += (gb - 15) * 10
    else:  # plan == ultimate
        cost += 70
        if mins > 3000:
            cost += (mins - 3000) * .01
            if texts > 1000:
                cost += (texts - 1000) * .01
                if gb > 30:
                    cost += (gb - 30) * 7
            else:
                if gb > 30:
                    cost += (gb - 30) * 7
        else:
            if texts > 1000:
                cost += (texts - 1000) * .01
                if gb > 30:
                    cost += (gb - 30) * 7
            else:
                if gb > 30:
                    cost += (gb - 30) * 7
    return cost

print(compute_monthly_bill(user_month_merged.iloc[2292]))  # testing a single row


# <div class="alert alert-block alert-warning">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# 
# Good function, but it's not a good practice to hardcode values. It's better to use variables/links. In our case we can get all information from plan df.
# </div>

# In[37]:


user_month_merged['revenue'] = user_month_merged.apply(compute_monthly_bill, axis=1)
print(user_month_merged.info())
print(user_month_merged)


# Now, we have compiled a data set that gives each user's plan usage among the three aspects of their plans and the revenue from each month. This will allow us to compare between the Surf and Ultimate plans.

# ## Study user behaviour

# <!--[Calculate some useful descriptive statistics for the aggregated and merged data, which typically reveal an overall picture captured by the data. Draw useful plots to help the understanding. Given that the main task is to compare the plans and decide on which one is more profitable, the statistics and the plots should be calculated on a per-plan basis.]
# 
# [There are relevant hints in the comments for Calls but they are not provided for Messages and Internet though the principle of statistical study is the same for them as for Calls.]
# -->
# In this section, we will be creating descriptive statistics and visualizations to help compare the similarities and differences between the Surf and Ultimate plans. We will individually looks at the four parts of each plan: call duration, text message frequency, internet usage, and revenue.

# ### Calls

# In[38]:


# Compare average duration of calls per each plan per each distinct month. Plot a bar plot to visualize it.
df_plan_month_calls = user_month_merged.groupby(['plan', 'month'])['duration'].mean()  #.reset_index()
print(df_plan_month_calls)
# Bar plot
df_plan_month_calls['surf'].plot(
    kind='bar',
    title='How Long Do Surf and Ultimate Users Talk?',
    ylabel='Average Duration in minutes',
    figsize=[8,6], 
    color='blue',
    alpha=.6
)
df_plan_month_calls['ultimate'].plot(
    kind='bar', 
    color='yellow',
    alpha=.5
)
plt.xlabel('Months')
plt.legend(['Surf', 'Ultimate'])
plt.show()


# In[39]:


# Compare the number of minutes users of each plan require each month. Plot a histogram.
# print(user_month_merged.groupby('plan')['duration'])
user_month_merged.groupby('plan')['duration'].plot(
    kind='hist',
    bins=20,
    title='How Long Do Users Talk on Their Phones?',
    figsize=[8,6]
)
plt.legend(title='Plans')
plt.xlabel('Minutes Per Call')
plt.show()


# <!--[Calculate the mean and the variable of the call duration to reason on whether users on the different plans have different behaviours for their calls.]
# -->
# To further examine each plan, we will calculate some descriptive statistics. 

# In[40]:


# Calculate the mean and the variance of the monthly call duration
print('Average Call Duration:\n', user_month_merged.groupby('plan')['duration'].mean())
print('\nCall Duration Variance:\n', user_month_merged.groupby('plan')['duration'].var())
print('\nCall Duration Standard Deviation:\n', user_month_merged.groupby('plan')['duration'].std())
print('\nCall Duration Descriptive Statistics:\n', user_month_merged.groupby('plan')['duration'].describe())
#print(user_month_merged.groupby('plan')['duration'].sum())


# In[41]:


# Plot a boxplot to visualize the distribution of the monthly call duration

user_month_merged.boxplot(
    column='duration', 
    by='plan',
    figsize=[10,5],
    vert = 0
)

plt.title('Surf vs Ultimate Call Length')
plt.ylabel('Plan')
plt.xlabel('Call Minutes')
plt.show()


# <!--[Formulate conclusions on how the users behave in terms of calling. Is their behaviour different between the plans?]
# -->

# Overall, the call duration between the plans are very similar, so the customer's behaviors regarding call duration tend to be the same. For further examination, we looked at the following:
# 
# * The Bar Graphs showed similar length of calls for most of the months except for February, which the Ultimate users had about 100 minutes more on average, and June, where Surf users had about 60 minutes more.
# * The Histogram of both plans showed a similar, right tailed shape that centered around 400 minutes. However, there are more Surf users than Ultimate users, so the distribution for the Ultimate users was shorter.
# * With the Descriptive Statistics, we see that the means, standard deviation, and quartiles are close in values, so user behavior with call length tends to be the same. 
# * As seen with the similar quartiles, the Boxplot for both plans have very similar shapes, with the IQR in the same area and a longer upper (or right) tail, which reinforces what the histogram shows. So, again, the users call length are similar between the plans.

# ### Messages

# In[42]:


# Compare the number of messages users of each plan tend to send each month
df_plan_month_msgs = user_month_merged.groupby(['plan', 'month'])['num_msgs'].mean()
print(df_plan_month_msgs)

df_plan_month_msgs['surf'].plot(
    kind='bar',
    title='How Much Do Surf and Ultimate Users Text?',
    ylabel='Number of Texts Sent per Month',
    figsize=[8,6], 
    color='blue',
    alpha=.6
)
df_plan_month_msgs['ultimate'].plot(
    kind='bar', 
    color='yellow',
    alpha=.5
)
plt.xlabel('Months')
plt.legend(['Surf', 'Ultimate'])
plt.show()


# In[60]:


# Compare the number of texts users of each plan require each month. Plot a histogram.
user_month_merged.groupby('plan')['num_msgs'].plot(
    kind='hist',
    bins=20,
    title='Who Texts More?',
    figsize=[8,6]
)
plt.legend(title='Plans')
plt.xlabel('Number of Texts Sent in a Month')
plt.show()


# In[44]:


# Calculate the mean and the variance of the monthly texts sent
print(user_month_merged.groupby('plan')['num_msgs'].mean())
print(user_month_merged.groupby('plan')['num_msgs'].var())
print(user_month_merged.groupby('plan')['num_msgs'].std())
print(user_month_merged.groupby('plan')['num_msgs'].describe())


# In[45]:


# Plot a boxplot to visualize the distribution of the monthly texts sent

user_month_merged.boxplot(
    column='num_msgs', 
    by='plan',
    figsize=[10,5],
    vert = 0
)

plt.title('Surf vs Ultimate Texting')
plt.ylabel('Plan')
plt.xlabel('Number of Texts')
plt.show()


# <!--[Formulate conclusions on how the users behave in terms of messaging. Is their behaviour different between the plans?]-->

# Overall, the number of text sent in a month tended to be a little higher for the Ultimate users. Since the Ultimate plan has a higher allotment of texts, this could play a large part for why it was higher than Surf users. We see this in the following:
# 
# * The Bar Graphs showed that every month the Ultimate users outpaced the Surf users by at least five texts.
# * Although the shapes of the Histograms seemed similar, the Surf users had a bulk of users below 50 texts, which is probably due to their text limit. For the Ultimate users, their values look more spread out with only a peak towards zero texts. 
# * When looking at the Descriptive Statistics, we see that the mean and quartiles are higher in the Ultimate users.
# * In the Boxplot, we see the IQR of the Ultimate users shifted higher than the Surf users. 

# ### Internet

# In[61]:


# Compare the amount of internet traffic consumed by users of each plan tend to send each month
df_plan_month_internet = user_month_merged.groupby(['plan', 'month'])['gb_used'].mean()
print(df_plan_month_internet)

df_plan_month_internet['surf'].plot(
    kind='bar',
    title='How Much Internet Traffic Do Surf and Ultimate Users Consume?',
    ylabel='GB of Data per Month',
    figsize=[9,7], 
    color='blue',
    alpha=.6
)
df_plan_month_internet['ultimate'].plot(
    kind='bar', 
    color='yellow',
    alpha=.5
)
plt.xlabel('Months')
plt.legend(['Surf', 'Ultimate'])
plt.show()


# In[59]:


# Compare the number of internet traffic users of each plan require each month. Plot a histogram.
user_month_merged.groupby('plan')['gb_used'].plot(
    kind='hist',
    bins=20,
    title='Surf vs Ultimate Internet Usage',
    figsize = [8,6]
)
plt.legend(title='Plans')
plt.xlabel('Gigabytes Used')
plt.show()


# In[48]:


# Calculate the mean and the variance of the monthly internet usage
print('Internet Usage Averages per Plan:\n', user_month_merged.groupby('plan')['gb_used'].mean())
print('\nInternet Usage Variance:\n', user_month_merged.groupby('plan')['gb_used'].var())
print('\nInternet Usage Standard Deviation:\n', user_month_merged.groupby('plan')['gb_used'].std())
print('\nInternet Usage Descriptive Statistics:\n', user_month_merged.groupby('plan')['gb_used'].describe())


# In[49]:


# Plot a boxplot to visualize the distribution of the monthly call duration
user_month_merged.boxplot(
    column='gb_used', 
    by='plan',
    figsize=[10,5],
    vert = 0
)

plt.title('Surf vs Ultimate Internet Usage')
plt.ylabel('Plan')
plt.xlabel('Gigabytes Downloaded')
plt.show()


# <!--[Formulate conclusions on how the users tend to consume the internet traffic? Is their behaviour different between the plans?]-->

# After examining the internet useage between plans, there were both similarities and differences as shown below:
# 
# * When we plotted the Bar Graphs per month, we noticed that Ultimate users most often had higher internet usage except for June and July, but only by a little. It should also be noted that the Ultimate plan has a higher allotment for gigabytes, which may have an affect on the users of that plan to consume more internet.
# * The Histogram of both plans showed a similar shape centering their peaks close together. But, since there are more Surf users than Ultimate users, the distribution for the Surf users was higher.
# * The Descriptive Statistics showed that the means, standard deviation, and quartiles are close in values, so user behavior with internet usage tends to be the same.
# * As seen with the similar quartiles, the Boxplot for both plans have very similar shapes, with the IQRs in generally the same area and outliers far to the right. But, there were higher outliers for some Surf plan users, which would have created more revenue in the end.

# ## Revenue

# <!--[Likewise you have studied the user behaviour, statistically describe the revenue between the plans.]-->
# Next, we will take a look at the behavior of the users through the revenue they bring in.

# In[50]:


df_plan_month_revenue = user_month_merged.groupby(['plan', 'month'])['revenue'].mean()
# df_plan_month_revenue = user_month_merged.groupby(['plan', 'month'])['revenue'].sum()
print(df_plan_month_revenue)

df_plan_month_revenue['surf'].plot(
    kind='bar',
    title='Revenue from Surf and Ultimate Users',
    ylabel='Revenue in $USD',
    figsize=[10,8], 
    color='blue',
    alpha=.6
)
df_plan_month_revenue['ultimate'].plot(
    kind='bar', 
    color='yellow',
    alpha=.5
)
plt.xlabel('Months')
plt.legend(['Surf', 'Ultimate'])
plt.show()


# In[58]:


# Compare the average revenue users of each plan provide each month. Plot a histogram.
user_month_merged.groupby('plan')['revenue'].plot(
    kind='hist',
    bins=[20, 35, 50, 65, 80, 100],  
    title='How Much Revenue Do Users Generate?', 
    alpha=.6
)
plt.legend(title='Plans')
plt.xlabel('Revenue in $USD')
plt.show()


# In[52]:


# Calculate the mean and the variance of the monthly revenue
print('Revenue Averages per Plan:\n', user_month_merged.groupby('plan')['revenue'].mean())
print('\nVariance per Plan:\n', user_month_merged.groupby('plan')['revenue'].var())
print('\nStandard Deviation per Plan:\n', user_month_merged.groupby('plan')['revenue'].std())
print('\nDescriptive Statistics per Plan:\n', user_month_merged.groupby('plan')['revenue'].describe())
print('\nTotal Revenue per Plan:\n', user_month_merged.groupby('plan')['revenue'].sum())


# In[57]:


# Plot a boxplot to visualize the distribution of the monthly call duration
user_month_merged.boxplot(
    column='revenue', 
    by='plan',
    figsize=[10,5],
    vert = 0
)

plt.title('Surf vs Ultimate User Revenue')
plt.ylabel('Plan')
plt.xlabel('Revenue in $USD')
plt.show()


# <!--[Formulate conclusions about how the revenue differs between the plans.]-->

# As we take a closer look at the revenue generated by users per month, we see very distinct differences. For further examination, we looked at the following:
# 
# * The Bar Graphs showed clearly showed the difference in revenue for all the months except December with at least a \\$10 difference. The caveat to this is that the base price for the Surf plan starts at \\$20 and the Ultimate plan starts at \\$70, so Surf users would have to exceed their default parameters by quite a bit.
# * The Histogram of both plans did not have the same shape and the peaks were not together with the Surf users peaking on the lower end while the Ultimate users were largely around \\$70. 
# * With the Descriptive Statistics, we see all the values differ. Again, the entry point to each plan might have a large effect on these values. Also, when we summed the revenue for each plan, the Surf generated more, but they have more than double the amount of users.
# * The Boxplot reinforced what the previous visualizations showed, which is there is a difference in the average revenue per user per month. The Surf users tended to create revenue from \\$20 to \\$80, while Ultimate users heavily stayed around \\$70. The latter's entire IQR is at \\$70 and not very spread out.

# <div class="alert alert-block alert-success">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# 
# Nice vizualisation and good conclusions!
# </div>

# <div class="alert alert-block alert-success">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# 
# Pro tip: we can add boxplot without outliers to get additional insights.
# </div>

# ## Test statistical hypotheses

# <!--[Test the hypothesis that the average revenue from users of the Ultimate and Surf calling plans differs.]
# -->
# We are going to test to see if there is a difference in the average revenue from users of the Ultimate and Surf plans by performing a T-test on the means of the two statistical populations are equal. 
# The statistical significance level will be 5%.

# <!--[Formulate the null and the alternative hypotheses, choose the statistical test, decide on the alpha value.]-->
# <b>H<sub>0</sub></b>: The average revenue is the same between plans.
# 
# <b>H<sub>1</sub></b>: The average revenue is different between plans.
# 

# In[54]:


# Test the hypotheses
surf_rev = user_month_merged[user_month_merged['plan']=='surf']['revenue']  # .mean()
ultimate_rev = user_month_merged[user_month_merged['plan']=='ultimate']['revenue']  # .mean()
print(surf_rev.std())
print(ultimate_rev.mean())

alpha = .05

results = st.ttest_ind(surf_rev, ultimate_rev, equal_var=False)
print('p-value:', results.pvalue)

if (results.pvalue < alpha):
    print("We reject the null hypothesis")
else:
    print("We can't reject the null hypothesis")


# <!--[Test the hypothesis that the average revenue from users in the NY-NJ area is different from that of the users from the other regions.]
# -->
# We are going to test to see if there is a difference in the average revenue from users in the NY-NJ area and other regions by performing a T-test on the equality of two population means.
# The statistical significance level will be 5%.

# <!--[Formulate the null and the alternative hypotheses, choose the statistical test, decide on the alpha value.]
# -->
# <b>H<sub>0</sub></b>: The average revenue is the same between the NY-NJ area and other regions.
# 
# <b>H<sub>1</sub></b>: The average revenue is different among the regions.

# In[55]:


# Test the hypotheses
ny_nj_rev = user_month_merged[user_month_merged['region']=='ny-nj']['revenue']  # .mean()
others_rev = user_month_merged[user_month_merged['region']=='']['revenue']  # .mean()
# print(ny_nj_rev)
# print(others_rev)
alpha = .05

results = st.ttest_ind(ny_nj_rev, others_rev, equal_var=False)
print('p-value:', results.pvalue)

if (results.pvalue < alpha):
    print("We reject the null hypothesis")
else:
    print("We can't reject the null hypothesis")


# <div class="alert alert-block alert-success">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# 
# Excellent.
# </div>

# ## General conclusion
# 
# <!--[List your important conclusions in this final section, make sure they cover all those important decisions (assumptions) that you've made and that led you to the way you processed and analyzed the data.]
# -->
# After gathering and compiling all the data across multiple data set for Megaline, our samples showed the following about the users from the Surf and Ultimate plans:
# 1. Call Duration: The users' behavior was very similar between the plans. A possibility is that more people text nowadays rather than calling.
# 2. Text Messaging: We saw that because the alloted number of texts is higher for the Ultimate plan, their users tended to text at an increased rate.
# 3. Internet Usage: While there were differences depending on the month, the internet usage behavior had some similarites with where users centered around in terms of gigabytes used.
# 4. Revenue: The comparison made between the revenue from Surf and Ultimate plans showed very different behaviors. A common thread was the initial fee each plan charged; Surf at \\$20 and Ultimate at \\$70. Because of this, the revenue from each plan tended towards their respective initial fee.
# 5. When we tested to see if there is a difference in the average revenue from users of the Ultimate and Surf plans, we rejected the null hypothesis given the significance level selected. This leads us to conclude that our assumption that the average revenue is the same between plans is most likely incorrect. This was further supported when we calculated the descriptive statistics and graphs on just the revenue per user per month. 
# 6. When we tested to see if the average revenue is the same between the NY-NJ area and other regions, we got that about 10% of the time, the average revenue from both sets of regions would be different. Thus, we could not reject the null hypothesis. 

# <div class="alert alert-block alert-success">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# 
# Great final conclusion.
# </div>

# <div class="alert alert-block alert-success">
# <b>Overall reviewer's comment</b> <a class="tocSkip"></a>
# 
# Mark, thank you for sending your project. You've done a really good job on it!
#     
# Especially impressed:
# 
# - very deep detailed conclusions
#     
# - good graphs
#     
# - high code level
# 
# - good project structure
#     
#     
# Thank you for in-depth analysis and logical conclusions!
#     
# I'm glad to say that your project has been accepted. Keep up the good work, and good luck on the next sprint!
# </div>

#  
