#!/usr/bin/env python
# coding: utf-8

# <div style="border:solid blue 2px; padding: 20px">
#   
# **Hello Mark**
# 
# My name is Dima, and I will be reviewing your project. 
# 
# You will find my comments in coloured cells marked as 'Reviewer's comment'. The cell colour will vary based on the contents - I am explaining it further below. 
# 
# **Note:** Please do not remove or change my comments - they will help me in my future reviews and will make the process smoother for both of us. 
# 
# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment</b> 
#     
# Such comment will mark efficient solutions and good ideas that can be used in other projects.
# </div>
# 
# <div class="alert alert-warning"; style="border-left: 7px solid gold">
# <b>⚠️ Reviewer's comment</b> 
#     
# The parts marked with yellow comments indicate that there is room for optimisation. Though the correction is not necessary it is good if you implement it.
# </div>
# 
# <div class="alert alert-danger"; style="border-left: 7px solid red">
# <b>⛔️ Reviewer's comment</b> 
#     
# If you see such a comment, it means that there is a problem that needs to be fixed. Please note that I won't be able to accept your project until the issue is resolved.
# </div>
# 
# You are also very welcome to leave your comments / describe the corrections you've done / ask me questions, marking them with a different colour. You can use the example below: 
# 
# <div class="alert alert-info"; style="border-left: 7px solid blue">
# <b>Student's comment</b>

# # Introduction
# 
# The data to be examined and analyzed was collected and compiled using SQL on several databases that included information regarding individual cab trips in Chicago, the companies each cab belongs to, the neighborhoods the cabs frequent, and the weather conditions on each hour of the days.
# 
# Objective:
# 
# In this project, we will be analyzing data for a rideshare company, Zuber, in the Chicago area.
# We are tasked to find patterns in the available data in order to understand passenger preferences and the impact of external factors on rides, like weather conditions and ride duration.

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Title and introduction are specified, good job
# </div>

# ## Process the Data
# 
# To begin, we will first perform the following on the data:
# - Load the necessary libraries
# - Import the files into Dataframes
# - Study the data they contain: 
#     - Display the data
#     - View the info
#     - Check the unique values
#     - Check for nulls
#     - Check for duplicates
#     - Make sure the data types are correct
# 

# In[1]:


# Load the necessary libraries
import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math as mt
from scipy import stats as st
import plotly.express as px


# In[2]:


# Import the files into Dataframes by reading in the csv's
df_company_trips = pd.read_csv('/datasets/project_sql_result_01.csv')
df_neighborhood_trips = pd.read_csv('/datasets/project_sql_result_04.csv')
df_loop_ohare = pd.read_csv('/datasets/project_sql_result_07.csv')


# ### Display Data

# Here are the descriptions of the columns of each of the dataframes that are displayed below:
# 
# df_company_trips:
# - company_name: taxi company name
# - trips_amount: the number of rides for each taxi company on November 15-16, 2017. 
# 
# df_neighborhood_trips:
# - dropoff_location_name: Chicago neighborhoods where rides ended
# - average_trips: the average number of rides that ended in each neighborhood in November 2017. 
# 
# df_loop_ohare: (Loop to O'Hare International Airport data)
# - start_ts: pickup date and time
# - weather_conditions: weather conditions at the moment the ride started, Good = non-rainy days; Bad = rainy days
# - duration_seconds: ride duration in seconds

# In[3]:


display(df_company_trips)


# In[4]:


display(df_neighborhood_trips)


# In[5]:


display(df_loop_ohare)


# ### Data Info

# In[6]:


df_company_trips.info()
print()
df_neighborhood_trips.info()
print()
df_loop_ohare.info()


# ### Uniqueness

# In[7]:


# Check uniqueness
print(df_company_trips['company_name'].nunique())
print(df_company_trips['company_name'].sort_values().unique())


# In[8]:


print(df_neighborhood_trips['dropoff_location_name'].nunique())
print(df_neighborhood_trips['dropoff_location_name'].sort_values().unique())


# In[9]:


# start_ts weather_conditions
print(df_loop_ohare['start_ts'].nunique())
print(df_loop_ohare['start_ts'].sort_values().unique())


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
# 
# The unique values in the data were checked, and we also looked at the info() method

# For the first two dataframes, there are a complete set of unique neighborhoods and cab names. When looking at the number of unique start times, it makes sense that there would be multiple records for each start time. We will look at possible duplicates in a later section.  

# ### Null Values

# In[10]:


# Nulls
print(df_company_trips.isna().sum())
print()
print(df_neighborhood_trips.isna().sum())
print()
print(df_loop_ohare.isna().sum())


# There are no null values we need to handle or fill.

# ### Duplicates

# In[11]:


# Duplicates
#print('\nUSERS\n', df_users[df_users.duplicated()])
print(df_company_trips[df_company_trips.duplicated()])
print(df_neighborhood_trips[df_neighborhood_trips.duplicated()])
print(df_loop_ohare[df_loop_ohare.duplicated()].head(14))


# The third dataframe on the weather status and ride duration has duplicates that we will need to examine and decide whether to keep or drop the rows.

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Data by missing values and duplicates was checked, good job
# </div>

# In[12]:


# Check a duplicate record
df_loop_ohare[(df_loop_ohare['start_ts'] == '2017-11-11 06:00:00') & (df_loop_ohare['duration_seconds'] == 1260)]


# We are unsure about the nature of these duplicates. The data does not show the minutes of the times, which are rounded to the nearest hour. Additionally, we don't know where each record of a trip started and ended. It is possible that multiple cabs picked up and dropped off in the same places within the same starting hour. So, the records could actually be distinct. For this possibility, we are not going to drop these rows even though they look like duplicates.

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
# 
# I agree with you, this data can be left

# ### Fix Data Types
# 
# For analysis purposes, we will change the start time column to a datetime format.

# In[13]:


# Data Types
df_loop_ohare['start_ts'] = pd.to_datetime(df_loop_ohare['start_ts'])
df_loop_ohare.info()
# df_calls['call_date'] = pd.to_datetime(df_calls['call_date'], format='%Y-%m-%d')


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# You also did not forget about the conversion of data types in dates</div>

# ## Data Analysis
# 
# In this section, we will analyze the data on neighborhood trips by doing the following:
# - Identify the top 10 neighborhoods in terms of drop-offs
# - Make graphs: taxi companies and number of rides, top 10 neighborhoods by number of dropoffs
# - Draw conclusions based on each graph and explain the results

# In[14]:


# Top 10 neighborhoods
top_10n = df_neighborhood_trips.sort_values('average_trips', ascending=False).head(10)
display(top_10n)


# In[15]:


#  top 10 neighborhoods by number of dropoffs
df_neighborhood_trips
top_10n
px_neighborhoods = px.bar(top_10n, x='dropoff_location_name', y='average_trips') #, color='genre', barmode="overlay", nbins=20) 
#, hover_data=['type']) # color='make'
px_neighborhoods.update_layout(
    title_text='Most Frequented Chicago Neighborhoods', # title of plot
    xaxis_title_text='Average Number of Trips', # xaxis label
    yaxis_title_text='Chicago Neighborhoods', # yaxis label
)
px_neighborhoods.show()


# The Loop is by far the most traveled neighborhood by cab. River North, Streeterville, and West Loop round out the top four neighborhoods. After those, there is a tapering off in the number of average trips made in the other neighborhoods by less than half of the top four locations.

# In[16]:


#  taxi companies and number of rides
df_company_trips
px_company = px.bar(df_company_trips.head(10), x='company_name', y='trips_amount') #, x='global_sales', color='genre', barmode="overlay", nbins=20) #, hover_data=['type']) # color='make'
px_company.update_layout(
    title_text='Chicago\'s Most Used Cab Companies', # title of plot
    xaxis_title_text='Cab Companies', # xaxis label
    yaxis_title_text='Number of Trips ', # yaxis label
    xaxis_tickangle=90
)
px_company.show()


# In[17]:


px_company = px.bar(df_company_trips, x='company_name', y='trips_amount') #, x='global_sales', color='genre', barmode="overlay", nbins=20) #, hover_data=['type']) # color='make'
px_company.update_layout(
    title_text='All Chicago\'s Cab Companies', # title of plot
    xaxis_title_text='Cab Companies', # xaxis label
    yaxis_title_text='Number of Trips ', # yaxis label
    xaxis_tickangle=90
)
px_company.show()


# The first graph shows the top 10 most used cab companies, while the second graph shows how the number of trips decreases among all the cab companies. We notice that many of the companies to the right have number prefixes in their names. This will be examined next.

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Great! We counted the data from files by dropoff_location_name and company_name, sorted graphs, applied a single color and indicated small outputs</div>

# <div class="alert alert-warning"; style="border-left: 7px solid gold">
# <b>⚠️ Reviewer's comment, v. 1</b> 
#     
# Also here you can swap the x and y axes in places so that the names are more readable

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Also I could recommend you this site, may be you could find smth interesting for visualization skills:
# 
# https://www.python-graph-gallery.com/

# In[18]:


# Extract the rows with a 4-digit prefix
num_in_name = df_company_trips['company_name'].str.extract(r'(\d\d\d\d).*')
print(num_in_name[~num_in_name[0].isna()])


# In[19]:


df_company_trips.tail(30)


# Some cab company names contain a 4-digit prefix in their names. We notice that these make up the bottom half of trips among all the cab companies. These companies also pertain to a person's name, so it makes sense that there would be fewer trips since there is probably only one car for these cab companies.

# In[20]:


display(df_company_trips.describe())
display(df_neighborhood_trips.describe())
display(df_loop_ohare.describe())


# There is a trend among the cab companies where the companies with a generalized taxi service rank among the most used companies whereas the companies that have a person's name rank on the lower end.
# 
# The neighborhoods connected to the Loop are some of the most frequented neighborhoods with Loop and West Loop. River North and Streeterville are the other most traveled by cabs.

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
# 
# Brilliant! Absolutely all the necessary tasks in this section have been completed, as well as very clear and comprehensive conclusions

# ## Hypothesis Testing
# 
# In this section, we will test the hypothesis:
# 
# "The average duration of rides from the Loop to O'Hare International Airport changes on rainy Saturdays." 
# 
# <!--Decide where to set the significance level (alpha) on your own.
# 
# Explain:
# - how you formed the null and alternative hypotheses
# - what criterion you used to test the hypotheses and why-->
# 
# We will examine this by performing a T-test on the means of the two statistical populations to see if the means are equal since the populations are different. The statistical significance level will be 5%.
# 
# <b>H<sub>0</sub></b>: The average duration of rides from the Loop to O'Hare International Airport is the same on rainy Saturdays as non-rainy Saturdays.
# 
# <b>H<sub>1</sub></b>: The average duration of rides from the Loop to O'Hare International Airport is different on rainy Saturdays as compared with non-rainy Saturdays.

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Hypotheses are formulated</div>

# In[21]:


# Test the hypotheses

df_loop_ohare_rain = df_loop_ohare[df_loop_ohare['weather_conditions'] == 'Bad']['duration_seconds']
df_loop_ohare_no_rain = df_loop_ohare[df_loop_ohare['weather_conditions'] == 'Good']['duration_seconds']
print(df_loop_ohare_rain.mean()) #['duration_seconds'].describe())
print(df_loop_ohare_no_rain.mean()) # ['duration_seconds'].describe())
alpha = .05

results = st.ttest_ind(df_loop_ohare_rain, df_loop_ohare_no_rain, equal_var=False)
print('p-value:', results.pvalue)

if (results.pvalue < alpha):
    print("We reject the null hypothesis")
else:
    print("We can't reject the null hypothesis")


# Since the p-value of our test was so small, we reject the null hypothesis. Thus, this evidence shows that the average duration of rides from the Loop to O'Hare International Airport changes on rainy Saturdays.

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b>
# 
# The entire output is formed perfectly

# ## General conclusion
# 
# Through our analysis and testing in the recent years, we observed the following:
# 
# - The Flash Cab company is the most popular cab company in Chicago.
# - Companies named after a person tend to have fewer trips compared to city or organization run services. 
# - The Loop area, River North, and Streeterville make up a large portion of where cab trips happen amongst the Chicago neighborhoods.
# - There is strong evidence that the length of a cab ride on a rainy Saturday in November is different than on non-rainy Saturdays, most likely increasing the duration of the ride.
# 
# Using these findings, Zuber can make a strategic plan to cater their rides on the aforementioned neighborhoods while being wary of the weather conditions and who the greatest competitors are to their business.

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b>
#   
# The general conclusion is formulated very nice! Each step of the study is described in detail and a conclusion on hypothesis testing is added

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>Review summary</b> 
#     
# Mark, the project is great! You have very strong analytical skills, visualization skills and understanding of statistical methods. There are no issues that need to be fixed, so I can accept the project now. Awesome job!
# 
# Thank you for your work and I wish you success in your next projects!) 😊
