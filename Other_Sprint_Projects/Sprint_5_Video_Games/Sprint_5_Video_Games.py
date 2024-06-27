#!/usr/bin/env python
# coding: utf-8

# <div style="border-radius: 15px; border: 3px solid indigo; padding: 15px;">
# <b> Reviewer's comment</b>
#     
# Hello, my name is Sveta Nosova and I am going to review this project. 
# 
# Before we start, I want to pay your attention to the color marking:
#     
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# Great solutions and ideas that can and should be used in the future are in green comments.   
# </div>    
#     
#     
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
# 
# Yellow color indicates what should be optimized. This is not necessary, but it will be great if you make changes to this project.
# </div>      
#     
#     
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
# 
# Issues that need to be corrected to get right results are indicated in red comments. Note that the project cannot be accepted until these issues are resolved.
# </div>    
# 
# <hr>
#     
# **Please, use some color other than those listed to highlight answers to my comments.**
# I would also ask you **not to change, move or delete my comments** so that it would be easier for me to navigate during the next review.
#     
# In addition, my comments are defined as headings. 
# They can mess up the content; however, they are convenient, since you can immediately go to them. I will remove the headings from my comments in the next review. 
#    
#     
#     
# <hr>
#     
# <font color='dodgerblue'>**A few words about the project:**</font> you did a good job, everything is clear and neat, there are  analysis and intermediate conclusions. I still have some questions that I've written in my comments. I've also left there some recommendations for improving the project.
#     
#     
# I will wait for the project for a second review :)
#     
#     
# 
# </div>

# <div class="alert alert-info"; style="border-left: 7px solid blue">
# <b>Student's comment</b>
#     I'll use this blue box for my comments. Thank you for reviewing my project! 
# </div>

# <div style="border-radius: 15px; border: 3px solid indigo; padding: 15px;">
# <b> Reviewer's comment 2</b>
# 
# 
# Thank you for the updates! I've left new comments titled as **Reviewer's comment 2**. Take a look, I've left a code example that you can use to fill in the gaps with the year values from `name` column.
#     
# 
# I don't have any questions, so the project can be accepted. Congratulations 😊
# 
# Good luck! 😊 
#     
#     
# <hr>
#     
#     
# Best regards,
#     
# S. N.    
# </div>

# # Video Game Ratings & Sales
# 
# We will be examining the data for the online store Ice, which has video game sales from different regions as well as the games' ratings across the years.
# 
# Goal:
# We will identify patterns that determine whether a game succeeds or not. This allows for spotting potential big winners and to plan advertising campaigns.
# 

# Data description
# - Name 
# - Platform 
# - Year_of_Release 
# - Genre 
# - NA_sales (North American sales in USD million) 
# - EU_sales (sales in Europe in USD million) 
# - JP_sales (sales in Japan in USD million) 
# - Other_sales (sales in other countries in USD million) 
# - Critic_Score (maximum of 100) 
# - User_Score (maximum of 10) 
# - Rating (ESRB): E for Everyone; M for Mature

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment</h2>
#     
# There's an introduction, which is good. It is important to write an introductory part, because it gives an idea about the content of the project.
# </div>

# ## Process the Data
# 
# Preprocessing:
# 
# - Load the data
# - Get the general view and info of the data
# - Clean up the data
# - Check for missing data and decide the best course of action for those values
# 
# 

# In[51]:


# Import all the libraries
import pandas as pd

from matplotlib import pyplot as plt

import numpy as np

import math as mt

# from math import factorial

from scipy import stats as st

import plotly.express as px


# In[52]:


# reading in the csv 
df_games = pd.read_csv('/datasets/games.csv')


# In[53]:


# Sample the dataframe
df_games.sample(10)


# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b>  Reviewer's comment </b>
#    
# 
# Please do not use `print` for dataframes.     
# </div>

# <div class="alert alert-info"; style="border-left: 7px solid blue">
#     Removed the print function  
# </div>

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#     
# Looks great! And sorry for that typo (pleasy). 
# 
# </div>

# In[54]:


# Inspecting the data
df_games.info()
df_games.shape


# In[55]:


# Inspecting the data
df_games.describe()


# ### Clean up data
# 
# 1. Replace the column names (make them lowercase).
# 2. Convert the data to the required types.
# 3. Describe the columns where the data types have been changed and why.
# 4. Missing values:
#     - Explain why you filled in the missing values as you did or why you decided to leave them blank.
#     - Why do you think the values are missing? Give possible reasons.
#     - Pay attention to the abbreviation TBD (to be determined). Specify how you intend to handle such cases.
# 5. Calculate the total sales (the sum of sales in all regions) for each game and put these values in a separate column.

# In[56]:


# Change column names to lowercase by renaming the columns
#df_games = df_games.rename(columns={
#    'Name': 'name',
#    'Platform': 'platform',
#    'Year_of_Release': 'year_of_release',
#    'Genre': 'genre',
#    'NA_sales': 'na_sales',
#    'EU_sales': 'eu_sales',
#    'JP_sales': 'jp_sales',
#    'Other_sales': 'other_sales',
#    'Critic_Score': 'critic_score',
#    'User_Score': 'user_score',
#    'Rating': 'rating',
#}) OLD CODE
df_games.columns = df_games.columns.str.lower()  # Using the suggested code
print(df_games.columns)


# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b>  Reviewer's comment </b>
#     
# This code would work as well:
# </div>
# 
# 
# ```python
# 
# df.columns = df.columns.str.lower()
# ```

# <div class="alert alert-info"; style="border-left: 7px solid blue">
#     Changed the code to lowercase the columns per suggestion. Definitely a lot cleaner of a way, which I will use in the future.
# </div>

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#     
# Yes, it's more pythonic. 
# </div>

# In[57]:


# calculating missing values
print(df_games.isna().sum()) 


# In[58]:


# Checking for any duplicates and the number of unique types
print('DUPLICATES\n', df_games[df_games.duplicated()])


# In[59]:


# Make sure all game names are lowercase
df_games['name'] = df_games['name'].str.lower()
df_games.sample(10)


# In[60]:


# Check for name duplicates
print(df_games[df_games['name'].duplicated()])


# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2>  Reviewer's comment 2 </h2>
#    
# 
# The `display` method will make the outputs looks better here as well. 
# 
# </div>

# In[61]:


# Check to see if there are games with multiple entries
print(df_games[df_games['name'] == 'grand theft auto v'])


# We notice that some game names are duplicated because they came out on multiple platforms.

# In[62]:


print(df_games[df_games['year_of_release'].isna()])


# We notice that there are names of games that contain the year in them that we can use to fill in the year_of_release column. We will handle this in the Fix Data section.

# In[63]:


df_games['year_of_release'] = df_games['year_of_release'].astype('Int64', errors='ignore')
#df_games['year_of_release'] = pd.to_datetime(df_games['year_of_release'], format='%Y')
df_games.info()
df_games['year_of_release'].unique()


# In[64]:


# checking for name-year-platform duplicates
print(df_games[df_games[['name', 'year_of_release', 'platform']].duplicated()])


# In[65]:


print(df_games[df_games['name'] == 'madden nfl 13'])
print(df_games[df_games['name'].isna()])


# There are two duplicated rows that have the same name-year-platform, but they have different sales values. So, before we delete the duplicates, we will combine the sales amounts.

# In[66]:


# Combine the two madden rows
df_games.loc[604, 'eu_sales'] = df_games.loc[604, 'eu_sales'] + df_games.loc[16230, 'eu_sales']
print(df_games.loc[604])


# In[67]:


# Combine the two null rows
df_games.loc[659, 'jp_sales'] = df_games.loc[659, 'jp_sales'] + df_games.loc[14244, 'jp_sales']
print(df_games.loc[659])


# In[68]:


# drop the duplicated rows
df_games.drop([14244, 16230],inplace=True)


# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#     
# 
# Sometimes **inplace** doesn't work properly, so I'd recommend that you re-write dataframe explicitly.
# 
# </div>

# In[69]:


# Reset the index and double check there's no name-year-platform duplicates
df_games = df_games.reset_index(drop=True)
print(df_games[df_games[['name', 'year_of_release', 'platform']].duplicated()])
#df_games.info()
#df_games.loc[16230]


# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# Try to check name-year-platform duplicates as well. 
# 
# </div>

# <div class="alert alert-info"; style="border-left: 7px solid blue">
# Found the two name-year-platform duplicates, combined the relevant info, and dropped the extra rows.
# </div>

# For the row without any identifiers like name and genre, we will also asign it 'unknown' as the name and the 'misc' genre because we will need them for analysis later.

# In[70]:


#name                  2
#genre                 2
#print(df_games[df_games['name'].isna()])  # index 659 & 14244
#print(df_games[df_games['genre'].isna()])  # index 659 & 14244


# In[71]:


df_games.loc[659, 'name'] = 'unknown'
df_games.loc[659, 'genre'] = 'Misc'
print(df_games.loc[659])


# #### Unique Values
# 
# We will check some of the columns for their data type and their unique values in order to be able to perform analysis and calculations later on. We want to check to see if there are values that can't be converted to another type like floats.

# In[72]:


# Checking for any duplicates and the number of unique types
print('Names\n', df_games['name'].sort_values().unique())


# In[73]:


print('Platforms\n', df_games['platform'].sort_values().unique())


# In[74]:


print('Genres\n', df_games['genre'].sort_values().unique())


# There should not be any null values for the genre since we converted the null values to 'Misc' earlier.

# In[75]:


print('Ratings\n', df_games['rating'].sort_values().unique())


# In[76]:


print(df_games[df_games['rating'].isna()].sample(12))


# In[77]:


#print(df_games[df_games['rating'] == 'M'])
print(df_games.groupby('rating')['rating'].count())


# Adults Only (AO), Everyone (E), Everyone 10+ (E10+), Early Childhood (EC), Kids to Adults (KA), Mature (M), Rating Pending (RP), Teen (T)
# 
# For the null values in ratings, since there is no discernable way of assigning a rating to the missing games, we will later fill them in the Rating Pending (RP) category.

# In[78]:


print('critic_score\n', df_games['critic_score'].sort_values().unique())
# df_games.info()


# In[79]:


# Make sure critic_score is a float column
df_games['critic_score'] = df_games['critic_score'].astype('float64')
df_games.info()


# In[80]:


# check user_score
print('user_score\n', df_games['user_score'].sort_values().unique())


# In[81]:


# Check the games that have "tbd" as the user_score
print(df_games[df_games['user_score']  == 'tbd'])  # tbd


# We will treat user scores with the value of 'tbd' as null values.
# 
# Then, we will convert the column to float for calculations and analysis later.

# In[82]:


# Change 'tbd' to null
df_games['user_score'] = df_games['user_score'].replace('tbd', float("NaN"))
# Make sure there is no more 'tbd' values
print('user_score\n', df_games['user_score'].sort_values().unique())


# In[83]:


# Change the column to float and check to see it worked
df_games['user_score'] = df_games['user_score'].astype('float64')
df_games.info()


# Both the critic_score and user_score columns should be prepared to be adjusted now that there are no tricky values and are floats.

# ### Fix Data 
# 
# Certain columns have information that can be extracted from other columns to help provide more meaningful data to the data set. In this section we will describe how we can fix the data for the analysis later on.
# 
# 

# #### Year of Release
# 
# Some of the games have the year in their names. For example, we see "madden nfl 2004" in the name column, so we can use this to fill in the year_of_release column with "2004" and similarly with other games whose years appear in their name.

# In[84]:


# Create a function for assigning the year_of_release column the year that appears in the name column
game_years_index = df_games[df_games['year_of_release'].isna()].index.tolist()
print(game_years_index)

def hashnumbers(inputString):
    x = inputString.split()
    y = ''
    for str in x:
        if str.isdigit(): 
            if float(str) > 1980:
                y = float(str)
                break;
    if y == '':
        return float("NaN")
    else:
        return y
    
# Check one of the games for its name
print(df_games.loc[183])


# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# 
# There's a shorter way: </div>

# In[85]:


# Reviewer's code

df_games.name.str.extract(r'(\d\d\d\d).*')


# <div class="alert alert-info"; style="border-left: 7px solid blue">
#     Tried implementing your code, but could not do it successfully. I'll definitely save this snippet of code for a future project.
# </div>

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 </h2>
#     
# I'll try to show you the example below: 
# 
# </div>

# In[86]:


# Reviewer's code 2

ddf = df_games.copy()

# Print them to see the indices of missing values (183, 377, etc)
display(ddf[ddf['year_of_release'].isna()].head())

ddf['year_of_release'] = ddf['year_of_release'].fillna(ddf.name.str.extract(r'(\d\d\d\d).*')[0].astype('float'))

# Check the rows that had missing values:
ddf.loc[[183, 377]]


# In[87]:


#yr_in_name = df_games['name'].str.extract(r'(\d\d\d\d).*')
#print(yr_in_name[~yr_in_name[0].isna()])


# In[88]:


#yr_finder = df_games[df_games['year_of_release'].isna()]
#def has_numbers(inputString):
#    return any(char.isdigit() for char in inputString)
#for row in yr_finder['name']:
#    if has_numbers(row):
#        print(row)


# In[89]:


#print(yr_finder.loc[yr_finder['name'].str.contains('2k')])
#print(yr_finder.loc[yr_finder['name'].str.contains('20')])


# In[90]:


# Use the function to fill in the appropriate years
for i in game_years_index:
    df_games.loc[i, 'year_of_release'] = hashnumbers(df_games.loc[i, 'name'])
    
# Check the same row as above to see if the year_of_release was placed correctly
print(df_games.loc[183])


# In[91]:


# Double check that there are less nulls
print(df_games[df_games['year_of_release'].isna()])
print(df_games.info())


# <!--#### Critic Score
# 
# Since there are games that have a user score, but no critic score, we will use the user score to fill in the critic score in order to get a larger sample size when doing analysis with user scores.-->

# In[92]:


# Example of row with no critic_score, but a user_score 
#print(df_games.loc[266])


# In[93]:


# Fill in critic_score with user_score * 10
#na_cs_us_index = df_games[(df_games['critic_score'].isna()) & (~df_games['user_score'].isna())].index #  & (df_games['rating'].isna())]
#print(na_cs_us_index)

#for i in na_cs_us_index:
#    df_games.loc[i, 'critic_score'] = df_games.loc[i, 'user_score']*10


# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# Why do you think it is a good and precise replacement?     
# 
# </div>

# <div class="alert alert-info"; style="border-left: 7px solid blue">
#     Good point, I shouldn't assume they would be good replacements for each other. I went back and forth thinking about using a mean or median to fill in the missing values, but that didn't quite feel correct either. Ultimately, I decided to comment out the old code out for filling both the critic and user score columns.
# </div>

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 </h2>
#     
# Yes, we do not have a decent replacement for such a huge number of gaps. 
#     
# </div>

# In[94]:


#print(df_games.loc[263])
#df_games.info()
#print(df_games['critic_score'].unique())


# <!--#### User Score
# 
# Since there are games that have a critic score, but no user score, we will use the critic score to fill in the user score in order to get a larger sample size when doing analysis with user scores.-->

# In[95]:


# Example of row with critic_score, but no user_score
#df_games.loc[182]


# In[96]:


# Fix user_score column
#df_games.info()
#print(df_games['user_score'].sort_values().unique())
# print(df_games['user_score'].mean())


# In[97]:


# user_score clean up
#na_us_cs_index = df_games[(~df_games['critic_score'].isna()) & (df_games['user_score'].isna())].index #  & (df_games['rating'].isna())]
#print(na_us_cs_index)


# In[98]:


# Fill empty user_score with critic_score/10
#for i in na_us_cs_index:
#    df_games.loc[i, 'user_score'] = df_games.loc[i, 'critic_score']/10


# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# Same question, we cannot be sure. 
# </div>

# <div class="alert alert-info"; style="border-left: 7px solid blue">
#     Commented out the code relating to filling in the user score with the critic scores.
# </div>

# In[99]:


# Check a row to see if the function worked
#df_games.loc[182]


# In[100]:


# 8   Critic_Score     8137 non-null   float64
# 9   User_Score       10014 
#df_games.info()
#print(df_games['critic_score'].sort_values().unique())
#print(df_games['user_score'].sort_values().unique())


# #### Rating
# 
# For this column, we will take the null values and fill them in with 'RP' for Rating Pending as the value since this already exists as an option.

# In[101]:


# There should be no nulls left in the rating column
df_games['rating'] = df_games['rating'].fillna('RP')
df_games.info()


# In[102]:


# Showing no null values
print(df_games['rating'].unique())
df_games.groupby('rating')['rating'].count()


# ### Enrich Data
# 
# Calculate the total sales (the sum of sales in all regions) for each game and put these values in a separate column.

# In[103]:


# Calculate the total sales (the sum of sales in all regions) for each game 
# and put these values in a separate column.
# df_games['global_sales'] = df_games['na_sales'] + df_games['eu_sales'] + df_games['jp_sales'] + df_games['other_sales']
df_games['global_sales'] = df_games[['na_sales','eu_sales','jp_sales','other_sales']].sum(axis=1)
df_games.info()
#print(df_games.sample(10))


# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment</h2>
#     
# You can also use **sum** with **axis=1** argument:
# </div>
# 
# ```python
# 
# 
# df['total_sales'] = df[['na_sales','eu_sales','jp_sales','other_sales']].sum(axis=1)
# ```

# <div class="alert alert-info"; style="border-left: 7px solid blue">
#     Implemented suggested code. Much more clean!
# </div>

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 </h2>
#     
# Though it may still work slower on larger dataframes.
#     
# </div>

# In[104]:


# Confirm if the 'global_sales' column was created and order by highest 'global_sales'
print(df_games.sort_values('global_sales', ascending=False).head(16))


# Looking at the highest values for global_sales, there appears to be some outliers on the upper end. Many of these high performing games were on the Wii platform and in the sports genre. We will further examine these trends later on.

# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# 
# Don't forget about intermediate conclusions. 
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 </h2>
#     
# 👍    
# </div>

# <div class="alert alert-info"; style="border-left: 7px solid blue">
#     Added the intermediate conclusions above
# </div>

# ## Analyze the data
# 
# With our data cleaned and fixed, we will cover the following points:
# 
# * 2.1: Look at how many games were released in different years. Is the data for every period significant?
# * 2.2: Look at how sales varied from platform to platform. Choose the platforms with the greatest total sales and build a distribution based on data for each year. Find platforms that used to be popular but now have zero sales. How long does it generally take for new platforms to appear and old ones to fade?
# * 2.3: Determine what period you should take data for. To do so, look at your answers to the previous questions. The data should allow you to build a prognosis for 2017.
#     * Work only with the data that you've decided is relevant. Disregard the data for previous years.
#     * Which platforms are leading in sales? Which ones are growing or shrinking? Select several potentially profitable platforms.
# * 2.4: Build a box plot for the global sales of all games, broken down by platform. Are the differences in sales significant? What about average sales on various platforms? Describe your findings.
# * 2.5: Take a look at how user and professional reviews affect sales for one popular platform (you choose). Build a scatter plot and calculate the correlation between reviews and sales. Draw conclusions.
# * Keeping your conclusions in mind, compare the sales of the same games on other platforms.
# * 2.6: Take a look at the general distribution of games by genre. What can we say about the most profitable genres? Can you generalize about genres with high and low sales?
# 

# ### Count of Games Over the Years

# In[105]:


# Look at how many games were released in different years. Is the data for every period significant?
num_games_by_yr = df_games.groupby('year_of_release')['name'].count().reset_index()
#print(num_games_by_yr.columns)
print(num_games_by_yr)


# In[106]:


fig_bar = px.bar(num_games_by_yr, x='year_of_release', y='name')
fig_bar.update_layout(
    title_text='Number of Games Release by Year', # title of plot
    xaxis_title_text='Years', # xaxis label
    yaxis_title_text='Number of Games', # yaxis label
)
fig_bar.show()


# Since video game technology was in its infancy in the 1980s and early 1990s, this timeframe might not be as important as more recent years as shown the low number of games in this time. We do see an increase in the number of games up through the late 2000s, but a taper off into more recent times.

# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# 
# A visualization would not be redundant here. 
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 </h2>
#     
# Nice chart!     
# </div>

# <div class="alert alert-info"; style="border-left: 7px solid blue">
#     Added the bar chart
# </div>

# ### Sales by Platform
# 
# Look at how sales varied from platform to platform. Choose the platforms with the greatest total sales and build a distribution based on data for each year. Find platforms that used to be popular but now have zero sales. How long does it generally take for new platforms to appear and old ones to fade?

# In[107]:


# Look at how sales varied from platform to platform. 
# Choose the platforms with the greatest total sales and build a distribution based on data for each year. 
# Find platforms that used to be popular but now have zero sales. 
# How long does it generally take for new platforms to appear and old ones to fade?


# In[108]:


# Create a pivot table to show the total sales of each platform per year
games_pivot = pd.pivot_table(df_games, values='global_sales', index='year_of_release', columns='platform', aggfunc='sum')
print(games_pivot)


# In[109]:


print(games_pivot.sum().sort_values(ascending=False))


# In[110]:


# Line chart with px
fig_time = px.line(games_pivot)
fig_time.update_layout(
    title_text='Sales Among the Platforms', # title of plot
    xaxis_title_text='Years', # xaxis label
    yaxis_title_text='Global Sales in $Millions', # yaxis label
)
fig_time.show()


# PS2, Xbox 360, PS3, Wii, DS, and PS have the highest total sales as seen by the table and chart above with the highest peaks belonging to these platforms.
# 
# The platforms of 2600, NES, GB, SNES, GEN, SCD, NG, SAT, and 3DO had reached the end of their sales by the end of the 1990s.
# 
# An interesting trend is that some platforms like Playstation (PS) and Xbox (X*) have successive platforms whose sales timelines overlap. When one platform is on the decline in sales, the successor platform will be on the rise in sales.

# In[111]:


# Get the count of years for each platform
gpv = games_pivot.count().sort_values(ascending=False)
print(gpv)


# In[112]:


# Mean and medians of platform lifespans
print('Average Length of a Platform:', gpv.mean())
print('Median Length of a Platform:', gpv.median())


# On average, a platform's sale life is about 7.5 years from the start of its earning to the time the console's sales tend to fade.
# 
# The Playstation (PS) line of platforms have been selling for a decade or more with the exception of PS4, which is the newest platform. 

# In[113]:


#platform_years = df_games.groupby(['platform', 'year_of_release'])['global_sales'].sum().reset_index()
#print(platform_years.sort_values('global_sales').tail(20))
#fig = px.line(platform_years[platform_years['year_of_release'] > 2008], x='year_of_release', y='global_sales', color='platform')
#fig.show()
#platform_years


# ### Recent Data
# 
# * Determine what period you should take data for. To do so, look at your answers to the previous questions. The data should allow you to build a prognosis for 2017.
#     * Work only with the data that you've decided is relevant. Disregard the data for previous years.
#     * Which platforms are leading in sales? 
#     * Which ones are growing or shrinking? Select several potentially profitable platforms.
# 

# From the "Sales Among the Platforms" chart above, we see that from 2013 on Xbox One and PS4 are the two newest consoles that are under the 7.5 year average lifespan of consoles. Also, within this timeframe are other consoles that are tapering down in sales like the predecessors, PS3 and Xbox 360. So, we will be focusing on the data from 2013 onward.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment</h2>
#     
# Agreed. 
# </div>

# In[114]:


# Selecting data from 2013 and on
df_2013_2016 = df_games[df_games['year_of_release'] >= 2013]
print(df_2013_2016['year_of_release'].unique())
df_2013_2016


# In[115]:


print(df_2013_2016.groupby('platform')['global_sales'].sum().sort_values(ascending=False))


# In[116]:


# Pivot table for 2013-2016 data by year and platform
pivot_2013_2016 = pd.pivot_table(df_2013_2016, values='global_sales', index='year_of_release', columns='platform', aggfunc='sum')
print(pivot_2013_2016)


# In[117]:


# Line chart with the 2013-2016 pivot data
fig_time = px.line(pivot_2013_2016)
fig_time.update_layout(
    title_text='Sales Among the Platforms From 2013 to 2016', # title of plot
    xaxis_title_text='Years', # xaxis label
    yaxis_title_text='Global Sales in $Millions', # yaxis label
    xaxis_tickformat='.0f'
)
fig_time.show()


# PS4 and XOne are leading in sales by a large margin over the other platforms even though their sales have been diminishing in recent years. Other platforms like 3DS, PC, WiiU and PSV have also been on the decline, but at a slower pace as can be seen in the chart above.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment</h2>
#     
# Correct. 
# </div>
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# 
# We can try to convert `year_of_release` to dates (years) in order to avoid float numbers on X axis. 
# 
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 </h2>
#     
# Your idea is even better.     
# </div>

# <div class="alert alert-info"; style="border-left: 7px solid blue">
#     Updated tickformat to show just the year.
# </div>

# ### Boxplot
# 
# Build a box plot for the global sales of all games, broken down by platform. Are the differences in sales significant? What about average sales on various platforms? Describe your findings.
# 

# In[118]:


# Boxplot of how sales varied from platform to platform
fig_boxplot = px.box(df_2013_2016, x="platform", y="global_sales")
fig_boxplot.update_layout(
    title_text='Platform Sales Since 2013', # title of plot
    xaxis_title_text='Platform Console', # xaxis label
    yaxis_title_text='Sales $USD Millions', # yaxis label
    yaxis_range=[-1, 6]
)
fig_boxplot.show()


# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# 
# Try to limit the Y axis values, since it's hard to compare the boxes.     
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 </h2>
#     
# It is definitely better now.      
# </div>

# <div class="alert alert-info"; style="border-left: 7px solid blue">
#     Set the yaxis_range attribute to zoom into the boxes.
# </div>

# In[119]:


df_2013_2016.groupby('platform')['global_sales'].describe()


# We see in both the boxplot and descriptive statistics that X360, XOne, Wii, and PS4 had distributions where the upper quartiles were higher than the other competitors. In addition, these platforms had at least an average of \\$600,000 in total sales.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#     
# 
# We deal with sum sales, so the outliers are just some popular games that were sold in huge volumes, which means that they are quite common. I would even say bestsellers. And this is a very illustrative thing. Look at the 3DS. Small box and lots of outliers. This suggests that a large part of the sales volume comes from these outliers. At the same time, the XOne has a different situation: a large box, a long whisker and few outliers. That indicates that a wide variety of games are bought on XOne, and, moreover, in considerable quantities.
# 
# </div>
# 

# ### User and Critic Reviews on Sales  
# 
# Take a look at how user and professional reviews affect sales for one popular platform (you choose). Build a scatter plot and calculate the correlation between reviews and sales. Draw conclusions.
# 
# Keeping your conclusions in mind, compare the sales of the same games on other platforms.

# #### PS4 User and Critc Reviews and Sales
# 
# We'll first examine the PS4 to see if there is any correlation between the reviews and sales.

# In[120]:


# Choosing PS4 as the platform to examine
ps4_recent = df_2013_2016[df_2013_2016['platform'] == 'PS4']
print(ps4_recent.info())
print(ps4_recent.isna().sum())


# In[121]:


# Scatter plot
fig_scatter = px.scatter(ps4_recent, x="user_score", y="global_sales", hover_data=['name'], trendline="ols")  #, size="days_listed", color='make')
fig_scatter.update_layout(
    title_text='How Do User Scores Affect the Sales of PS4 Games', # title of plot
    xaxis_title_text='User Score out of 10', # xaxis label
    yaxis_title_text='Sales $USD Millions', # yaxis label
)
fig_scatter.show()


# In[122]:


# Correlation Coefficient between user_score and global_sales
print(ps4_recent['user_score'].corr(ps4_recent['global_sales']))


# We see a low correlation between user scores and global sales for the PS4 with the regression line in the scatter plot and with the calculated correlation coefficient that is close to 0. Thus, the user's score for games don't determine how well the games sell.

# In[123]:


# Scatter plot with critic_score
fig_scatter_cr = px.scatter(ps4_recent, x="critic_score", y="global_sales", hover_data=['name'], trendline="ols")  #, size="days_listed", color='make')
fig_scatter_cr.update_layout(
    title_text='How Do Critic Scores Affect the Sales of PS4 Games', # title of plot
    xaxis_title_text='Critic Score out of 100', # xaxis label
    yaxis_title_text='Sales $USD Millions', # yaxis label
)
fig_scatter_cr.show()


# In[124]:


# Correlation Coefficient between critic_score and global_sales
print(ps4_recent['critic_score'].corr(ps4_recent['global_sales']))


# When looking at the critic score and its relationship with sales, there is a bit higher and positive relationship than with the user scores as seen with the regression line in the scatter plot and the correlation coefficient being higher. So, a critic's score holds more weight than user's score in terms of sales.

# #### Other Platforms' User and Critc Reviews and Sales
# 
# Now, we are going to do a similar analysis with two more platforms, XOne and PS3.

# In[125]:


recent_xone = df_2013_2016[df_2013_2016['platform'] == 'XOne']
#recent_3ds = df_2013_2016[df_2013_2016['platform'] == '3DS']
recent_ps3 = df_2013_2016[df_2013_2016['platform'] == 'PS3']
print(recent_xone.info())
#print(recent_3ds.info())
print(recent_ps3.info())


# In[126]:


# Scatter plot with user score of XOne
fig_scatter = px.scatter(recent_xone, x="user_score", y="global_sales", hover_data=['name'], trendline="ols")  #, size="days_listed")
fig_scatter.update_layout(
    title_text='How Do User Scores Affect the Sales of XOne', # title of plot
    xaxis_title_text='User Score out of 10', # xaxis label
    yaxis_title_text='Sales $USD Millions', # yaxis label
)
fig_scatter.show()


# In[127]:


# Scatter plot with critic_score of XOne
fig_scatter = px.scatter(recent_xone, x="critic_score", y="global_sales", hover_data=['name'], trendline="ols")  #, size="days_listed")
fig_scatter.update_layout(
    title_text='How Do Critic Scores Affect the Sales of XOne', # title of plot
    xaxis_title_text='Critic Score out of 100', # xaxis label
    yaxis_title_text='Sales $USD Millions', # yaxis label
)
fig_scatter.show()


# In[128]:


# Correlation Coefficients for user_score and critic_score of XOne
print('XOne Correlation Coefficients')
print('User-Sales:', recent_xone['user_score'].corr(recent_xone['global_sales']))
print('Critic-Sales:', recent_xone['critic_score'].corr(recent_xone['global_sales']))


# In[129]:


# Scatter plot with user score of PS3
fig_scatter = px.scatter(recent_ps3, x="user_score", y="global_sales", hover_data=['name'], trendline="ols")  #, size="days_listed")
fig_scatter.update_layout(
    title_text='How Do User Scores Affect the Sales of PS3', # title of plot
    xaxis_title_text='User Score out of 10', # xaxis label
    yaxis_title_text='Sales $USD Millions', # yaxis label
)
fig_scatter.show()


# In[130]:


# Scatter plot with critic_score of XOne
fig_scatter = px.scatter(recent_ps3, x="critic_score", y="global_sales", hover_data=['name'], trendline="ols")  #, size="days_listed")
fig_scatter.update_layout(
    title_text='How Do Critic Scores Affect the Sales of Ps3', # title of plot
    xaxis_title_text='Critic Score out of 100', # xaxis label
    yaxis_title_text='Sales $USD Millions', # yaxis label
)
fig_scatter.show()


# In[131]:


# Correlation Coefficients for user_score and critic_score of PS3
print('PS3 Correlation Coefficients')
print('User-Sales:', recent_ps3['user_score'].corr(recent_ps3['global_sales']))
print('Critic-Sales:', recent_ps3['critic_score'].corr(recent_ps3['global_sales']))


# Just like our analysis of the relationship between ratings and sales of PS4 games, we see a similar pattern with XOne and PS3 games. User scores have very little correlation to their total sales, whereas critic scores have a stronger and a positive relationship to total sales. So, critic scores may have more of an influence on total sales.
# 
# <!--Overall, critic and user scores have little correlation to the performace in terms of sales. Whether comparing one of the better performing platforms, like PS4, or all platforms, there does not appear to be a strong connection between the scores and the total sales.-->

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# - These are not user scores. 
#     
#     
# - According to the task, we have to choose 1 platform, analyze how sales depend on critics' and users' scores, display scatter plots and calculate the correlation coefficients. Then we have to choose  2 or more other platforms, repeat the correlation analysis for them and compare the results. Would you add it?  
# 
# 
# The wording in this task is ambiguous, but since we are comparing platforms, it is reasonable to take scores for all games on the platform within a chosen period. In other words, you do not need to check each game in the dataset. 
# 
# All you need here is to take 2 or more other platforms and repeat the analysis. Then compare the results. </div>

# <div class="alert alert-info"; style="border-left: 7px solid blue">
#     
# - Fixed the PS4 critic score scatter plot
# 
# - Made the new comparisons with XOne and PS3 platforms; and updated summary of findings for this section
# 
# </div>

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#     
# 
# Great, this is enough for that task. 
# </div>
# 

# In[132]:


#platform_corr = df_2013_2016.groupby('platform').corr().reset_index()
#print(platform_corr[platform_corr['platform'] == 'PS4'])
#print(platform_corr.head(16))


# In[133]:


# Create a table of correlation coefficients 
#score_corr = platform_corr[(platform_corr['level_1'] == 'critic_score') | (platform_corr['level_1'] == 'user_score')]
#print(score_corr[['platform', 'level_1', 'global_sales']])
#pivot_platform_corr = pd.pivot_table(platform_corr, values='global_sales', index='year_of_release', columns='platform', aggfunc='sum')


# <!--The above table shows the correlation coefficients for all the platforms between critic score and global sales, and also between user score and global sales. As we can see, there is not a high correlation among most of the platforms in both user and critic scores.
# 
# It is important to note that with our current data, there were only two user and critic scores for the PSP platform, which creates a perfect correlation between the two points as seen in the scatter plots, and hence, a -1 coefficient. -->

# ### Popular Genres
# 
# Take a look at the general distribution of games by genre. 
# * What can we say about the most profitable genres? 
# * Can you generalize about genres with high and low sales?

# In[134]:


#Take a look at the general distribution of games by genre. What can we say about the most profitable genres? 
global_by_genre = df_2013_2016.groupby('genre')['global_sales'].sum()
print(global_by_genre.sort_values(ascending=False))
#Can you generalize about genres with high and low sales?




# In[135]:


# Histogram of of games by genre
pt_histo = px.histogram(df_2013_2016, x='global_sales', color='genre', barmode="overlay", nbins=20) #, hover_data=['type']) # color='make'
pt_histo.update_layout(
    title_text='Global Sales Across Genres', # title of plot
    xaxis_title_text='Global $USD in millions', # xaxis label
    yaxis_title_text='Frequency', # yaxis label
)
pt_histo.show()


# In[136]:


# Boxplot by genre
fig_boxplot_genre = px.box(df_2013_2016, x="genre", y="global_sales")
fig_boxplot_genre.update_layout(
    title_text='Genre Sales Since 2013', # title of plot
    xaxis_title_text='Genres', # xaxis label
    yaxis_title_text='Sales $USD Millions', # yaxis label
)
fig_boxplot_genre.show()


# In[137]:


df_2013_2016.groupby('genre')['global_sales'].describe()


# Shooter type games have overall been dominant in terms of sales as seen in the histogram with higher frequencies in the right-tailed section of the sales; the boxplot shows that the upper quartiles are higher than other genres; and the descriptive statistics of the mean and median are much higher than other genres.
# 
# Sports is another genre that performs well as seen in the boxplot and the descriptive statistics with the second highest mean and median.
# 
# Action shows some of the highest sales across the genres. The histogram shows the most frequently in the first and second bins. In the boxplot, we see that they have lots of upper outliers as well two of the maxes.  
# 
# Puzzle and strategy games tend to do poorly amongst the genres. They have the lowest number of games available, which may contribute to the low overall sales.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment </h2>
#     
# Correct. Action could make such volumes only due to the large number of games, so it's better to evaluate median or mean as well. 
# </div>

# ## Regional Profiles
# 
# Create a user profile for each region. For each region (NA, EU, JP), determine:
# - The top five platforms. Describe variations in their market shares from region to region.
# - The top five genres. Explain the difference.
# - Do ESRB ratings affect sales in individual regions?

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
#     
# Don't forget to consider relevant time interval. 
# </div>

# <div class="alert alert-info"; style="border-left: 7px solid blue">
#     Updated the following code below using the data since 2013, which is the data used in the previous sections.
# </div>

# In[138]:


# Creating a function to print out the top five by column and region
def region_sales(column):
    reg = {'North America': 'na', 
           'Europe': 'eu', 
           'Japan': 'jp'
          }
    for x in reg:
        print(x, df_2013_2016.groupby(column)[reg[x] + '_sales'].sum().sort_values(ascending=False).head())
        print()
        


# In[139]:


# Create the 3 regions by platform
region_sales('platform')
#platform_sales_na = df_games.groupby('platform')['na_sales'].sum()
#platform_sales_eu = df_games.groupby('platform')['eu_sales'].sum()
#platform_sales_jp = df_games.groupby('platform')['jp_sales'].sum()
#print(platform_sales_na)
#print('North America', platform_sales_na.sort_values(ascending=False).head())
#print('\nEurope', platform_sales_eu.sort_values(ascending=False).head())
#print('\nJapan', platform_sales_jp.sort_values(ascending=False).head())


# PS4 occured in the top five in all three regions and was the highest in both North America and Europe. PS3 and 3DS were also popular in all three regions, but varying among the top five. Xbox 360 and XOne came in two of the three regions' top five. 
# 
# <!--PS2 ranked in the top 5 for all regions, whereas Xbox 360, Wii, PS3, DS and PS came in two of the three regions. -->

# In[140]:


# Top Five Genres by Region
region_sales('genre')
#genre_sales_na = df_games.groupby('genre')['na_sales'].sum()
#genre_sales_eu = df_games.groupby('genre')['eu_sales'].sum()
#genre_sales_jp = df_games.groupby('genre')['jp_sales'].sum()
#print('North America', genre_sales_na.sort_values(ascending=False).head())
#print('\nEurope', genre_sales_eu.sort_values(ascending=False).head())
#print('\nJapan', genre_sales_jp.sort_values(ascending=False).head())


# Action, Shooter and Role-Playing are in the top five for all regions with Action usually towards the top of the list across all regions. Role-Playing edged over Action only in Japan.
# 
# Sports occured in two of the three regions.
# 
# An interesting note is that Misc games appeared in the list of two of the three regions possibly due to the original data not having entries in the genre column and so got placed in the Misc category.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 </h2>
# 
#     
# There're some similiarities between Europe the Northern America, while  people in Japan prefer portable platforms and japanese market. We definitely should not recommend them XBox :)  In addition, Japanese also don't like shooters as much as in NA and EU people do.  
#  
# </div>   

# <!--Top 5 Genres:
# - Action          1744.17
# - Sports          1331.27
# - Shooter         1052.45
# - Role-Playing     934.56
# - Platform         827.77-->

# In[141]:


# Do ESRB ratings affect sales in individual regions
# Filter out PR games
region_sales('rating')


# Mature (M) games ranked among not only the top five in all regions, but first in two of them. Everyone (E), Everyone 10+ (E10+), and Teen (T) rounded out the top five in all regions as well in varying orders. 
# 
# An important caveat in the rankings is that of Rating Pending (RP) occuring in all three regions. This may be due to assigning the null values in the rating column as RP.
# 
# <!--The rating of Everyone (E) is the highest grossing rating for games across the regions with the caveat of Rating Pending (RP) being second or first in the three regions. This may be due to assigning the null values in the rating column as RP.  The other types that were always in the top five are Teen (T) and Mature (M).
# Finally, Everyone 10+ (E10+) rounded out the fifth spot in all three regions.-->

# In[142]:


# Boxplot of global_sales by genre
#fig_boxplot_genre = px.box(df_2013_2016, x="genre", y="global_sales")
#fig_boxplot_genre.update_layout(
#    title_text='Genre Sales Since 2013', # title of plot
#    xaxis_title_text='Genre', # xaxis label
#    yaxis_title_text='Sales $USD Millions', # yaxis label
#    yaxis_range=[-1, 6]
#)
#fig_boxplot_genre.show()


# ## Hypotheses Testing :
# 
# In this section, we will test the following hypotheses:
# 
# - 4.1: Average user ratings of the Xbox One and PC platforms are the same. 
# - 4.2: Average user ratings for the Action and Sports genres are different.
# 
# Set the alpha threshold value yourself.
# Explain:
# 
# - How you formulated the null and alternative hypotheses 
# - What significance level you chose to test the hypotheses, and why

# ### Xbox One vs PC Ratings
# 
# Average user ratings of the Xbox One and PC platforms are the same.
# 
# We will examine this by performing a T-test on the means of the two statistical populations to see if they are equal. The statistical significance level will be 5%.
# 
# <b>H<sub>0</sub></b>: The average user ratings of the Xbox One and PC platforms are the same.
# 
# <b>H<sub>1</sub></b>: The average user ratings of the Xbox One and PC platforms are different.

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
#     
# Here we also need relevant data. 
# </div>

# <div class="alert alert-info"; style="border-left: 7px solid blue">
#     Also updated the following code below using the data since 2013, which is the data used in the previous sections.
# </div>

# In[143]:


# Test the hypotheses
df_xone = df_2013_2016[df_2013_2016['platform'] == 'XOne']['user_score']
df_xone_na = df_xone[~df_xone.isna()]
df_pc = df_2013_2016[df_2013_2016['platform'] == 'PC']['user_score']
df_pc_na = df_pc[~df_pc.isna()]
print(df_xone_na.mean())
print(df_pc_na.mean())

alpha = .05

results = st.ttest_ind(df_xone_na, df_pc_na, equal_var=False)
print('p-value:', results.pvalue)

if (results.pvalue < alpha):
    print("We reject the null hypothesis")
else:
    print("We can't reject the null hypothesis")


# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 </h2>
# 
#     
# By the way, you can apply `dropna` right after `df_2013_2016[df_2013_2016['platform'] == 'PC']['user_score']`:
# </div>   

# In[144]:


# Reviewer's code 2 

df_2013_2016[df_2013_2016['platform'] == 'PC']['user_score'].dropna()


# According to the test and our significance level, there is not sufficient evidence that we should reject the null hypothesis, which in this case means that the average user rating for Xbox One is the same as the average user rating for PC's.

# ### Action vs Sports Ratings
# 
# Average user ratings for the Action and Sports genres are different.
# 
# We will examine this by performing a T-test on the means of the two statistical populations to see if they are equal. The statistical significance level will be 5%.
# 
# <b>H<sub>0</sub></b>: The Average user ratings for the Action and Sports genres are the same. (The difference between the means is 0.)
# 
# <b>H<sub>1</sub></b>: The Average user ratings for the Action and Sports genres are different.

# In[145]:


# Test the hypotheses
df_action = df_2013_2016[df_2013_2016['genre'] == 'Action']['user_score']
df_action_na = df_action[~df_action.isna()]
df_sports = df_2013_2016[df_2013_2016['genre'] == 'Sports']['user_score']
df_sports_na = df_sports[~df_sports.isna()]
print(df_action_na.mean())
print(df_sports_na.mean())

alpha = .05

results = st.ttest_ind(df_action_na, df_sports_na, equal_var=False)
print('p-value:', results.pvalue)

if (results.pvalue < alpha):
    print("We reject the null hypothesis")
else:
    print("We can't reject the null hypothesis")


# Since we rejected the hypothesis with such a small p-value, this indicates that the average user ratings for Action and Sports games are highly likely not the same. Thus, there is a statistically significant difference in means between average user ratings for Action and Sports games.

# ## Conclusion
# 
# Through our analysis and testing in the recent years, we observed the following: 
# 
# - Platforms have about a 7 years lifespan of sales with some successors having overlap in their sales, like the PS line of platforms.
# - PS4 and XOne are two of the most popular and well performing platforms with PS4 being highly profitable in North America and Europe. 
# - Mature (M), Everyone (E), Everyone 10+ (E10+), and Teen (T) are the most popular genres with Mature being highly popular in North America and Europe.
# - Shooter, Action and Role-Playing games tend to be the most popular in America, Europe, and Japan.
# <!--- In these regions, PS2 and Xbox 360 had the largest overall sales, but their successors, PS4 and Xbox One were seeing rising sales in the recent years.
# - Also, across the three regions, the game rating of Everyone was the most popular followed by Teen, Mature and Everyone 10+. -->
# - Critic scores have a larger bearing than user scores on how much total sales a game would have.
# - Average user ratings of the Xbox One and PC platforms are the same.
# - Action and Sports games have different average user ratings.
# 
# Using these observations for 2017, the advertising department or agency can focus their attention on the previously mentioned game genres on the mentioned platforms with the mentioned ratings. For example, more ads can be focused on an action game coming out on PS4 with a rating of Mature (M) in North America. 
# 

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 </h2>
# 
#     
# Excellent! 
# </div>   
