#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 2</b> 
#     
# Great, Mark! You've done a great job on all the comments and now your project has been accepted.
#     
# Thank you for your work and I wish you success in the following projects!

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

# # Yandex.Music

# <div style="border:solid green 2px; padding: 20px">
#     
# <div class="alert alert-success">
# <b>Review summary</b> 
#     
# Mark, thanks for submitting the project. You've done a very good job and I enjoyed reviewing it.
#     
# - You completed all the tasks.
# - Your code was optimal and easy to read. 
# - You wrote your own functions.
#     
# There is only one critical comment that need to be corrected. You will find it in the red-colored cell in relevant section. You may also work on the yellow comments. If you have any questions please write them when you return your project. 
#     
# I'll be looking forward to getting your updated notebook.

# # Contents <a id='back'></a>
# 
# * [Introduction](#intro)
# * [Stage 1. Data overview](#data_review)
#     * [Conclusions](#data_review_conclusions)
# * [Stage 2. Data preprocessing](#data_preprocessing)
#     * [2.1 Header style](#header_style)
#     * [2.2 Missing values](#missing_values)
#     * [2.3 Duplicates](#duplicates)
#     * [2.4 Conclusions](#data_preprocessing_conclusions)
# * [Stage 3. Testing the hypotheses](#hypotheses)
#     * [3.1 Hypothesis 1: user activity in the two cities](#activity)
#     * [3.2 Hypothesis 2: music preferences on Monday and Friday](#week)
#     * [3.3 Hypothesis 3: genre preferences in Springfield and Shelbyville](#genre)
# * [Findings](#end)

# ## Introduction <a id='intro'></a>
# Whenever we're doing research, we need to formulate hypotheses that we can then test. Sometimes we accept these hypotheses; other times, we reject them. To make the right decisions, a business must be able to understand whether or not it's making the right assumptions.
# 
# In this project, you'll compare the music preferences of the cities of Springfield and Shelbyville. You'll study real Yandex.Music data to test the hypotheses below and compare user behavior for these two cities.
# 
# ### Goal: 
# Test three hypotheses:
# 1. User activity differs depending on the day of the week and from city to city. 
# 2. On Monday mornings, Springfield and Shelbyville residents listen to different genres. This is also true for Friday evenings. 
# 3. Springfield and Shelbyville listeners have different preferences. In Springfield, they prefer pop, while Shelbyville has more rap fans.
# 
# ### Stages 
# Data on user behavior is stored in the file `/datasets/music_project_en.csv`. There is no information about the quality of the data, so you will need to explore it before testing the hypotheses. 
# 
# First, you'll evaluate the quality of the data and see whether its issues are significant. Then, during data preprocessing, you will try to account for the most critical problems.
#  
# Your project will consist of three stages:
#  1. Data overview
#  2. Data preprocessing
#  3. Testing the hypotheses
#  
# [Back to Contents](#back)

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Title and introduction are essential parts of the project. Make sure you do not forget to include it in your further projects. 
#     
# It is optimal if introduction consists of:
#     
# - brief description of the situation;
# - goal of the project;
# - description of the data we are going to use.
# </div>
# 

# ## Stage 1. Data overview <a id='data_review'></a>
# 
# Open the data on Yandex.Music and explore it.

# You'll need `pandas`, so import it.

# In[1]:


# importing pandas
import pandas as pd


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> Needed library has been added </div>

# Read the file `music_project_en.csv` from the `/datasets/` folder and save it in the `df` variable:

# In[2]:


# reading the file and storing it to df
df = pd.read_csv('/datasets/music_project_en.csv')


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> The correct path to the file is specified: the slash at the beginning of the path is very important, as it indicates that you need to search for the file in the root folder. </div>

# Print the first 10 table rows:

# In[3]:


# obtaining the first 10 rows from the df table
df.head(10)


# Obtaining the general information about the table with one command:

# In[4]:


# obtaining general information about the data in df
df.info()


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Great - you've used a comprehensive set of methods to have a first look at the data.
#     

# The table contains seven columns. They all store the same data type: `object`.
# 
# According to the documentation:
# - `'userID'` — user identifier
# - `'Track'` — track title
# - `'artist'` — artist's name
# - `'genre'`
# - `'City'` — user's city
# - `'time'` — the exact time the track was played
# - `'Day'` — day of the week
# 
# We can see three issues with style in the column names:
# 1. Some names are uppercase, some are lowercase.
# 2. There are spaces in some names.
# 3. `Detect the third issue yourself and describe it here`. The column names should be in camel case, userID -> user_id.
# 
# The number of column values is different. This means the data contains missing values.
# 

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Perfect!

# ### Conclusions <a id='data_review_conclusions'></a> 
# 
# Each row in the table stores data on a track that was played. Some columns describe the track itself: its title, artist and genre. The rest convey information about the user: the city they come from, the time they played the track. 
# 
# It's clear that the data is sufficient to test the hypotheses. However, there are missing values.
# 
# To move forward, we need to preprocess the data.

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Please note that it is highly recommended to add a conclusion / summary after each section and describe briefly your observations and / or major outcomes of the analysis.

# [Back to Contents](#back)

# ## Stage 2. Data preprocessing <a id='data_preprocessing'></a>
# Correct the formatting in the column headers and deal with the missing values. Then, check whether there are duplicates in the data.

# ### Header style <a id='header_style'></a>
# Print the column header:

# In[5]:


# the list of column names in the df table
df.columns


# Change column names according to the rules of good style:
# * If the name has several words, use snake_case
# * All characters must be lowercase
# * Delete spaces

# In[6]:


# renaming columns
df = df.rename(columns={
    '  userID': 'user_id', 
    'Track': 'track',
    '  City  ': 'city',
    'Day': 'day'
})


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# This is a good way to rename the columns.

# Check the result. Print the names of the columns once more:

# In[7]:


# checking result: the list of column names
df.columns


# [Back to Contents](#back)

# ### Missing values <a id='missing_values'></a>
# First, find the number of missing values in the table. To do so, use two `pandas` methods:

# In[8]:


# calculating missing values
print(df.isna().sum()) 


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# The isna() method is selected to find the missing values, it's great!

# Not all missing values affect the research. For instance, the missing values in `track` and `artist` are not critical. You can simply replace them with clear markers.
# 
# But missing values in `'genre'` can affect the comparison of music preferences in Springfield and Shelbyville. In real life, it would be useful to learn the reasons why the data is missing and try to make up for them. But we do not have that opportunity in this project. So you will have to:
# * Fill in these missing values with markers
# * Evaluate how much the missing values may affect your computations

# Replace the missing values in `'track'`, `'artist'`, and `'genre'` with the string `'unknown'`. To do this, create the `columns_to_replace` list, loop over it with `for`, and replace the missing values in each of the columns:

# In[9]:


# looping over column names and replacing missing values with 'unknown'
columns_to_replace = ['track', 'artist', 'genre']
for row in columns_to_replace:
    df[row] = df[row].fillna('unknown')
#df.head(18)


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Well done, the loop for filling in the missing values is ready.

# Make sure the table contains no more missing values. Count the missing values again.

# In[10]:


# counting missing values
print(df.isna().sum()) 


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Now we can be absolutely sure that all the missing values have been filled it )

# [Back to Contents](#back)

# ### Duplicates <a id='duplicates'></a>
# Find the number of obvious duplicates in the table using one command:

# In[11]:


# counting clear duplicates
print(df.duplicated().sum())
# print(df[df.duplicated()])


# Call the `pandas` method for getting rid of obvious duplicates:

# In[12]:


# removing obvious duplicates
#duplicated_df = df[df.duplicated()]
#print(duplicated_df.head()) 
df = df.drop_duplicates()


# Count obvious duplicates once more to make sure you have removed all of them:

# In[13]:


# checking for duplicates
print(df.duplicated().sum())


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Great, you found and removed the duplicates. And did very thorough checks to make sure the duplicates are gone.

# Now get rid of implicit duplicates in the `genre` column. For example, the name of a genre can be written in different ways. Such errors will also affect the result.

# Print a list of unique genre names, sorted in alphabetical order. To do so:
# * Retrieve the intended DataFrame column 
# * Apply a sorting method to it
# * For the sorted column, call the method that will return all unique column values

# In[48]:


# viewing unique genre names
# print(df[120:129])
# df.info()
# print(sorted(df['genre'].unique()))   # old code, but still worked
print(df['genre'].sort_values().unique())
# Revised 10/9/23


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# This is good. 
#     

# <div class="alert alert-warning"; style="border-left: 7px solid gold">
# <b>⚠️ Reviewer's comment, v. 1</b> 
#     
# You could also use the pandas `sort_values()` method:
#     
#     df['genre'].sort_values().unique()

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 2</b> 
#     
# Perfect!

# Look through the list to find implicit duplicates of the genre `hiphop`. These could be names written incorrectly or alternative names of the same genre.
# 
# You will see the following implicit duplicates:
# * `hip`
# * `hop`
# * `hip-hop`
# 
# To get rid of them, declare the function `replace_wrong_genres()` with two parameters: 
# * `wrong_genres=` — the list of duplicates
# * `correct_genre=` — the string with the correct value
# 
# The function should correct the names in the `'genre'` column from the `df` table, i.e. replace each value from the `wrong_genres` list with the value in `correct_genre`.

# In[15]:


# function for replacing implicit duplicates
def replace_wrong_genres(wrong_genres, correct_genres):
    for wrong_genre in wrong_genres:
        df['genre'] = df['genre'].replace(wrong_genres, correct_genres)


# Call `replace_wrong_genres()` and pass it arguments so that it clears implicit duplcates (`hip`, `hop`, and `hip-hop`) and replaces them with `hiphop`:

# In[16]:


# removing implicit duplicates
duplicates_hiphop = ['hip', 'hop', 'hip-hop']
replace_wrong_genres(duplicates_hiphop, 'hiphop')


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Yes, this is what was needed!

# Make sure the duplicate names were removed. Print the list of unique values from the `'genre'` column:

# In[17]:


# checking for implicit duplicates
print(sorted(df['genre'].unique())) 


# [Back to Contents](#back)

# ### Conclusions <a id='data_preprocessing_conclusions'></a>
# We detected three issues with the data:
# 
# - Incorrect header styles
# - Missing values
# - Obvious and implicit duplicates
# 
# The headers have been cleaned up to make processing the table simpler.
# 
# All missing values have been replaced with `'unknown'`. But we still have to see whether the missing values in `'genre'` will affect our calculations.
# 
# The absence of duplicates will make the results more precise and easier to understand.
# 
# Now we can move on to testing hypotheses. 

# [Back to Contents](#back)

# ## Stage 3. Testing hypotheses <a id='hypotheses'></a>

# ### Hypothesis 1: comparing user behavior in two cities <a id='activity'></a>

# According to the first hypothesis, users from Springfield and Shelbyville listen to music differently. Test this using the data on three days of the week: Monday, Wednesday, and Friday.
# 
# * Divide the users into groups by city.
# * Compare how many tracks each group played on Monday, Wednesday, and Friday.
# 

# For the sake of practice, perform each computation separately. 
# 
# Evaluate user activity in each city. Group the data by city and find the number of songs played in each group.
# 
# 

# In[18]:


# Counting up the tracks played in each city
print(df.groupby('city')['user_id'].count())
shelbyville_data = df[df['city'] == 'Shelbyville']
springfield_data = df[df['city'] == 'Springfield']
print(shelbyville_data['user_id'].count())
print(springfield_data['user_id'].count())


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Well done and it's good that you are grouped by the second column to show the data rationally

# Springfield has more tracks played than Shelbyville. But that does not imply that citizens of Springfield listen to music more often. This city is simply bigger, and there are more users.
# 
# Now group the data by day of the week and find the number of tracks played on Monday, Wednesday, and Friday.
# 

# In[44]:


# Calculating tracks played on each of the three days
# print(springfield_data.head(10))
print('Spingfield Tracks Played: \n', springfield_data.groupby('day')['user_id'].count())
print('\nShelbyville Tracks Played: \n',shelbyville_data.groupby('day')['user_id'].count())
# print(shelbyville_data[shelbyville_data['day'] == 'Friday']['user_id'].count())
# Revised 10/9/23


# <div class="alert alert-danger"; style="border-left: 7px solid red">
# <b>⛔️ Reviewer's comment, v. 1</b> 
# 
# To get rid of the repeating values, please specify one column to be displayed.

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 2</b> 
#     
# Now it is done absolutelly correctly!

# Wednesday is the quietest day overall. But if we consider the two cities separately, we might come to a different conclusion.

# You have seen how grouping by city or day works. Now write a function that will group by both.
# 
# Create the `number_tracks()` function to calculate the number of songs played for a given day and city. It will require two parameters:
# * day of the week
# * name of the city
# 
# In the function, use a variable to store the rows from the original table, where:
#   * `'day'` column value is equal to the `day` parameter
#   * `'city'` column value is equal to the `city` parameter
# 
# Apply consecutive filtering with logical indexing.
# 
# Then calculate the `'user_id'` column values in the resulting table. Store the result to a new variable. Return this variable from the function.

# In[20]:


# <creating the function number_tracks()>
# We'll declare a function with two parameters: day=, city=.
# Let the track_list variable store the df rows where
# the value in the 'day' column is equal to the day= parameter and, at the same time, 
# the value in the 'city' column is equal to the city= parameter (apply consecutive filtering 
# with logical indexing).
# Let the track_list_count variable store the number of 'user_id' column values in track_list
# (found with the count() method).
# Let the function return a number: the value of track_list_count.

# The function counts tracked played for a certain city and day.
# It first retrieves the rows with the intended day from the table,
# then filters out the rows with the intended city from the result,
# then finds the number of 'user_id' values in the filtered table,
# then returns that number.
# To see what it returns, wrap the function call in print().
def number_tracks(day, city):
    track_list = df[df['city'] == city]
    track_list_count = track_list[track_list['day'] == day]['user_id'].count()
    return track_list_count
    
# print(df[df['city'] == 'Shelbyville'])
print(number_tracks('Friday', 'Shelbyville'))


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Excellent function, works as required

# Call `number_tracks()` six times, changing the parameter values, so that you retrieve the data on both cities for each of the three days.

# In[21]:


# the number of songs played in Springfield on Monday
print(number_tracks('Monday', 'Springfield'))


# In[22]:


# the number of songs played in Shelbyville on Monday
print(number_tracks('Monday', 'Shelbyville'))


# In[23]:


# the number of songs played in Springfield on Wednesday
print(number_tracks('Wednesday', 'Springfield'))


# In[24]:


# the number of songs played in Shelbyville on Wednesday
print(number_tracks('Wednesday', 'Shelbyville'))


# In[25]:


# the number of songs played in Springfield on Friday
print(number_tracks('Friday', 'Springfield'))


# In[26]:


# the number of songs played in Shelbyville on Friday
print(number_tracks('Friday', 'Shelbyville'))


# Use `pd.DataFrame` to create a table, where
# * Column names are: `['city', 'monday', 'wednesday', 'friday']`
# * The data is the results you got from `number_tracks()`

# In[45]:


# table with results
days_of_week = ['Monday', 'Wednesday', 'Friday']
cities = ['Shelbyville', 'Springfield']
data_tracks = []

for city in cities:
    track_row = [city]
    for day in days_of_week:
        track_row.append(number_tracks(day, city))
    data_tracks.append(track_row)

# print(data_tracks)

headers = ['city', 'monday', 'wednesday', 'friday']
springfield_vs_shelbyville = pd.DataFrame(data=data_tracks, columns=headers)
display(springfield_vs_shelbyville)
# Revised 10/9/23


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# And your table contains correct data. Perfect!

# <div class="alert alert-warning"; style="border-left: 7px solid gold">
# <b>⚠️ Reviewer's comment, v. 1</b> 
#     
# The result would be shown a little better if you use `display` instead of `print` :)

# **Conclusions**
# 
# The data reveals differences in user behavior:
# 
# - In Springfield, the number of songs played peaks on Mondays and Fridays, while on Wednesday there is a decrease in activity.
# - In Shelbyville, on the contrary, users listen to music more on Wednesday. User activity on Monday and Friday is smaller.
# 
# So the first hypothesis seems to be correct.

# [Back to Contents](#back)

# ### Hypothesis 2: music at the beginning and end of the week <a id='week'></a>

# According to the second hypothesis, on Monday morning and Friday night, citizens of Springfield listen to genres that differ from ones users from Shelbyville enjoy.

# Get tables (make sure that the name of your combined table matches the DataFrame given in the two code blocks below):
# * For Springfield — `spr_general`
# * For Shelbyville — `shel_general`

# In[28]:


# create the spr_general table from the df rows, 
# where the value in the 'city' column is 'Springfield'
spr_general = df[df['city'] == 'Springfield']
print(spr_general)


# In[29]:


# create the shel_general from the df rows,
# where the value in the 'city' column is 'Shelbyville'
shel_general = df[df['city'] == 'Shelbyville']
print(shel_general)


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# 
# Well done - you created separate dataframes with Springfield and Shelbyville data.

# Write the `genre_weekday()` function with four parameters:
# * A table for data (`df`)
# * The day of the week (`day`)
# * The first timestamp, in 'hh:mm' format (`time1`)
# * The last timestamp, in 'hh:mm' format (`time2`)
# 
# The function should return info on the 15 most popular genres on a given day within the period between the two timestamps.

# In[30]:


# 1) Let the genre_df variable store the rows that meet several conditions:
#    - the value in the 'day' column is equal to the value of the day= argument
#    - the value in the 'time' column is greater than the value of the time1= argument
#    - the value in the 'time' column is smaller than the value of the time2= argument
#    Use consecutive filtering with logical indexing.

# 2) Group genre_df by the 'genre' column, take one of its columns, 
#    and use the count() method to find the number of entries for each of 
#    the represented genres; store the resulting Series to the
#    genre_df_count variable

# 3) Sort genre_df_count in descending order of frequency and store the result
#    to the genre_df_sorted variable

# 4) Return a Series object with the first 15 genre_df_sorted value - the 15 most
#    popular genres (on a given day, within a certain timeframe)

# Write your function here
def genre_weekday(df, day, time1, time2):
    # consecutive filtering
    # Create the variable genre_df which will store only those df rows where the day is equal to day=
    genre_df = df[df['day'] == day] # write your code here

    # filter again so that genre_df will store only those rows where the time is smaller than time2=
    genre_df = genre_df[genre_df['time'] < time2] # write your code here

    # filter once more so that genre_df will store only rows where the time is greater than time1=
    genre_df = genre_df[genre_df['time'] > time1] # write your code here

    # group the filtered DataFrame by the column with the names of genres, take the genre column, and find the number of rows for each genre with the count() method
    genre_df_count = genre_df.groupby('genre')['genre'].count() # write your code here
    # print(genre_df_count)
    # sort the result in descending order (so that the most popular genres come first in the Series object)
    genre_df_sorted = genre_df_count.sort_values(ascending=False) # write your code here

    # we will return the Series object storing the 15 most popular genres on a given day in a given timeframe
    return genre_df_sorted[:15]


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# 
# This is the most difficult function in the project. You wrote it flawlessly. 

# Compare the results of the `genre_weekday()` function for Springfield and Shelbyville on Monday morning (from 7AM to 11AM) and on Friday evening (from 17:00 to 23:00):

# In[31]:


# calling the function for Monday morning in Springfield (use spr_general instead of the df table)
print(genre_weekday(spr_general, 'Monday', '07:00', '11:00'))


# In[32]:


# calling the function for Monday morning in Shelbyville (use shel_general instead of the df table)
print(genre_weekday(shel_general, 'Monday', '07:00', '11:00'))


# In[33]:


# calling the function for Friday evening in Springfield
print(genre_weekday(spr_general, 'Friday', '17:00', '23:00'))


# In[34]:


# calling the function for Friday evening in Shelbyville
print(genre_weekday(shel_general, 'Friday', '17:00', '23:00'))


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Well done, you now have data to make a conclusion for the hypothesis.

# **Conclusion**
# 
# Having compared the top 15 genres on Monday morning, we can draw the following conclusions:
# 
# 1. Users from Springfield and Shelbyville listen to similar music. The top five genres are the same, only rock and electronic have switched places.
# 
# 2. In Springfield, the number of missing values turned out to be so big that the value `'unknown'` came in 10th. This means that missing values make up a considerable portion of the data, which may be a basis for questioning the reliability of our conclusions.
# 
# For Friday evening, the situation is similar. Individual genres vary somewhat, but on the whole, the top 15 is similar for the two cities.
# 
# Thus, the second hypothesis has been partially proven true:
# * Users listen to similar music at the beginning and end of the week.
# * There is no major difference between Springfield and Shelbyville. In both cities, pop is the most popular genre.
# 
# However, the number of missing values makes this result questionable. In Springfield, there are so many that they affect our top 15. Were we not missing these values, things might look different.

# [Back to Contents](#back)

# ### Hypothesis 3: genre preferences in Springfield and Shelbyville <a id='genre'></a>
# 
# Hypothesis: Shelbyville loves rap music. Springfield's citizens are more into pop.

# Group the `spr_general` table by genre and find the number of songs played for each genre with the `count()` method. Then sort the result in descending order and store it to `spr_genres`.

# In[35]:


# on one line: group the spr_general table by the 'genre' column, 
# count the 'genre' values with count() in the grouping, 
# sort the resulting Series in descending order, and store it to spr_genres
spr_genres = spr_general.groupby('genre')['genre'].count().sort_values(ascending=False)


# Print the first 10 rows from `spr_genres`:

# In[36]:


# printing the first 10 rows of spr_genres
print(spr_genres.head(10))


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Excellent - you made a new dataframe exactly as was required, in two lines. 

# Now do the same with the data on Shelbyville.
# 
# Group the `shel_general` table by genre and find the number of songs played for each genre. Then sort the result in descending order and store it to the `shel_genres` table:
# 

# In[37]:


# on one line: group the shel_general table by the 'genre' column, 
# count the 'genre' values in the grouping with count(), 
# sort the resulting Series in descending order and store it to shel_genres
shel_genres = shel_general.groupby('genre')['genre'].count().sort_values(ascending=False)


# Print the first 10 rows of `shel_genres`:

# In[38]:


# printing the first 10 rows from shel_genres
print(shel_genres.head(10))


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Absolutely accurate!

# **Conclusion**

# The hypothesis has been partially proven true:
# * Pop music is the most popular genre in Springfield, as expected.
# * However, pop music turned out to be equally popular in Springfield and Shelbyville, and rap wasn't in the top 5 for either city.
# 

# [Back to Contents](#back)

# # Findings <a id='end'></a>

# We have tested the following three hypotheses:
# 
# 1. User activity differs depending on the day of the week and from city to city. 
# 2. On Monday mornings, Springfield and Shelbyville residents listen to different genres. This is also true for Friday evenings. 
# 3. Springfield and Shelbyville listeners have different preferences. In Springfield, they prefer pop, while Shelbyville has more rap fans.
# 
# After analyzing the data, we concluded:
# 
# 1. User activity in Springfield and Shelbyville depends on the day of the week, though the cities vary in different ways. 
# 
# The first hypothesis is fully accepted.
# 
# 2. Musical preferences do not vary significantly over the course of the week in both Springfield and Shelbyville. We can see small differences in order on Mondays, but:
# * In Springfield and Shelbyville, people listen to pop music most.
# 
# So we can't accept this hypothesis. We must also keep in mind that the result could have been different if not for the missing values.
# 
# 3. It turns out that the musical preferences of users from Springfield and Shelbyville are quite similar.
# 
# The third hypothesis is rejected. If there is any difference in preferences, it cannot be seen from this data.
# 
# ### Note 
# In real projects, research involves statistical hypothesis testing, which is more precise and more quantitative. Also note that you cannot always draw conclusions about an entire city based on the data from just one source.
# 
# You will study hypothesis testing in the sprint on statistical data analysis.

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Overall conclusion is an important part, where we should include the summary of the outcomes of the project.

# <div class="alert alert-info"; style="border-left: 7px solid blue">
# <b>Student's comment</b>
#     
#     Made changes to the code with the red comments. 
#     I made note
#     Thanks again for reviewing my project! 

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 2</b> 
#     
# Mark, I was glad to help you!

# [Back to Contents](#back)
