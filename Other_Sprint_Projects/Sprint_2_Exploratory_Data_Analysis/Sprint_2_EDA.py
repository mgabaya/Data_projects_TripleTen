#!/usr/bin/env python
# coding: utf-8

# <div style="border:solid green 2px; padding: 20px">
#     
# <b>Mark, Hello!</b> We're glad to see you in code-reviewer territory. You've done a great job on the project, but let's get to know each other and make it even better! We have our own atmosphere here and a few rules:
# 
# 
# 1. My name is Alexander Matveevsky. I work as a code reviewer, and my main goal is not to point out your mistakes, but to share my experience and help you become a data analyst.
# 2. We speak on a first-come-first-served basis.
# 3. if you want to write or ask a question, don't be shy. Just choose your color for your comment.  
# 4. this is a training project, you don't have to be afraid of making a mistake.  
# 5. You have an unlimited number of attempts to pass the project.  
# 6. Let's Go!
# 
# 
# ---
# I'll be color-coding comments, please don't delete them:
# 
# <div class="alert alert-block alert-danger">✍
#     
# 
# __Reviewer's comment №1__
# 
# Needs fixing. The block requires some corrections. Work can't be accepted with the red comments.
# </div>
#     
# ---
# 
# <div class="alert alert-block alert-warning">📝
#     
# 
# __Reviewer's comment №1__
# 
# 
# Remarks. Some recommendations.
# </div>
# 
# ---
# 
# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №1__
# 
# Success. Everything is done succesfully.
# </div>
#     
# ---
#     
# I suggest that we work on the project in dialogue: if you change something in the project or respond to my comments, write about it. It will be easier for me to track changes if you highlight your comments:   
#     
# <div class="alert alert-info"> <b>Student сomments:</b> Student answer..</div>
#     
# All this will help to make the recheck of your project faster. If you have any questions about my comments, let me know, we'll figure it out together :)   
#     
# ---

# 

# <div class="alert alert-block alert-danger">✍
#     
# 
# __Reviewer's comment №1__
# 
# An excellent practice is to describe the goal and main steps in your own words (a skill that will help a lot on a final project). It would be good to add the progress and purpose of the study.

# <div class="alert alert-info"> <b>Student сomments:</b> Hello Alexander! Thank you so much for taking the time to review and provide feedback! I'll use this blue code block to ask or comment on your feedback. My goal and steps are outlined below</div> 
# 
# 
# 

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №2__
# 
# Mark, hello. Thank you for your feedback and edits

# <h2>Objective</h2>
# With the data provided on Instacart orders, we will address the multiple data sets and large data sets by: 
# 1. Cleaning up data before processing it by handling null values, duplicate values, and changing data types when applicable
# 2. Utilizing data visualizations, bar graphs or histograms
# 3. Merging data sets to gain more insight
# 4. Advanced filtering to find any other trends in the data
# 

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


# In[2]:


# reading in the csv and inspecting the data
df_instacart_orders = pd.read_csv('/datasets/instacart_orders.csv', sep=';')
print(df_instacart_orders.head())
df_instacart_orders.info()
# df_instacart_orders.describe()


# <div class="alert alert-block alert-warning">📝
#     
# 
# __Reviewer's comment №1__
# 
# 
# I suggest separating logically different things:
# 
#     - library imports
#     - data opening (this may not be a trivial process)
#     - working code
#     
# This approach will improve code readability, reduce the time of searching for errors, if there are any, and exclude code re-execution
# </div>

# <div class="alert alert-info"> <b>Student сomments:</b> To clarify on data opening, should I have grouped all the read_csv lines together and then call the head and info methods in a separate code block as the working code?</div>

# <div class="alert alert-block alert-warning">📝
#     
# 
# __Reviewer's comment №2__
# 
# 
# According to RER8 standards they should be separated, below show:

# import pandas as pd
# 
# from matplotlib import pyplot as plt
# 
# import numpy as np

# ---

# --reading in the csv and inspecting the data
# 
# df_instacart_orders = pd.read_csv('/datasets/instacart_orders.csv', sep=';')

# ---

# print(df_instacart_orders.head())
# 
# df_instacart_orders.info()
# 
# --df_instacart_orders.describe()

# 

# 

# 

# 

# In[3]:


# reading in the csv and inspecting the data
df_products = pd.read_csv('/datasets/products.csv', sep=';')
print(df_products.head())
df_products.info()


# In[4]:


# reading in the csv and inspecting the data
df_order_products = pd.read_csv('/datasets/order_products.csv', sep=';')
print(df_order_products.head())
df_order_products.info(show_counts=True)
# added show_counts=True for large data sets


# In[5]:


# reading in the csv and inspecting the data
df_aisles = pd.read_csv('/datasets/aisles.csv', sep=';')  # 2 cols
print(df_aisles.head())
df_aisles.info()


# In[6]:


df_departments = pd.read_csv('/datasets/departments.csv', sep=';') # 2 cols
print(df_departments.head())
df_departments.info()


# <div class="alert alert-block alert-warning">📝
#     
# 
# __Reviewer's comment №1__
# 
# It would be good to comment on the data somehow

# <div class="alert alert-info"> <b>Student сomments:</b> added a summary on the data below.</div>
#     
# When looking at the info on the data, some data sets have missing values, like in the days_since_prior_order column for the df_instacart_orders dataframe. This will be handled in the preprocessing later, along with duplicate values.

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №2__
# 
# Great

# ## Find and remove duplicate values (and describe why you make your choices)

# ### `orders` data frame

# In[7]:


# Check for duplicated orders
print(df_instacart_orders[df_instacart_orders['order_id'].duplicated()]['order_id'].count())  # Counting that there is 15 duplicates
df_instacart_orders_sorted = df_instacart_orders[df_instacart_orders.duplicated()]
print(df_instacart_orders_sorted.sort_values(by='order_id'))  # Printing the order_id's that are duplicated
print(df_instacart_orders[df_instacart_orders['order_id'] == 2845099])  # Made sure this order id has a duplicate; will test again later 


# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №1__
# 
# Duplicate checking is the basis of data preprocessing

# In[8]:


# Check for all orders placed Wednesday at 2:00 AM
# Our assumption will be that 0 is Sunday through 6 for Saturday; 3 is Wednesday in this case
# For the hours, 0 is 12am and 23 is 11pm; 2 is 2am in this case
# Filter with 2 conditions using the and operator (&)
wed_at_2am = df_instacart_orders[(df_instacart_orders['order_dow'] == 3) & (df_instacart_orders['order_hour_of_day'] == 2)]
print(wed_at_2am)


# In[9]:


# Remove duplicate orders
# Since there were originally 478967 rows and 15 duplicate row, there should be 478952 rows left
# Using the drop_duplicates method
df_instacart_orders.drop_duplicates(keep="first", inplace=True)


# In[10]:


# Double check for duplicate rows
print(df_instacart_orders[df_instacart_orders.duplicated()])  # Checking to see if there is an empty dataframe
# print(df_instacart_orders[df_instacart_orders['order_id'].duplicated()]['order_id'].count()) # count is 0 means duplicates is 0
print(df_instacart_orders[df_instacart_orders['order_id'] == 2845099])  # Made sure there's only one record left for this order id 
df_instacart_orders.info()  # Total entries went down tto 478952, 15 rows less


# In[11]:


# Double check for duplicate order IDs only
print(df_instacart_orders[df_instacart_orders['order_id'].duplicated()]['order_id'].count())


# The above shows there's no more duplicated order IDs in the instacart orders dataframe.

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №1__
# 
# Great

# ### `products` data frame

# In[12]:


# Check for fully duplicate rows
print(df_products[df_products.duplicated()])  # checking to see if complete rows are duplicated


# In[13]:


# Check for just duplicate product IDs
print(df_products[df_products['product_id'].duplicated()])  # checking to see if there are duplicated product ids, should be empty
df_products['product_id'].value_counts().sum()  # There should be 49694 rows showing no product ids were duplicated


# In[14]:


# Check for just duplicate product names (convert names to lowercase to compare better)
print(df_products['product_name'].value_counts().sum())  # Testing to see if there are 49694 names; should be less due to duplicates
df_products['product_name_lower'] = df_products['product_name'].str.lower() # Create a new column with all the product names in lowercase
df_products_duplicates = df_products[df_products['product_name_lower'].duplicated()]  # Printing out any duplicated lowercase names
print(df_products_duplicates)


# In[15]:


# Check for duplicate product names that aren't missing
na_df_products = df_products[~df_products['product_name_lower'].isna()]  # Filtered out rows with nulls
print(na_df_products)
# print(na_df_products[na_df_products['product_name_lower'].duplicated()])
# Checking if there's any duplicated product names in lowercase
print(na_df_products[na_df_products['product_name_lower'].duplicated()]['product_name_lower'].value_counts().sort_values())


# There is 103 items with the same name that is not missing (NaN).

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №1__
# 
# That's right, well done!)

# ### `departments` data frame

# In[16]:


print(df_departments[df_departments.duplicated()])  # Checking whole rows for duplicates
print(df_departments[df_departments['department'].duplicated()])  # Checking for duplicate departments
df_departments['department_lower'] = df_departments['department'].str.lower()  # Making departments lowercase
print(df_departments[df_departments['department_lower'].duplicated()])  # Rechecking deparment names for duplicates


# In[ ]:





# There are no duplicate departments.

# ### `aisles` data frame

# In[17]:


print(df_aisles[df_aisles.duplicated()])  # Checking whole rows for duplicates
print(df_aisles[df_aisles['aisle'].duplicated()])  # Checking for duplicate aisle names


# In[18]:


df_aisles['aisle_lower'] = df_aisles['aisle'].str.lower()  # Making aisle names lowercase
print(df_aisles[df_aisles['aisle_lower'].duplicated()])  # Rechecking aisle names for duplicates


# There are no duplicate aisles.

# ### `order_products` data frame

# In[19]:


# Check for fullly duplicate rows
# 4545007 rows
print(df_order_products[df_order_products.duplicated()])  # shows empty dataframe
df_order_products.info(show_counts=True)


# In[20]:


# Double check for any other tricky duplicates
# checking to see if an order ID tried to add the same product ID
print(df_order_products[df_order_products[['order_id', 'product_id']].duplicated()])  
# print(df_order_products[df_order_products['product_id'].duplicated()])
# print(df_order_products[df_order_products['order_id'] == 1717990])
# print(len(df_order_products[df_order_products['order_id'] == 1717990]['product_id'].unique()))


# The check for tricky duplicates tests to see if an order ID has a product ID used more than once. Since none came up, there are no duplicates.

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №1__
# 
# all right

# ## Find and remove missing values
# 

# ### `products` data frame

# In[21]:


df_products.isna().sum()  # Checking for null values


# In[22]:


# Are all of the missing product names associated with aisle ID 100?
df_products_na = df_products[df_products['product_name'].isna()]
print('List of aisle ID\'s for missing product names:', df_products_na['aisle_id'].unique())
print('Number of aisle ID\'s for missing product names:', df_products_na['aisle_id'].nunique())


# In[23]:


# Are all of the missing product names associated with department ID 21?
print('List of department ID\'s for missing product names:', df_products_na['department_id'].unique())
print('Number of department ID\'s for missing product names:', df_products_na['department_id'].nunique())


# In[24]:


# What is this ailse and department?
print('Aisle ID:', df_products_na['aisle_id'].unique(), ' and Aisle Name:', df_aisles.iloc[99, 1]) # df_aisles[df_aisles['aisle_id'] == 100]['aisle'])
print('Department ID:', df_products_na['department_id'].unique(), ' and Department Name:', df_departments.iloc[20, 1])  # [df_departments['department_id'] == 21]['department'])


# In[25]:


# Fill missing product names with 'Unknown'
# Replaced Null values with 'Unknown'
df_products['product_name'] = df_products['product_name'].fillna('Unknown')
df_products['product_name_lower'] = df_products['product_name_lower'].fillna('Unknown')
df_products.isna().sum()


# The data in the Products data has been cleaned up. There are no null values left.

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №1__
# 
# You chose the most correct way to replace the omissions in the categorical column. Well done!)

# ### `orders` data frame

# In[26]:


df_instacart_orders.info()  # Checking for null values
df_instacart_orders.isna().sum()


# In[27]:


# Are there any missing values where it's not a customer's first order?
# df_instacart_orders_na = df_instacart_orders.query("days_since_prior_order.isna()")
df_instacart_orders_na = df_instacart_orders[~df_instacart_orders['days_since_prior_order'].isna()]
print(df_instacart_orders_na.isna().sum())
print(df_instacart_orders_na.info())


# I assumed the null values in the days_since_prior_order column meant it was the customer's first order. Then I filtered out the dataframe where there were null values in the 'days_since_prior_order' column. Calling the info method on this new, filtered variable shows that there are no null values in the other columns.

# ### `order_products` data frame

# In[28]:


df_order_products.info()
# print(df_order_products.head())
# print(df_order_products.sample())
df_order_products.isna().sum()  # Checking for null values


# In[29]:


# What are the min and max values in this column?
print('Min:', df_order_products['add_to_cart_order'].min())
print('Max', df_order_products['add_to_cart_order'].max())
# print(df_order_products.query("add_to_cart_order == 0"))
df_order_products['add_to_cart_order'].describe()  # Verifying the min and max with the describe method


# In[30]:


# Save all order IDs with at least one missing value in 'add_to_cart_order'
add_to_cart_order_na = df_order_products.query("add_to_cart_order.isna()")
print(add_to_cart_order_na['order_id'])


# In[31]:


# Do all orders with missing values have more than 64 products?
# assigning all the order_id numbers into a list
list_of_ids = add_to_cart_order_na['order_id'].tolist()
# Creating a function to check if any of the order_ids have an order with more than 64 products; returns the total number of orders above 64
def count_number_of_products(ids):
    count = 0
    for id in ids:
        max_checker = df_order_products[df_order_products['order_id'] == id]['add_to_cart_order'].max() 
        if max_checker > 64:
            count += 1
    return count
count_number_of_products(list_of_ids)


# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №1__
# Good for you for using your own functions

# In[32]:


# Replace missing values with 999 and convert column to integer type
# df_order_products['add_to_cart_order'] = df_order_products['add_to_cart_order'].fillna(999)  # Replacing with 999
print(df_order_products.dtypes)  # Showing add_to_cart_order is still a float
df_order_products['add_to_cart_order'] = df_order_products['add_to_cart_order'].astype('Int64', errors='ignore')  # Changing to int64
print(df_order_products.dtypes) # Showing add_to_cart_order is integer type

df_order_products.info(show_counts=True)


# The order_products data has been cleaned up amd the add_to_cart_order column has been changed to an int64 type. 
# <!--by replacing the null values in the add_to_cart_order column with 999.-->

# <div class="alert alert-block alert-warning">📝
#     
# 
# __Reviewer's comment №1__
# 
# 
# Replacing it with 999/-1 etc is not really good. Tip for the future, look, it's not good to try to fill in the blanks left and right. In real work - replace with zero now, then forget and average count a month later. Plus there may be times when a bonus purchase or gift, will cost zero. In general, if it is possible to leave skips as skips - and here it can be done, it is better to do so, the results, even if we want to calculate something - it will not affect the results
# 
# It is better to use this code construction:
#     
#     data[''] = data[''].astype('Int64', errors='ignore')
# </div>

# <div class="alert alert-info"> <b>Student сomments:</b> The previous code block had the instructions "Replace missing values with 999 and convert column to integer type" so that's why I chose 999 originally.
#     
# Your comment makes sense, too, so I've updated the code and it looks like it didn't effect anything else later on.</div>

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №2__
# 
# Okay

# # [A] Easy (must complete all to pass)

# ### [A1] Verify that the `'order_hour_of_day'` and `'order_dow'` values in the `orders` tables are sensible (i.e. `'order_hour_of_day'` ranges from 0 to 23 and `'order_dow'` ranges from 0 to 6)

# In[33]:


print(df_instacart_orders.describe())
print('Hours of the day min:', df_instacart_orders['order_hour_of_day'].min(), 'Max:', df_instacart_orders['order_hour_of_day'].max())
print('Day of week min:', df_instacart_orders['order_dow'].min(), 'Max:', df_instacart_orders['order_dow'].max())


# We will assume 0 is 12am and goes up to 23 as 11pm.
# Similarly, 0 is Sunday through 6 as Saturday for the days of the week.

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №1__
# 
# All right

# ### [A2] What time of day do people shop for groceries?

# In[34]:


hours_of_day_counts = df_instacart_orders['order_hour_of_day'].value_counts()
print(hours_of_day_counts)
df_instacart_orders['order_hour_of_day'].plot(
    kind='hist',
    bins=24,
    title='When Do People Like to Shop?'    
)
plt.xlabel('Hour of the Day')
plt.show()


# The chart shows that users like to shop around the morning beginning at 10am through 5pm.

# ### [A3] What day of the week do people shop for groceries?

# In[35]:


dow_orders = df_instacart_orders['order_dow'].value_counts()
print(dow_orders)
#dow_orders.plot(
#    title='Days of the Week Ordering', 
#    kind='bar',
#    ylabel='Number of Orders'
#)
#plt.show()
df_instacart_orders['order_dow'].plot(
    kind='hist',
    bins=7,
    title='When Do People Like to Shop?'    
)
plt.xlabel('Day of the Week')
plt.show()


# The chart shows that users like to shop at the beginning of the week on Sunday (0) and Monday (1).

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №1__
# 
# You're right to sign the charts

# <div class="alert alert-info"> <b>Student сomments:</b> Thanks for verifying this! Also, can you clarify what you mean by "sign the charts"?</div>

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №2__
# 
# Using  `title = ''`

# ### [A4] How long do people wait until placing another order?

# In[36]:


days_since_prior_order_counts = df_instacart_orders[~df_instacart_orders['days_since_prior_order'].isna()]
print(days_since_prior_order_counts['days_since_prior_order'].value_counts())
days_since_prior_order_counts['days_since_prior_order'].mean()


# The table shows that people reorder the most after a month, 30 days, and around a week.
# The mean shows about an average about 11 days, but the distribution below shows a spread of values.

# In[37]:


days_since_prior_order_counts['days_since_prior_order'].plot(
    kind='hist',
    bins=30,
    title='Days Since Reordering'
)
plt.xlabel('Number of Days')
plt.show()
# df_instacart_orders['days_since_prior_order'].hist()


# The histogram shows that people reorder the most after a month, 30 days, and around a week's time at 6 and 7 days apart.

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №1__
# 
# The 0 values probably correspond to customers who placed more than one order on the same day.

# # [B] Medium (must complete all to pass)

# ### [B1] Is there a difference in `'order_hour_of_day'` distributions on Wednesdays and Saturdays? Plot the histograms for both days and describe the differences that you see.

# In[38]:


# Distributions for Wednesday and Saturday
# Filtering into two variables for Wednesday and Saturday, respectively.
df_wednesday = df_instacart_orders[df_instacart_orders['order_dow'] == 3]
df_saturday = df_instacart_orders[df_instacart_orders['order_dow'] == 5]
print(df_wednesday['order_hour_of_day'].value_counts())
print(df_saturday['order_hour_of_day'].value_counts())


# In[39]:


# histogram
df_wednesday['order_hour_of_day'].plot(
    kind='hist',
    bins=24,
    title='Wednesday vs. Saturday Shopping Hours',
    ylabel='Frequency',
    xlabel='Hour of Day'
)
df_saturday['order_hour_of_day'].plot(
    kind='hist',
    bins=24,
    alpha=0.3
)
plt.legend(['Wednesday', 'Saturday'])
plt.xlabel('Hour of Day')
plt.show()


# In[ ]:





# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №1__
# 
# You're right to sign the charts

# Both Wednesday and Saturday have their peak hours around 10am and 3pm. The shape of their distributions are similar with the only difference is that Saturday has more orders in the daytime, around 7am to 3pm.

# ### [B2] What's the distribution for the number of orders per customer?

# In[40]:


orders_per_user = df_instacart_orders.groupby('user_id')['order_id'].count().sort_values(ascending=False) 
print(orders_per_user)
orders_per_user.plot(
    kind='hist',
    title='How Often Customers Order',
    bins=28,
    figsize=[9, 6]
)
plt.xlabel("Orders Per Customer")
plt.show()


# In[41]:


# orders_per_user.sort_values(ascending=False)
# df_instacart_orders['user_id']['order_id']


# <div class="alert alert-block alert-danger">✍
#     
# 
# __Reviewer's comment №1__
# 
# ![image.png](attachment:image.png)
#     
# Some blocks of code are not working. Please see what went wrong. Before submitting a project, you should check if the code works - you can do this by clicking on the Jupiter Hub Kernel and Restart & Run All panels

# <div class="alert alert-info"> <b>Student сomments:</b> That code block was meant to be commented out as I was testing some code. Thanks for looking out!</div>

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №2__
# 
# Thank you for the correction

# The distribution shows that most customers order 1 to 4 times.

# ### [B3] What are the top 20 popular products (display their id and name)?

# In[42]:


print(df_order_products.groupby('product_id')['order_id'].count().sort_values(ascending=False))  # Getting the counts for each product ID
#print(df_order_products[df_order_products['product_id']==11440].count())


# In[43]:


# Merging the tables
most_popular = df_order_products.merge(df_products,
                             on='product_id',
                             )
print(most_popular.sample(10))


# In[44]:


# Printing out the 20 most popular with name and ID
product_in_orders = most_popular.groupby(['product_id', 'product_name'])['order_id'].count().sort_values(ascending=False)
print(product_in_orders.head(20))


# Organic Fruits and Vegetables top the list of the most popular prodducts ordered.

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №2__
# 
# All right

# # [C] Hard (must complete at least two to pass)

# ### [C1] How many items do people typically buy in one order? What does the distribution look like?

# In[45]:


# Checking an order for how many items are in it
print(df_order_products[df_order_products['order_id']==1959075])


# In[46]:


# Get the counts of product_ids for each order_id
df_order_products_counts = df_order_products.groupby('order_id')['product_id'].count()
print(df_order_products_counts.sort_values())


# In[47]:


# Histogram of the distribution

df_order_products_counts.plot(
    kind='hist',
    bins=20,
    title='Amount People Buy Per Order',
    figsize=[10, 7]
)
plt.xlabel('Products Per Order')
plt.show()


# The histogram shows the majority of orders have less than 20 unique products per order.

# ### [C2] What are the top 20 items that are reordered most frequently (display their names and product IDs)?

# In[48]:


# Filtering out rows to just the reordered products
reordered_prods = df_order_products[df_order_products['reordered'] == 1]
reordered_prods.info(show_counts=True)
reordered_prods.head()


# In[49]:


print(reordered_prods.groupby('product_id')['order_id'].count().sort_values(ascending=False))


# In[50]:


# df_products[df_products['product_id']==21137]
df_merged = reordered_prods.merge(df_products,
                             on='product_id',
                             )
# df_merged.info()  # 2683838
# df_merged.head()


# In[51]:


top_20_reordered = df_merged.groupby(['product_id', 'product_name'])['order_id'].count().sort_values(ascending=False)
print(top_20_reordered.head(20))
# top_twenty = pd.Series(top_20_reordered)
# top_twenty.columns=['A','B','c']
# print(top_twenty)


# Just like earlier, the top 20 list for most reordered products are organic fruits and vegetables. Similarly, bananas are the top products again.

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №2__
# 
# I agree

# ### [C3] For each product, what proportion of its orders are reorders?

# In[ ]:





# In[ ]:





# In[ ]:





# ### [C4] For each customer, what proportion of their products ordered are reorders?

# In[ ]:





# In[ ]:





# ### [C5] What are the top 20 items that people put in their carts first? 

# In[ ]:





# In[ ]:





# In[ ]:





# <h1>Findings:
#     
# 1. The beginning of the week and midday tends to be the most popular times to order.
# 2. Fruits and vegetables are the most ordered and reordered products.

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №1__
# 
# Well done for describing the final conclusion

# <div class="alert alert-info"> <b>Student сomments:</b> Thanks again for reviewing my project! </div>

# <div class="alert alert-block alert-success">✔️
#     
# 
# __Reviewer's comment №2__
# 
# 
# Otherwise it's great😊. Your project is begging for github =)   
#     
# Congratulations on the successful completion of the project 😊👍
# And I wish you success in new works 😊
