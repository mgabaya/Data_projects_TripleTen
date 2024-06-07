# Data-projects-TripleTen
Data projects (TripleTen)


Sprint 4 project: Software Development Tools
https://github.com/mgabaya/vehicles_sprint4_proj.git
Project description
This project aims to provide you with additional practice on common software engineering tasks. These tasks will augment and complement your data skills, and make you a more attractive job candidate to potential employers. 

You will be asked to develop and deploy a web application to a cloud service so that it is accessible to the public.

In this project, we will provide you with the dataset on car sales advertisements, which you’ve already worked with in the past.

The project is split into several steps that replicate the process described in our blog post here, but we will be using the Render platform instead of Heroku.

Demo of the web application you’ll build in this project: https://www.youtube.com/watch?v=bna15Zj6jUI&ab_channel=TripleTen

Step 1. Getting started
Create an account on github.com.
Create a new git repository with a README.md file and a .gitignore file (choose a Python template).
Make a new Python environment. Install the following packages: pandas, streamlit, plotly.express, and altair. Feel free to install more packages if you want to implement additional features in your web app.
Create an account on render.com and link it to your GitHub account.
Use git to clone your new project’s git repository to your local machine. From now on, you’ll be working on the project on your local machine, and then committing and pushing the changes back to the GitHub repository.
Install VS Code and load the project into VS Code (by opening the project directory with VS Code).
If you have not used VS Code before, check out the extra lessons on Working in VS Code and Source Control in VS Code.
In VS Code, set the Python interpreter to the one used by the virtual environment. Make sure you have the necessary packages installed.
Step 2. Download the data file
Download the car advertisement dataset (vehicles_us.csv) or find your own dataset in a CSV format.
Place the dataset in the root directory of the project.
Note: Check out this video for project ideas and inspiration.

Step 3. Exploratory Data Analysis
Create an EDA.ipynb Jupyter notebook in VS Code.
Save the notebook in the notebooks directory of your project.
Perform some basic exploratory analysis of the dataset in the notebook.
Create a couple of histograms and scatterplots using plotly-express library.
Note: 

if you are using the car advertisement dataset, it won’t be sufficient to simply recreate the plots described in the blog post to complete the project. You’ll have to get creative and come up with your own plots and histograms.
it’s often very convenient to experiment with data visualizations in Jupyter, and then copy-paste code into a web application file later
Step 4. Develop the web application dashboard
Note that this is not the same development flow that we used in the lesson on rendering.

Create an app.py file in the root of the project’s directory. The following steps (2-4) will require you to write code into this app.py file.
In the app.py file, import streamlit, pandas, and plotly.express.
Read the dataset’s CSV file into a Pandas DataFrame.
From the Jupyter notebook, create and/or copy:
at least one st.header with text
at least one Plotly Express histogram using st.write or st.plotly_chart
at least one Plotly Express scatter plot using st.write or st.plotly_chart
at least one checkbox using st.checkbox that changes the behavior of any of the above components
Don’t forget to update the README file when you are done. It should contain a basic description of the project, explaining that this is a tool to simulate random events, and the methods and libraries used to implement it. It should also contain instructions on how other people could launch your project on their local machine if they wanted to.
IMPORTANT: Your project will only build on the online service if all project files are present in your GitHub repository. Therefore, you must commit and push each new file change to your repository as soon as you’ve completed it.
Notes: 

As you add new Streamlit components to develop your application, you can run the streamlit run app.py command from the terminal to see what the result will look like. Do this on your local machine (preferably from a system terminal) to test that everything works before committing and pushing your changes to GitHub.
Upon reaching certain application development milestones (e.g., adding a working component and having the application run without errors), it’s good practice to commit and push your work to a remote repository on GitHub. Don’t forget to write a meaningful commit message!
In this case, Render will fail to build your project unless all project files are pushed.

Step 5. Deploy the application to Render
1. To make Streamlit compatible with Render, we need to create two new files: requirements.txt and .streamlit/config.toml.
First, we need to create the requirements.txt file. Create this new file in your project folder’s root level. Then, add your project’s required packages. It should look something like this (although you can include other packages):
pandas==2.0.3
scipy==1.11.1
streamlit==1.25.0
altair==5.0.1
plotly==5.15.0
Second, we need to add the configuration file to your git repository. Create the .streamlit directory, then add the config.toml file there (this can all be done with the right-click menu in the left-hand tab of VS Code).

Add the following content to the .streamlit/config.toml file:

[server]
headless = true
port = 10000

[browser]
serverAddress = "0.0.0.0"
serverPort = 10000
This configuration file will tell Render where to look in order to listen to your Streamlit app when hosting it on its servers.

2. Open your account on render.com and create a new web service

3. Create a new web service linked to your GitHub repository

4. Configure the new Render web service. To your Build Command, add
pip install streamlit & pip install -r requirements.txt
To your Start Command, add: streamlit run app.py

5. Deploy to Render, wait for the build to succeed

6. Verify that your application is accessible at the following URL: https://<APP_NAME>.onrender.com/
Note: it can take several minutes after a successful deployment for the app to be available online on a free tier. Also note that apps go “asleep” after being inactive for a few minutes. If so, just load and refresh your app a few times for it to get awoken.

------------------------------------------------------------------
------------------------------------------------------------------

Sprint 8: Supervised Learning Machine Learning
Project description
Beta Bank customers are leaving: little by little, chipping away every month. The bankers figured out it’s cheaper to save the existing customers rather than to attract new ones.

We need to predict whether a customer will leave the bank soon. You have the data on clients’ past behavior and termination of contracts with the bank.

Build a model with the maximum possible F1 score. To pass the project, you need an F1 score of at least 0.59. Check the F1 for the test set.

Additionally, measure the AUC-ROC metric and compare it with the F1.

Project instructions
Download and prepare the data. Explain the procedure.
Examine the balance of classes. Train the model without taking into account the imbalance. Briefly describe your findings.
Improve the quality of the model. Make sure you use at least two approaches to fixing class imbalance. Use the training set to pick the best parameters. Train different models on training and validation sets. Find the best one. Briefly describe your findings.
Perform the final testing.
Data description
The data can be found in /datasets/Churn.csv file. Download the dataset.

Features

RowNumber — data string index
CustomerId — unique customer identifier
Surname — surname
CreditScore — credit score
Geography — country of residence
Gender — gender
Age — age
Tenure — period of maturation for a customer’s fixed deposit (years)
Balance — account balance
NumOfProducts — number of banking products used by the customer
HasCrCard — customer has a credit card
IsActiveMember — customer’s activeness
EstimatedSalary — estimated salary
Target

Exited — сustomer has left

------------------------------------------------------------------
------------------------------------------------------------------

Sprint 9: Machine Learning in Business
Project description
You work for the OilyGiant mining company. Your task is to find the best place for a new well.

Steps to choose the location:

Collect the oil well parameters in the selected region: oil quality and volume of reserves;
Build a model for predicting the volume of reserves in the new wells;
Pick the oil wells with the highest estimated values;
Pick the region with the highest total profit for the selected oil wells.
You have data on oil samples from three regions. Parameters of each oil well in the region are already known. Build a model that will help to pick the region with the highest profit margin. Analyze potential profit and risks using the Bootstrapping technique.

Project instructions
Download and prepare the data. Explain the procedure.
Train and test the model for each region:

 2.1. Split the data into a training set and validation set at a ratio of 75:25.

 2.2. Train the model and make predictions for the validation set.

 2.3. Save the predictions and correct answers for the validation set.

 2.4. Print the average volume of predicted reserves and model RMSE.

 2.5. Analyze the results.

Prepare for profit calculation:

 3.1. Store all key values for calculations in separate variables.

 3.2. Calculate the volume of reserves sufficient for developing a new well without losses. Compare the obtained value with the average volume of reserves in each region.

 3.3. Provide the findings about the preparation for profit calculation step.

Write a function to calculate profit from a set of selected oil wells and model predictions:

 4.1. Pick the wells with the highest values of predictions. 

 4.2. Summarize the target volume of reserves in accordance with these predictions

 4.3. Provide findings: suggest a region for oil wells' development and justify the choice. Calculate the profit for the obtained volume of reserves.

Calculate risks and profit for each region:

     5.1. Use the bootstrapping technique with 1000 samples to find the distribution of profit.

     5.2. Find average profit, 95% confidence interval and risk of losses. Loss is negative profit, calculate it as a probability and then express as a percentage.

     5.3. Provide findings: suggest a region for development of oil wells and justify the choice.

Data description
Geological exploration data for the three regions are stored in files:

geo_data_0.csv. download dataset
geo_data_1.csv. download dataset
geo_data_2.csv. download dataset
id — unique oil well identifier
f0, f1, f2 — three features of points (their specific meaning is unimportant, but the features themselves are significant)
product — volume of reserves in the oil well (thousand barrels).
Conditions:

Only linear regression is suitable for model training (the rest are not sufficiently predictable).
When exploring the region, a study of 500 points is carried with picking the best 200 points for the profit calculation.
The budget for development of 200 oil wells is 100 USD million.
One barrel of raw materials brings 4.5 USD of revenue The revenue from one unit of product is 4,500 dollars (volume of reserves is in thousand barrels).
After the risk evaluation, keep only the regions with the risk of losses lower than 2.5%. From the ones that fit the criteria, the region with the highest average profit should be selected.
The data is synthetic: contract details and well characteristics are not disclosed.

------------------------------------------------------------------
------------------------------------------------------------------

Sprint 13: Time Series
Project description
Sweet Lift Taxi company has collected historical data on taxi orders at airports. To attract more drivers during peak hours, we need to predict the amount of taxi orders for the next hour. Build a model for such a prediction.

The RMSE metric on the test set should not be more than 48.

Project instructions
1. Download the data and resample it by one hour.
2. Analyze the data.
3. Train different models with different hyperparameters. The test sample should be 10% of the initial dataset.
4. Test the data using the test sample and provide a conclusion.

Data description
The data is stored in the /datasets/taxi.csv file. Download the dataset. 

The number of orders is in the num_orders column.

------------------------------------------------------------------
------------------------------------------------------------------

Sprint 17: Final Project
Telecom Project
Intro: The telecom operator Interconnect would like to forecast the churn of their clients.
Business Problem Statement: The company wants to forecast which users are planning to leave

About the Dataset
The data consists of files obtained from different sources:
contract.csv - contract info
personal.csv - the client's personal data
internet.csv - info about Internet services
phone.csv - info about telephone services

Business Value: To ensure loyalty, those who are going to leave, will be offered with promotional codes and special plan options.
Tasks
To complete the final sprint successfully, you'll need a score of five story points (SP). These are conventional units for measuring the task's difficulty. You'll get:

4 to 6 SP for the main project
1 SP for the additional assignment
You're going to build a prototype of a machine learning model following these instructions:

Make a work plan. When you first see the task, you'll notice that it's incomplete and contains unnecessary information. Perform exploratory data analysis to figure out which questions you need to ask.
Investigate the task. Ask your team leader any questions you may have.
Develop a model. Submit your code to the project reviewer.
Prepare a report. Send the report to your team leader so that they can make sure you've completed the tasks correctly.
The final score depends on the quality of your model.

The additional assignment is based on the same data as the main project. 

At the end of the sprint, your code will be reviewed by the team leader.


Final Project: Solution Report
Make a report at the end of the Jupyter Notebook with the solution. The team leader will check it. The code will be reviewed by the team leader only if there are some points of doubt.

In the report, please answer the following questions:

What steps of the plan were performed and what steps were skipped (explain why)?
What difficulties did you encounter and how did you manage to solve them?
What were some of the key steps to solving the task?
What is your final model and what quality score does it have?
