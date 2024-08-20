# Sprint 17: Final Project

## Telecom Project
Intro: The telecom operator Interconnect would like to forecast the churn of their clients.
Business Problem Statement: The company wants to forecast which users are planning to leave

## About the Dataset
The data consists of files obtained from different sources:
contract.csv - contract info<br/>
personal.csv - the client's personal data<br/>
internet.csv - info about Internet services<br/>
phone.csv - info about telephone services

**Business Value:** To ensure loyalty, those who are going to leave, will be offered with promotional codes and special plan options.

## Tasks
To complete the final sprint successfully, you'll need a score of five story points (SP). These are conventional units for measuring the task's difficulty. You'll get:

4 to 6 SP for the main project<br/>
1 SP for the additional assignment<br/>
You're going to build a prototype of a machine learning model following these instructions:

- Make a work plan. When you first see the task, you'll notice that it's incomplete and contains unnecessary information. Perform exploratory data analysis to figure out which questions you need to ask.
- Investigate the task. Ask your team leader any questions you may have.
- Develop a model. Submit your code to the project reviewer.
- Prepare a report. Send the report to your team leader so that they can make sure you've completed the tasks correctly.

The final score depends on the quality of your model.

The additional assignment is based on the same data as the main project. 

At the end of the sprint, your code will be reviewed by the team leader.


## Final Project: Solution Report
Make a report at the end of the Jupyter Notebook with the solution. The team leader will check it. The code will be reviewed by the team leader only if there are some points of doubt.

In the report, please answer the following questions:

What steps of the plan were performed and what steps were skipped (explain why)?<br/>
What difficulties did you encounter and how did you manage to solve them?<br/>
What were some of the key steps to solving the task?<br/>
What is your final model and what quality score does it have?

## Conclusions
From the data provided by Interconnect, we saw the following:

- The churn customers tend to leave in the first half year
- There was a high churn rate for those with a higher monthly total
- The data should use one-hot encoding and MaxAbsScalar
- When predicting the customers that will churn, the four optimized models (XGBClassifier, LogisticRegression, RandomForestClassifier, and DecisionTreeClassifier) work well for predicting with at least a AUC-ROC score of 0.82
- Although XGBClassifier had the highest score, the other three models came close with less than a 0.02 difference
- When looking at the categorical columns, churned customers tended to pay month-to-month, pay with electronic check, and did not use the internet options like online backup.

With this information, Interconnect can forecast who is most likely to churn before they churn and offer some kind of promotions or incentives. For example, since we saw a large amount of customers stay for less than half a year, the company can offer a multiple month package for a standard rate, regardless of what features they sign up for. This can help curb some of the customers who have a high monthly total to stay with the company since the per month cost would be set.
