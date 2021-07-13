# Individual Classification Project: Predictin Credit Card Fraud

## Project Description
 - The purpose of this notebook is to acquire, prep, explore a csv file downloaded from kaggle.com that contains Credit Card transactions and use classification modeling to predict whether or not the transactions made were fraud.
 - Project created using the data science pipeline (Acquisition, Preparation, Exploration, Analysis & Statistical Testing, and finally Modeling and Evaluation)

## Project Goals
 - Create a Final Jupyter Notebook that reads like a report and follows the data science pipeline
 - In the Jupyter Notebook Create a classification model that performs well in the recall metric (deemed most important metric) with a business sense in mind.
 - Create Function Files to help peers execute project reproduction
 - Draw valuable conclusion that define potential model applications and identify next steps and possible points of improvement

## Deliverables
 - Final Jupyter Notebook
 - Function Files for reproduction
 - Trello Board (Agenda Board)
 - Detailed Readme

## Executive Summary
The purpose of this notebook is to use classification modeling to predict our positive cases of our target variable (is_fraud) as accurately as possible while maintaining a low false negative and positive rate.
 - After visual exploration and statistical testing the features that were inputted into our models were
    - high_dollars (transactions that were deemed expensive)
    - late_night (transactions that occured in the hours of 11PM and 12AM
    - amt (the amount of the transaction)
    - high_fraud_cat (transactions that fell under the online shopping or grocery shopping category)

Our most successful model that was used on our Out-of-sample (test) dataframe was the Random Forest Model II which performed with the folling metrics:
 - Accuracy: 90.7182%
 - True Positive Rate: 90.740%
 - True Negative Rate: 90.718%
 - False Positive Rate: 9.282%
 - False Negative Rate: 9.260%

 ## Hypothesis
 - Most victims of fraudulent transactions will fall in high prone areas and we will look at states that fraud is most prevelant
 - Most fraudelent transactions will occur late at night when potential warning systems aren't read by the victims attentively
 - Most victims of fraud will be of higher age 

 ## Findings & Takeaways
 The Random Forest Model II peformed the best on the validate in metrics deemed most important (True Positive Rate & True Negative Rate)

 - Out-of-sample (Test Dataframe) Results:
    - Accuracy: 90.7182%
    - True Positive Rate: 90.740%
    - True Negative Rate: 90.718%
    - False Positive Rate: 9.282%
    - False Negative Rate: 9.260%
    - Overall this model is a success!:

Given the well rounded evaluation metrics produced by this model looking at the true positive/negative and false positive/negative percentages. This Random Forest model will be successful in helping millions of customers correctly identify and catch fradulent transaction, and at the same time it will be careful about sending out incorrect warning messages about transactions that were valid and not fraud.

The well roundedness of this model will ensure customer retention through accurate fraud prevention and not bother too many valid purchasing customers due to it's low false positive percentage and careful selectiveness.

# The Pipeline

## Planninng

With some domain knowledge I want to make as many categorical features as possible to really find some key features in predicting our target variable is_fraud. The idea is to build the best model possible in predicting true positive rates (catching fraud correctly).

## Acquire

The credit card transaction data is on kaggle.com and the first steps to acquire the data after building the project repository is to download the train csv file and put it into our repository.

After this step is achieved a function was made in the acquire.py file that reads the csv into a pandas dataframe.

There was no subsampling neccessary and the dataframe after reading from the csv is ready to prepare.

## Prepare
 - I took my original 22 column dataframe and duplicated the potential modeling features by adding 35 additional columns using mostly np.where statements on the state and category columns splitting them into likely fradulent categorical columns (States prone to fraud & categories of most fradulent transactions
 - I also split the gender column into a is_male and is_female column
 - Also using datetime (converting the transaction_time column) I was able to create columns for transaction: hour, weekday, month, and year
 - With the tran_time being date time I was also able to convert the customer dob (date of birth column) into datetime and use a Time delta to obtain the customer age as a column
### Other Features
 - I was also able to make bins for the amt (amount of transaction column) and customer_age column and turn those into respective categorical features as well.
### Overall

The new prepared dataframe has 0 nulls, plenty of features to work with, and an even amount of positive cases after performing the train, validate, and test split

## Explore

The goal of explore is to visualize data relationships and perform statistical testing to determine if the features the project plans on using have a significant relationship with the target variable.

### Visual Exploration:
 - Looking at the bar charts we can see that most of the transaction occur during the evening hours of the day
    - Most likely due to those being high traffic times for purchases due to most of the working population beeing off work at those times
 - Most of the fradulent transaction occur late in the evening around the hours of 11PM and 12 AM
 - Most of the fradulent transactions fall under the shopping_net (online shopping) category and grocery_pos category (grocery in store shopping)

### Statistical Exploration:

After performing Chi Squared tests on the following features:
 - shopping_pos vs is_fraud
 - high_dollars vs is_fraud
 - late_night vs is_fraud
 - high_fraud_cat vs is_fraud

Every Chi Squared test returned a p-value that was near zero and less than our alpha of 0.05. Therefore we can determine that the relationships between those best correlated features are significant to our target variable is_fraud, and we can fell confident using them in modeling.

## Modeling & Evaluation

The goal of the modeling and evalutaion component of the pipeline is to use the best features determined from explore to predict our target variable is_fraud using classification modeling.

### Features Used in Modeling
 - late_night
 - high_fraud_cat
 - amt
 - high_dollars

### Model Performance Train

| Model                     | Accuracy | True Positive Rate | True Negative Rate | False Positive Rate | False Negative Rate |
|---------------------------|----------|--------------------|--------------------|---------------------|---------------------|
| Decision Tree Classifier  | 87.4077% | 95.055%            | 79.760%            | 20.240%             | 4.945%              |
| Random Forest Model I     | 99.5658% | 99.627%            | 99.504%            | 0.496%              | 0.373%              |
| KNN Model                 | 97.6964% | 97.142%            | 98.251%            | 1.749%              | 2.858%              |
| Logistic Regression Model | 86.1075% | 85.630%            | 86.585%            | 13.415%             | 14.370%             |
| Random Forest Model II    | 91.3975% | 91.715%            | 91.080%            | 8.920%              | 8.285%              |

### Model Performance Validate

| Model                    | Accuracy | True Positive Rate | True Negative Rate | False Positive Rate | False Negative Rate |
|--------------------------|----------|--------------------|--------------------|---------------------|---------------------|
| Decision Tree Classifier | 79.9195% | 94.836%            | 79.833%            | 20.167%             | 5.164%              |
| KNN Model                | 97.7571% | 73.182%            | 97.900%            | 2.100%              | 26.818%             |
| Random Forest Model II   | 91.0855% | 90.394%            | 91.090%            | 8.910%              | 9.606%              |

### Out-of-sample Test on Random Forest Model II 

| Model                  | Accuracy | True Positive Rate | True Negative Rate | False Positive Rate | False Negative Rate |
|------------------------|----------|--------------------|--------------------|---------------------|---------------------|
| Random Forest Model II | 91.0328% | 90.473%            | 91.036%            | 8.964%              | 9.527%              |

#### This meets the project goals performing well in the recall category (true positive rate) while maintaining low false positive and negative rates.

## Data Dictionary

### Target Variable

| Column Name | Data Type | Value                                               |
|-------------|-----------|-----------------------------------------------------|
| is_fraud    | int64     | 1 for positive fraud case 0 for negative fraud case |

### All Columns After Prepare

| Column Name     | Data Type      | Value                                                                                                  |
|-----------------|----------------|--------------------------------------------------------------------------------------------------------|
| trans_time      | datetime64     | credit card purchase transaction time                                                                  |
| cc_num          | int64          | credit number of customer                                                                              |
| merchant        | object         | name of the purchase location                                                                          |
| category        | object         | category of the transaction ex: (online shopping, grocery, gas, etc.)                                  |
| amt             | float64        | amount of the transaction                                                                              |
| first           | object         | first name of the cardholder                                                                           |
| last            | object         | last name of the cardholder                                                                            |
| gender          | object         | gender of the card holder                                                                              |
| street          | object         | street location of the card holder                                                                     |
| city            | object         | city location of the card holder                                                                       |
| state           | object         | state location of the card holder                                                                      |
| zip             | int64          | postal code location of the card holder                                                                |
| lat             | float64        | latitude coordinate for the card holder                                                                |
| long            | float64        | longitude coordinate for the card holder                                                               |
| city_pop        | int64          | population of the city of the card holder                                                              |
| job             | object         | job title of the card holder                                                                           |
| dob             | datetime64     | date of birth of the card holder                                                                       |
| trans_num       | object         | transaction number for the cc purchase                                                                 |
| unix_time       | int64          | unix time of the purchase                                                                              |
| merch_lat       | float64        | latitude coordinate for the cc purchase location                                                       |
| merch_long      | float64        | longitude coordinate for the cc purchase location                                                      |
| weekday         | int64          | weekday value for the cc purchase                                                                      |
| trans_hour      | int64          | transaction hour value for the cc purchase                                                             |
| year            | int64          | transaction year value for the cc purchase                                                             |
| month           | int64          | transaction month value for the cc purchase                                                            |
| customer_age    | int64          | age of the card holder (trans_time - dob)                                                              |
| late_night      | int64          | categorical column 1 for cc purchases that fell in the hours of 11PM and 12AM 0 if not                 |
| entertainment   | int64          | categorical column 1 if purchase category is entertainment 0 if not                                    |
| home            | int64          | categorical column 1 if purchase category is home 0 if not                                             |
| shopping_net    | int64          | categorical column 1 if purchase category is online shopping 0 if not                                  |
| misc_net        | int64          | categorical column 1 if purchase category is misc 0 if not                                             |
| grocery_net     | int64          | categorical column 1 if purchase category is online groceries 0 if not                                 |
| grocery_pos     | int64          | categorical column 1 if purchase category is grocery in store 0 if not                                 |
| travel          | int64          | categorical column 1 if purchase category is travel 0 if not                                           |
| high_fraud_cat  | int64          | if purchase falls under online shopping or grocery pos value = 1, 0 if not                             |
| is_male         | int64          | 1 if customer is male 0 if not                                                                         |
| is_female       | int64          | 1 if customer is female 0 if not                                                                       |
| Nevada          | int64          | 1 if card holder is in Nevada and 0 if not                                                             |
| California      | int64          | 1 if card holder is in California and 0 if not                                                         |
| New_Mexico      | int64          | 1 if card holder is in New_Mexico and 0 if not                                                         |
| Florida         | int64          | 1 if card holder is in Florida 0 if not                                                                |
| Texas           | int64          | 1 if card holder is in Texas 0 if not                                                                  |
| Virginia        | int64          | 1 if card holder is in Virginia 0 if not                                                               |
| Arizona         | int64          | 1 if card holder is in Arizona 0 if not                                                                |
| age_bin         | category       | Bins the customer ages (0, 20, 40, 60, 96) represented with the values 1, 2, 3, 4                      |
| amt_bin         | category       | Bins the transaction amount (0, 25, 40, 50, 100, 200, 30000) represented with the values 1, 2, 3, 4, 5 |
| 0-25_dollars    | int64          | 1 if transaction amount is between 0 and 25 dollars 0 if not                                           |
| 25-40_dollars   | int64          | 1 if transaction amount is between 25 and 40 dollars 0 if not                                          |
| 40-50_dollars   | int64          | 1 if transaction amount is between 40 and 50 dollars 0 if not                                          |
| 50-100_dollars  | int64          | 1 if transaction amount is between 50 and 100 dollars 0 if not                                         |
| 100-200_dollars | int64          | 1 if transaction amount is between 100 and 300 dolalrs 0 if not                                        |
| high_dollars    | int64          | 1 if transaction exceeds 200 dollars 0 if not                                                          |
| 0-20_age        | in64           | 1 if customer age is between 0 and 20 0 if not                                                         |
| 20-40_age       | int64          | 1 if customer age is between 20 and 40 0 if not                                                        |
| 40-60_age       | int64          | 1 if customer age is between 40 and 60 0 if not                                                        |
| 60-96_age       | int64          | 1 if customer age is between 60 and 96 0 if not                                                        |



 ## Project Recreation
 - Download the archived folder from kaggle.com containing the train csv for the credit card transaction data
 - Use the functions in the .py files and follow the pipeline flow of the notebook
 - Feel free to adjust hyperparameters of the models and try out the results yourself!