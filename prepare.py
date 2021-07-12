import warnings
warnings.filterwarnings('ignore')

from math import sqrt
from scipy import stats
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns




############################# Function File for Preparing Data ############################

def prep_dates_cc_fraud(df):
    '''
    This function is designed to take in our dataframe rename the long transaction time column
    to tran_time.
    
    It will also convert the column into datetime and make categorical features for month,
    weekday, and transaction hour as well as a column for light at night purchases around 11 PM
    and 12 AM where fraud is most likely to occur
    
    It will also use a timedelta and subtract the transaction time from the customer date of
    birth column to generate a customer age column
    '''
    
    ## renaming column for easier coding
    df = df.rename(columns = {'trans_date_trans_time': 'trans_time'}) 
    
    ## turn the transaction time to datetime format
    df.trans_time = pd.to_datetime(df.trans_time) 
    
    df['weekday'] = df['trans_time'].dt.weekday ## getting weekday column
    print('Looking at our weekday Values')
    print(df['weekday'])
    print('----------------------------\n')
    
    df['trans_hour'] = df['trans_time'].dt.hour ## getting hour column
    print('Looking at our transaction hour Values')
    print(df['trans_hour'])
    print('----------------------------\n')
    
    df['year'] = df['trans_time'].dt.year ## getting year column
    print('Looking at our year Values')
    print(df['year'])
    print('----------------------------\n')
    
    df['month'] = df['trans_time'].dt.month ## getting month column
    print('Looking at our month Values')
    print(df['month'])
    print('----------------------------\n')
    
    ## Getting the Age of the Customer
    df['dob'] = pd.to_datetime(df['dob'])
    df['customer_age'] = np.round((df['trans_time'] - df['dob'])/np.timedelta64(1,'Y'))
    df['customer_age'] = df['customer_age'].astype(int)
    print('Looking at our customer ages')
    print(df['customer_age'])
    print('----------------------------\n')
    
    ## making a late_night feature where the transactions of fraud are most commmon
    df['late_night'] = np.where(((df.trans_hour == 22) | (df.trans_hour == 23)), 1, 0)
    print('Looking at out late_night Value Counts')
    print(df.late_night.value_counts())
    
    return df


def create_features(df):
    '''
    This function is designed to take our category of transaction column and split it up
    into multiple categorical features for moedling
    
    It will also make categorical column by combining grocery_pos and shopping_net into one
    boolean column of 1 or 0 because those categories have the highest amount of fraud
    transactions
    
    It also takes our transaction state column and spits those states with the most fraud 
    according to the internet into their own categorical columns
    
    It will also split up the gender column into is_male and is_female categorical columns as 
    well
    '''
    ## making boolean columns for hypothesized common fraud purchase categories

    df['entertainment'] = np.where(df.category == 'entertainment', 1, 0)
    df['home'] = np.where(df.category == 'home', 1, 0)
    df['shopping_net'] = np.where(df.category == 'shopping_net', 1, 0)
    df['misc_net'] = np.where(df.category == 'misc_net', 1, 0)
    df['grocery_net'] = np.where(df.category == 'grocery_net', 1, 0)
    df['grocery_pos'] = np.where(df.category == 'grocery_pos', 1, 0)
    df['travel'] = np.where(df.category == 'travel', 1, 0)
    
    ## making a high fraud category where fraud transactions is most likely based on crosstabs
    df['high_fraud_cat'] = np.where(((df.category == 'grocery_pos') | (df.category == 
                                                                       'shopping_net')), 1, 0)
    
    ## making boolean columns for other columns that may be valueable to split up
    
    df['is_male'] = np.where(df.gender == 'M', 1, 0)
    df['is_female'] = np.where(df.gender == 'F', 1, 0)
    
    ## States with highest amounts of CC Fraud (looked on internet)

    df['Nevada'] = np.where(df.state == 'NV', 1, 0)
    df['California'] = np.where(df.state == 'CA', 1, 0)
    df['New_Mexico'] = np.where(df.state == 'NM', 1, 0)
    df['Florida'] = np.where(df.state == 'FL', 1, 0)
    df['Texas'] = np.where(df.state == 'TX', 1, 0)
    df['Virginia'] = np.where(df.state == 'VA', 1, 0)
    df['Arizona'] = np.where(df.state == 'AZ', 1, 0)
    
    return df

def make_bins_and_feats(df):
    '''
    This function is designed to make different bins using pd.cut andthe amount of transaction 
    column and the age of the customer column
    
    This function will then take those bins and use them to create respective categorical 
    features
    '''
    
    print('Looking at our maximum age and minimum age to make some age bins')
    print(df.customer_age.max(), df.customer_age.min())
    print('----------------------------\n')
    
    ## creating the age bins with integer labels for easier feature engineering
    df['age_bin'] = pd.cut(df.customer_age, 
                           bins = [0, 20, 40, 60, 96],
                           labels = [1, 2, 3, 4])
    print('Looking at our Age Bins')
    print(df['age_bin'].value_counts()) ## <-- quality assurance check of column values
    print('----------------------------\n')
    
    print('Looking at our maximum amount and minimum amount to make some transaction amount bins')
    print(df.amt.max(), df.amt.min())
    print('----------------------------\n')
    
    print('Visualizing amounts to help my decision with binning')
    sns.boxplot(data = df, x = 'amt')
    plt.show()
    print('----------------------------\n')
    
    ## creating the amount bins with integer labels for easier feature engineering
    df['amt_bin'] = pd.cut(df.amt, 
                           bins = [0, 25, 40, 50, 100, 200, 30000],
                           labels = [1, 2, 3, 4, 5, 6])
    print('Looking at our Amount Bins')
    print(df['amt_bin'].value_counts()) ## <-- quality assurance check of my amt_bin column
    print('----------------------------\n')
    
    ## Making categorical Columns for the amount bin that I created

    df['0-25_dollars'] = np.where(df.amt_bin == 1, 1, 0)
    df['25-40_dollars'] = np.where(df.amt_bin == 2, 1, 0)
    df['40-50_dollars'] = np.where(df.amt_bin == 3, 1, 0)
    df['50-100_dollars'] = np.where(df.amt_bin == 4, 1, 0)
    df['100-200_dollars'] = np.where(df.amt_bin == 5, 1, 0)
    df['high_dollars'] = np.where(df.amt_bin == 6, 1, 0)
    
    ## Making categorical Columns for the age bin that I created
    
    df['0-20_age'] = np.where(df.age_bin == 1, 1, 0)
    df['20-40_age'] = np.where(df.age_bin == 2, 1, 0)
    df['40-60_age'] = np.where(df.age_bin == 3, 1, 0)
    df['60-96_age'] = np.where(df.age_bin == 4, 1, 0)
    
    return df

def split_data(df):
    '''
    This function is designed to split out data for modeling into a train, validate, and test 
    dataframe stratifying on our target variable is_fraud
    
    It will also perform quality assurance checks on each dataframe to make sure the target 
    variable was correctcly stratified into each dataframe.
    '''
    
    ## splitting the data stratifying for out target variable is_fraud
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123,
                                        stratify = df.is_fraud)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123,
                                   stratify= train_validate.is_fraud)
    
    print('Making Sure Our Shapes Look Good')
    print(f'Train: {train.shape}, Validate: {validate.shape}, Test: {test.shape}')
    print('----------------------------\n')
    
    print('Making Sure We Have Positive Cases In Each Split\n')
    
    print('Train Target Value Counts:')
    print(train.is_fraud.value_counts())
    print('----------------------------\n')
    
    print('Validate Target Value Counts:')
    print(validate.is_fraud.value_counts())
    print('----------------------------\n')
    
    print('Test Target Value Counts:')
    print(test.is_fraud.value_counts())
    print('----------------------------\n')
    
    return train, validate, test