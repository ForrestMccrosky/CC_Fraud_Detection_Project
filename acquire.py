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

import numpy as np
import pandas as pd
import seaborn as sns




############################# Function File for Acquiring Data ############################


def get_fraud_data():
    '''
    This function is designed to pull our credit card transaction data from the csv file into 
    a pandas dataframe then return the dataframe.
    
    It will also print out the shape of the dataframe after removing an unneccasry column
    '''
    df = pd.read_csv('fraudTrain.csv') ## <-- reading csv into pandas
    
    df = df.drop(columns = "Unnamed: 0") ## <-- dropping the extra index
    
    print("Shape of Dataframe (rows, columns):\n")
    print(df.shape)  ## look at shape
    
    return df

def summarize_df(df):
    '''
    this function is designed to look at our dataframe and print out a short summary of the
    dataframe.
    
    This will include things like:
    Transposed Numerical Statistics
    info on the columns and data types
    a target variable value counts
    value counts of hypothesized categorical columns
    '''
    
    print('Numerical Transposed Statistics:\n')
    print(df.describe().T)   ## <-- Looking at our transposed numerical statistics
    print('------------------------------------------------\n')
    
    
    print('Info on Columns and Datatypes:\n')
    print(df.info()) ## <-- looking at our columns and datatypes
    print('------------------------------------------------\n')
    
    print('Target Variable Values:\n')
    print(df.is_fraud.value_counts()) ## <-- looking at target variable value counts
    print('------------------------------------------------\n')
    
    ## creating a list of columns I want value counts for

    cols = ['state', 'gender', 'category', 'job']
    
    for col in cols:
        print(f'{col} Value Counts:\n')
        print(df[col].value_counts())
        print('------------------------------------------------\n')