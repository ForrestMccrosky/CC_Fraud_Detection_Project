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




############################# Function File for Exploring Data ############################

def variable_distributions(df):
    '''
    This function is designed to take in a list of column and output their variable distributions
    using histograms
    '''
    cols = ['category', 'gender', 'is_fraud', 'late_night', 'amt',
           'entertainment', 'home', 'shopping_net', 'misc_net', 'grocery_net',
           'grocery_pos', 'travel', 'high_fraud_cat', 'is_male', 'is_female',
           'Nevada', 'California', 'New_Mexico', 'Florida', 'Texas', 'Virginia',
           'Arizona', 'age_bin', 'amt_bin', '0-25_dollars',
           '25-40_dollars', '40-50_dollars', '50-100_dollars', '100-200_dollars',
           'high_dollars', '0-20_age', '20-40_age', '40-60_age', '60-96_age']
    
    for col in cols:
        plt.hist(df[col])
        plt.title(f'Distribution of {col}')
        plt.ylabel('Frequency')
        plt.xlabel(f'{col}: values')
        plt.show()
        

        
def observe_cross(x, y):
    '''
    This function is designed to create a crosstab of a categorical column and compare it to 
    our target is_fraud to see which categories have the most fradulent transactions
    '''
    
    ctab = pd.crosstab(x, y)
    ctab = ctab.sort_values(by = 1, ascending = False)
    
    return ctab.head(5)


def observe_chi(x, y):
    '''
    This function is designed to create a crosstab of a categorical column and compare it to our
    target variable is_fraud
    
    The function will then use this observed crosstab and use it to run a Chi Squared statistical 
    test on the variables testing for significance 
    '''
    
    ## creating our observed crosstab of the two categories
    observed = pd.crosstab(x, y)
    print('Comparing our Variables\n')
    print(observed)
    print('----------------------\n')
    
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print('Observed\n')
    print(observed.values)
    print('---\nExpected\n')
    print(expected)
    print('---\n')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    