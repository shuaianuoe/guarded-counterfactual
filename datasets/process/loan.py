# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:41:10 2023

@author: lenovo
"""

import numpy as np
import pandas as pd
from scipy import stats
from saving import save_to_file


def binary_map_categories(df):

    Dependents_map = {
        "0": "0",
        "1": "Non-0",
        "2": "Non-0",
        "3+": "Non-0",
    }
    df["Dependents"] = df["Dependents"].map(Dependents_map)

    Property_Area_map = {
        "Urban": "Urban",
        "Rural": "Non-Urban",
        "Semiurban": "Non-Urban",
    }
    df["Property_Area"] = df["Property_Area"].map(Property_Area_map)


    bins = [0, 2000, 4000, 6000, 8000, 10000, float('inf')]
    labels = [2000, 4000, 6000, 8000, 10000, 12000]
    

    df['ApplicantIncome'] = pd.cut(df['ApplicantIncome'], bins=bins, labels=labels, right=False).astype(int)
    

    bins_coapplicant = [1, 1000, 2000, 3000, 4000, 5000, 6000, float('inf')]
    labels_coapplicant = [1000, 2000, 3000, 4000, 5000, 6000, 6000]
    
    mask = df['CoapplicantIncome'] != 0  
    df.loc[mask, 'CoapplicantIncome'] = pd.cut(df.loc[mask, 'CoapplicantIncome'], bins=bins_coapplicant, labels=labels_coapplicant, right=False, ordered=False).astype(int)


    bins = [0, 10000, 20000, 30000, 40000, 50000, float('inf')]
    labels = [10000, 20000, 30000, 40000, 50000, 60000]
    
    df['LoanAmount'] = pd.cut(df['LoanAmount'], bins=bins, labels=labels, right=False).astype(int)
    
        
    df = df.dropna()

    return df



trainFile = "loan.csv"

# Read Data from csv
train_df = pd.read_csv( trainFile, index_col=False)

# drop rows with missing values
train_df = train_df.dropna(axis=0)


label_map = {'Y': 1, 'N': 0}
train_df["Loan_Status"] = train_df["Loan_Status"].map(label_map)
train_df['LoanAmount'] = train_df['LoanAmount'] *100

df = binary_map_categories(train_df.copy())

save_to_file(df, "loan")
    
# for column in train_df.columns:
#     print(column, train_df[column].unique())

# save_to_file(train_df, "loan")