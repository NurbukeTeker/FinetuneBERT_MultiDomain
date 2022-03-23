# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 08:18:46 2022

@author: nurbuketeker
"""
import pandas as pd

def getBankingData():
    df_banking_train = pd.read_csv("BankingData/train.csv")
    
    banking_train_texts = df_banking_train["text"].tolist()
    banking_train_labels = df_banking_train["category"].tolist()
    
    df_banking_test = pd.read_csv("BankingData/test.csv")
    
    banking_test_texts = df_banking_test["text"].tolist()
    banking_test_labels = df_banking_test["category"].tolist()
    
    columns = ["text", "intent"]    
    df_banking_train = pd.DataFrame(list(zip(banking_train_texts, banking_train_labels)), columns =columns)
    df_banking_test = pd.DataFrame(list(zip(banking_test_texts, banking_test_labels)), columns =columns)
    
    df_banking_train["intent"] = 2
    df_banking_test["intent"] = 2
    
    return df_banking_train ,df_banking_test
    
