# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 08:54:14 2022

@author: nurbuketeker
"""
from datasets import load_dataset
import pandas as pd

def getCLINCData():
    

    dataset_small_train = load_dataset("clinc_oos", "small",split="train")
    dataset_small_test = load_dataset("clinc_oos", "small",split="test")
    
    df_train = pd.DataFrame(dataset_small_train)
    df_test = pd.DataFrame(dataset_small_test)

    df_train["intent"] = 4
    df_test["intent"] = 4
    
    return df_train, df_test