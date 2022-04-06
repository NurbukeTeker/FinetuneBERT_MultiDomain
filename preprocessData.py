# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 08:14:44 2022

@author: nurbuketeker
"""
import pandas as pd

from getACIDData import getACIDData
from getBankingData import getBankingData
from getAtisData import getAtisData
from getCLINCData import getCLINCData
from getSnipsData import getSnipsData

atis_train, atis_test = getAtisData()
banking_train, banking_test = getBankingData()
clinc_train,clinc_test = getCLINCData()
acid_train , acid_test  = getACIDData()
snips_train, snips_test = getSnipsData()

df_train = pd.concat([atis_train, banking_train, clinc_train, acid_train,snips_train],ignore_index=True)
df_test = pd.concat([atis_test, banking_test, clinc_test, acid_test,snips_test],ignore_index=True)

# df_train.to_csv("domain_train_5Domain.csv", encoding='utf-8')
# df_test.to_csv("domain_test_5Domain.csv",  encoding='utf-8')


