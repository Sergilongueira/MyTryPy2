# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:47:59 2025

@author: slongueira
"""

import pandas as pd

dfDaq = pd.read_pickle('DAQ_01.pkl')

dfMot = pd.read_csv('Motor_01.csv', header=0, index_col=False, delimiter=',', decimal='.')