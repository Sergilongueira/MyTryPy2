# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:47:59 2025

@author: slongueira
"""

import pandas as pd
import matplotlib.pyplot as plt

dfDaq = pd.read_pickle('DAQ-2508-F-3.pkl')

plt.plot(dfDaq['Time (s)'], dfDaq['Signal'], color='red')
# plt.plot(dfDaq['Time (s)'], dfDaq['LINMOT_ENABLE'], color='blue')
# plt.plot(dfDaq['Time (s)'], dfDaq['LINMOT_UP_DOWN'], color='green')
# - dfDaq['LINMOT_ENABLE'] - dfDaq['LINMOT_UP_DOWN']