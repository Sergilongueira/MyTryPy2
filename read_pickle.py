import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt, find_peaks


def read_pickle(exp_dir, pickle_name):
    pickle_path = os.path.join(exp_dir, "RawData", pickle_name)
    dfDaq = pd.read_pickle(pickle_path)
    
    j = dfDaq['LINMOT_ENABLE'].eq(1).idxmax()
    dfDaq['Signal'] -= dfDaq['Signal'][:j].mean()
    
    mask = dfDaq['LINMOT_UP_DOWN'] == 1
    peaks, props = find_peaks(dfDaq['Signal'][mask], prominence=0.5, width=5)
    left_ips = props["left_ips"].astype(int)
    right_ips = props["right_ips"].astype(int)
    
    mask_without_peaks = mask.copy()
    for l, r in zip(left_ips, right_ips):
        mask_without_peaks[l:r+1] = False
    
    dfDaq.loc[mask, 'Signal'] -= dfDaq.loc[mask_without_peaks, 'Signal'].mean()
    
    dfDaq['Signal'] = medfilt(dfDaq['Signal'], kernel_size=7)
    
    dfDaq.to_pickle(pickle_path)
    
    mask = dfDaq['Time (s)'] < 10  # Plot only 10 first seconds
    plt.plot(dfDaq['Time (s)'][mask], dfDaq['Signal'][mask], color='yellow', label='Voltage')
    plt.plot(dfDaq['Time (s)'][mask], dfDaq['LINMOT_ENABLE'][mask], color='blue', label="LinMot Enable")
    plt.plot(dfDaq['Time (s)'][mask], dfDaq['LINMOT_UP_DOWN'][mask], color='green', label='LinMot Up/Down')
    plt.title('10 first seconds of experiment')
    plt.xlabel('Time (s)')
    plt.ylabel('Signals (V)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)


