# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 19:11:48 2020

@author: sjelic
"""



import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import torch
from torch import nn
import matplotlib.pyplot as plt
import functools as fnc
#from scikit-learn import shuffle

train = 0.8

HIRST_DATA_PATH = "../Libraries/HIRST_2019021601_2019101706.xlsx"

hirst = pd.read_excel(HIRST_DATA_PATH)
hirst['Unnamed: 0'] = list(map(lambda x: x[:-12],list(hirst['Unnamed: 0'])))





types = hirst.columns[1:]

totals = {}
for t in types:
    #print("Pollen type: " + t)
    #print("Total count")
    conc = np.array(list(hirst[t])).astype(float)
    totals[t] = conc.sum()
    #print(conc.sum())



sort_orders = sorted(totals.items(), key=lambda x: x[1], reverse=False)

newDict = dict(filter(lambda elem: elem[1] >= 10 , totals.items()))

#zero_hours = hirst[hirst["AMBR"] == 0][['Unnamed: 0', "AMBR"]]
#nonzero_hours = hirst[hirst["AMBR"] > 0][['Unnamed: 0', "AMBR"]]


#zero_hours = zero_hours.sort_values(by = "Unnamed: 0")
#nonzero_hours = nonzero_hours.sort_values(by = ["AMBR", "Unnamed: 0"], ascending=[False, True])

dir_path = "../../data/novi_sad_2019_"

def numpar_per_hour(dir_path, hirst_data_path):
    hirst = pd.read_excel(hirst_data_path)
    #print(hirst)
    
    #totals = []
    lista = list(hirst["Unnamed: 0"])
    #print(lista)
    #cnt = 0;
    dt = [];
    calibs = ["2019-02-18 16", "2019-02-25 12", "2019-03-11 12", "2019-03-15 12", "2019-03-15 13", 
          "2019-03-19 07", "2019-03-19 08", "2019-03-25 08","2019-03-25 09", "2019-03-26 08", 
          "2019-03-29 08", "2019-04-04 10", "2019-04-04 11", "2019-04-09 07", "2019-04-15 07", 
          "2019-04-18 16", "2019-04-22 08", "2019-04-22 14", "2019-04-29 10", "2019-05-03 08", 
          "2019-05-07 16", "2019-05-10 16", "2019-05-24 13", "2019-06-03 11", "2019-06-13 15", 
          '2019-05-15 03', '2019-05-15 04', '2019-08-03 03', '2019-09-23 04', '2019-09-24 21', 
          '2019-09-27 13', '2019-10-03 02', '2019-10-03 06', '2019-10-06 00', '2019-10-07 06', 
          '2019-10-11 07', '2019-10-11 08', '2019-10-12 02', '2019-10-13 03']
    for x in lista:
        if (x[:-12] in calibs):
            #print("Calibration hour. SKIPPED.")
            continue
        fpath = os.path.join(dir_path,x[:-12] + '.pkl')
        #print(fpath)
        if os.path.exists(fpath):
             with open(fpath, 'rb') as fp:
                 data = pickle.load(fp)
                 #print([x[:-12], len(data[0])])
                 dt.append([x[:-12], len(data[0])])
                 #df.loc[cnt] = [x[:-12], len(data[0])]
                
                 #print(x, tt)
                 #cnt += 1
                 #totals.append(tt)
    #print(dt)
    df = pd.DataFrame(dt, columns = ['Hour', 'Total'])
    df.to_excel('particle_counts_hour.xlsx')
    return df




def divide_set_in_season(dataset, num_cat, hirst_data_path):
    df = numpar_per_hour(dir_path,hirst_data_path)
    #print(df)
    #num_cat = 4
    dataset["Category"] = pd.qcut(dataset["AMBR"], num_cat, labels=False, duplicates='drop', retbins=True)[0]
    #print(dataset["Category"].sum())
    train_set = []
    valid_set = []
    num_of_train = 0
    num_of_valid = 0
    for i in range(num_cat):
        hour_cat = dataset[dataset["Category"] == i]
        #print(df["Hour"])
        hour_cat = df[df["Hour"].isin(list(hour_cat["Unnamed: 0"]))]
        #print(hour_cat)
        train_cat = hour_cat.sample(frac=0.8)
        valid_cat = hour_cat.drop(train_cat.index)
        #print(train_set)
        train_set += list(train_cat["Hour"])
        valid_set += list(valid_cat["Hour"])
        num_of_train += train_cat["Total"].sum()
        num_of_valid += valid_cat["Total"].sum()
        # napraviti sample izvan sezone sa toliko čestica
    #print(train_set)
    return train_set, valid_set, num_of_train, num_of_valid


def divide_set_out_season(dataset, num_cat):
    #num_cat = 4
    dataset["Category"] = pd.qcut(dataset["Total"], num_cat, duplicates='drop', labels=False, retbins=True)[0]
    #print(dataset["Category"].sum())
    train_set = []
    valid_set = []
    num_of_train = 0
    num_of_valid = 0
    for i in range(num_cat):
        
        hour_cat = dataset[dataset["Category"] == i]
        #print(hour_cat)
        #hour_cat = df[df["Hour"].isin(list(hour_cat["Unnamed: 0"]))]
        #print(hour_cat)
        train_cat = hour_cat.sample(frac=0.8)
        valid_cat = hour_cat.drop(train_cat.index)
        train_set += list(train_cat["Hour"])
        valid_set += list(valid_cat["Hour"])
        num_of_train += train_cat["Total"].sum()
        num_of_valid += valid_cat["Total"].sum()
        # napraviti sample izvan sezone sa toliko čestica
    #print(train_set)
    return train_set, valid_set, num_of_train, num_of_valid

    


def split_train_test(dir_path, hirst_data_path):
    hirst = pd.read_excel(hirst_data_path)
    hirst['Unnamed: 0'] = list(map(lambda x: x[:-12],list(hirst['Unnamed: 0'])))
    zero_hours = hirst[hirst["AMBR"] == 0][['Unnamed: 0', "AMBR"]]
    nonzero_hours = hirst[hirst["AMBR"] > 0][['Unnamed: 0', "AMBR"]] 

    df = numpar_per_hour(dir_path, hirst_data_path)
    zero_totals = df[df["Hour"].isin(list(zero_hours["Unnamed: 0"]))]
    
    train_set, valid_set, num_of_train, num_of_valid = divide_set_in_season(nonzero_hours, 4, hirst_data_path)
    train_set_outs, valid_set_outs, num_of_train_outs, num_of_valid_outs = divide_set_out_season(zero_totals, 4)

    
    #zero_totals = zero_totals.sort_values(by = ["Total", "Hour"], ascending=[False, True])

    
    train_set += train_set_outs
    valid_set += valid_set_outs
    
    train_set = sorted(train_set)
    valid_set = sorted(valid_set)
    #print(len(valid_set))   
    return train_set, valid_set


#train_set, valid_set = split_train_test("../data/novi_sad_2019_", "./Libraries/HIRST_2019021601_2019101706.xlsx")

#
