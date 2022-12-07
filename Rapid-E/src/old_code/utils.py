# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 03:24:11 2020

@author: Slobodan
"""
import os
import pandas as pd
import logging
import datetime
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
import json
import itertools
import torch
import sys
#from clustering.equal_groups import EqualGroupsKMeans


def open_excel_file_in_pandas(file_path):
    try:
        df = pd.read_excel(file_path, header=0)
    except FileNotFoundError:
        raise RuntimeError(f'File {file_path} does not exist. Please, provide correct filepath.')
    return df

def save_pandas_to_excel_file(df,file_path):
    df.to_excel(file_path, index=False)

def select_pollen(dfH, dfP):
    pollen_codes = list(dfP['CODE'])
    pollen_codes_selected = []
    pollen_codes_skipped = []
    for code in pollen_codes:
        if code in list(dfH.columns):
            pollen_codes_selected.append(code)
        else:
            pollen_codes_skipped.append(code)
    
    if (len(pollen_codes_skipped) > 0):
        message = 'Following pollen types are not found in Hirst data collection: \n'
        #logging.warning()
        for code in pollen_codes_skipped:
            message += '\t' + code +' - ' + dfP[dfP.CODE == code].LATIN.to_string(index=False) +'\n'
        message += 'These pollen types will be skipped.'
        logging.warning(message)
        
    dfH = dfH[['HOUR'] + pollen_codes_selected]
    dfH['HOUR'] = pd.to_datetime(dfH['HOUR'])
    return dfH

def exclude_calibration_hours(dfH, dfC):
    dfC['Time'] = pd.to_datetime(dfC['Time'])
    dfH = dfH[~dfH.HOUR.isin(dfC.Time)]
    return dfH

def read_data_dir(dir_path):
    fnames = os.listdir(dir_path)
    dt = [];
    for fname in fnames:
        fp = os.path.join(dir_path,fname)
        if os.path.exists(fp):
            with open(fp, 'rb') as file:
                data = pickle.load(file)
                #print(fname[:-4])
                dt.append([datetime.datetime.strptime(fname[:-4] + ':00:00', '%Y-%m-%d %H:%M:%S'),fname, len(data[0])])
    df = pd.DataFrame(dt, columns = ['HOUR','FILENAME', 'TOTAL'])
    return df

def join_hirst_rapid_data(dfH,dfR):
    pollen_codes = dfH.columns[1:]
    dfH = dfH.set_index('HOUR')
    dfR = dfR.set_index('HOUR')
    df = pd.merge(dfH,dfR, how='inner', left_index=True, right_index=True)
    df['HOUR'] = df.index
    df.reset_index(drop=True, inplace=True)
    df = df[['HOUR', 'FILENAME', 'TOTAL'] + list(pollen_codes)]
    return df

def set_time_resolution(df, res='hour'):
    if res == 'hour':
        return df
    elif res == 'day':
        colnames = ['TOTAL'] + list(df.columns[3:])
        df['DATE'] = list(map(lambda x: x.date(), df['HOUR']))
        
        gr = df.groupby('DATE').filter(lambda x: len(x['HOUR']) == 24).groupby('DATE')
        df = gr[colnames].sum()
        df['DATE'] = df.index
        df.reset_index(drop=True, inplace=True)
        df = df[['DATE'] + colnames]
        return df
    else:
        raise RuntimeError(f'{res} is not supported aggregation method.')
        
def count_elements(clusters,num_cl):
    sizes = [0]*num_cl
    for x in clusters:
        sizes[x] += 1
    return sizes

def num_of_clusters(silhouette_avgs, min_sizes):
    for i in range(len(silhouette_avgs)):
        if (min_sizes[i] < 50):
            silhouette_avgs[i] = 0
    return np.argmax(np.array(silhouette_avgs))

    

def cluster_analysis_of_dataset(df, pollen_types, num_clust = 30):
    cols = ['TOTAL'] + pollen_types
    X = np.array(df[cols])
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    silhouette_avgs = []
    min_sizes = []
    labels = []
    for nc in range(2,num_clust+1,1):
        kmeans = KMeans(n_clusters=nc)
        clusters = kmeans.fit(X)
        silhouette_avgs.append(silhouette_score(X, clusters.labels_))
        sizes = count_elements(clusters.labels_,nc)
        min_sizes.append(np.argmin(np.array(sizes)))
        labels.append(clusters.labels_)
    num_cl = num_of_clusters(silhouette_avgs, min_sizes)
    #print(silhouette_avgs)
    #print(min_sizes)
    print(f"Optimal number of clusters: {num_cl+2}")
    #kmeans = KMeans(n_clusters=num_cl)
    #clusters = kmeans.fit(X)
    return labels[num_cl]


def clusters_from_seasons(df, pollen_types, df_pollen_info):
    
    inc = np.zeros((len(pollen_types), 12))
    for i,tp in enumerate(pollen_types):
        start = list(df_pollen_info[df_pollen_info.CODE == tp]['START'])[0]
        end = list(df_pollen_info[df_pollen_info.CODE == tp]['END'])[0]
        inc[i,(start-1):end] = 1

    #prev = inc[:,0]
    clusters = [[0]]
    
    for i in range(1,12):
        found = False
        for cluster in clusters:
            if np.array_equal(inc[:,i], inc[:,cluster[0]]):
                cluster.append(i)
                found = True
                break 
        if not found:
           clusters.append([i])
           
    labels = []
    months = list(map(lambda x: x-1, list(df['MONTH'])))
    for x in months:
        for j,cl in enumerate(clusters):
            if x in cl:
                labels.append(j)
                break
    return labels
        
          
        

# def cluster_analysis_of_dataset_equal_clustersizes(df, pollen_types, num_clust = 3):
#     cols = pollen_types
#     X = np.array(df[cols])
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)
#     #silhouette_avgs = []
#     #min_sizes = []
#     #labels = []
#     #for nc in range(2,num_clust+1,1):
#     kmeans = EqualGroupsKMeans(n_clusters=num_clust)
#     clusters = kmeans.fit(X)
#     #silhouette_avgs.append(silhouette_score(X, clusters.labels_))
#     #sizes = count_elements(clusters.labels_,nc)
#     #min_sizes.append(np.argmin(np.array(sizes)))
#     #labels.append(clusters.labels_)
#     #num_cl = num_of_clusters(silhouette_avgs, min_sizes)
#     #print(silhouette_avgs)
#     #print(min_sizes)
#     #print(f"Optimal number of clusters: {num_cl+2}")
#     #kmeans = KMeans(n_clusters=num_cl)
#     #clusters = kmeans.fit(X)
#     return clusters.labels_

def assign_clusters(df, pollen_types, clusters):
    df['CLUSTER'] = clusters
    df = df[['MONTH', 'HOUR', 'FILENAME', 'CLUSTER'] + pollen_types]
    return df
        
    
def train_test_split(groups, num_splits = 10):
    X = np.array(range(len(groups)))
    y = groups
    split = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    return split.split(X, y)



def load_dataset(dir_path, hirst_data_path, calib_info_path, pollen_info_path, pollen_types, time_res = 'hour'):
    dfH = open_excel_file_in_pandas(hirst_data_path)
    dfC = open_excel_file_in_pandas(calib_info_path)
    dfP = open_excel_file_in_pandas(pollen_info_path)
    logging.info('Pollen selection started.')
    dfH = select_pollen(dfH, dfP)
    logging.info('Pollen selection finished.')
    logging.info('Removing calibration hours started.')
    dfH = exclude_calibration_hours(dfH, dfC)
    logging.info('Removing calibration hours finished.')
    logging.info(f'Counting of particles per hour in RapidE data in directory {dir_path} started.')
    dfR = read_data_dir(dir_path)
    logging.info(f'Counting of particles per hour in RapidE data in directory {dir_path} finished.')
    logging.info('Joining of Hirst and RapidE data started.')
    df = join_hirst_rapid_data(dfH, dfR)
    logging.info('Joining of Hirst and RapidE data finished.')
    logging.info(f'Adjasting of time resolution to {time_res} started.')
    df = set_time_resolution(df,time_res)
    logging.info(f'Adjasting of time resolution to {time_res} finished.')
    df['MONTH'] = list(map(lambda x: x.month, list(df['HOUR'])))
    logging.info('Cluster analysis of selected pollen types started.')
    labels = clusters_from_seasons(df, pollen_types, dfP)
    logging.info('Cluster analysis of selected pollen types finished.')
    df = assign_clusters(df, pollen_types, labels)
    #df =  df[['MONTH','HOUR', 'FILENAME', 'CLUSTER'] + pollen_types]
    return df

def gridsearch_hparam_from_json(filepath):
    with open(filepath) as file:
        hps = json.load(file)
        
    values = []
    hpnames = []
    for hpname in hps:
        values.append(hps[hpname])
        hpnames.append(hpname)
    
    comb = []
    for element in itertools.product(*values):
        hp = {}
        for i, hpname in enumerate(hpnames):
            hp[hpname] = element[i]
        comb.append(hp)
    return comb


def my_collate(batch):

    #data = [item[0] for item in batch]
    #target = torch.cat([item[1] for item in batch], dim=0)
    #print(data)
    #print(target)
    #sys.exit()
    #target = (target - torch.mean(target, dim=0))/torch.std(target,dim=0)
    #weights = torch.cat([item[2] for item in batch], dim=0)
    return batch
            
            
    
    
        
    




    
    
   #dfH['HOUR'] = list(map(lambda x: x[:-12],list(dfH['HOUR'])))
    

        #dfC = open_excel_file_in_pandas(calib_info_path)
        #dfS = select





#dfH = open_excel_file_in_pandas(hirst_data_path)
#dfC = open_excel_file_in_pandas(pollen_data_path)
