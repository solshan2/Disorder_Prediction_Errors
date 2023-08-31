#!/usr/bin/env python3
import os
import sys
import io
import pandas as pd
import math
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from sklearn.utils import shuffle
import json
import hashlib
from joblib import Parallel, delayed

project_path = '/data/hx-hx1/kbaacke/datasets/UCLA_Decoding/'
sep = '/'

input_path = f'{project_path}'
meta_path = f'{project_path}metadata{sep}'
pred_path = f'{project_path}comparison{sep}'
acc_path = f'{project_path}'


# Read in full data'
uid = '69354adf'
full_data = pd.read_csv(f'/data/hx-hx1/kbaacke/datasets/UCLA/{uid}_FunctionalConnectomes.csv')

parcel_info = pd.read_csv('/data/hx-hx1/kbaacke/Code/Parcellation/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
node_names = list(parcel_info['ROI Name'])

def connection_names(corr_matrix, labels):
  name_idx = np.triu_indices_from(corr_matrix, k=1)
  out_list = []
  for i in range(len(name_idx[0])):
    out_list.append(str(labels[name_idx[0][i]]) + '|' + str(labels[name_idx[1][i]]))
  return out_list

dummy_array = np.zeros((200,200))
colnames = connection_names(dummy_array, node_names)

##standardizing data 
def scale_subset(df, cols_to_exclude):
  df_excluded = df[cols_to_exclude]
  df_temp = df.drop(cols_to_exclude, axis=1, inplace=False)
  df_temp = mean_norm(df_temp)
  df_ret = pd.concat([df_excluded, df_temp], axis=1, join='inner')
  return df_ret

def mean_norm(df_input):
  return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

scaled_data = scale_subset(full_data, list(full_data.columns)[:16] + ['Subject','Task'])

full_data = scaled_data
##filter out controls##
no_controls_df = full_data.loc[full_data["diagnosis"] != "CONTROL"]
## Filter out rest and bht ##
no_controls_df = no_controls_df[
  (no_controls_df['Task']!= 'bht') &
  (no_controls_df['Task']!= 'rest')
]

def generate_uid(metadata, length=8):
  dhash = hashlib.md5()
  encoded = json.dumps(metadata, sort_keys=True).encode()
  dhash.update(encoded)
  run_uid = dhash.hexdigest()[:length]
  return run_uid

def svc_predgen(train_x, train_y, test_x, test_y, split, C=1, random_state=812, kernel= 'linear', class_weight= None):
  start_time = dt.datetime.now()
  clf_svm = SVC(kernel=kernel, random_state=random_state, C = c, class_weight=class_weight)
  clf_svm.fit(train_x, train_y)
  training_accuracy = clf_svm.score(train_x, train_y)
  predictions = clf_svm.predict(test_x)
  test_accuracy = clf_svm.score(test_x, test_y)
  y_pred = clf_svm.predict(test_x)
  classification_rep = classification_report(test_y, y_pred, output_dict=True)
  confusion_mat = confusion_matrix(test_y, y_pred)
  meta_dict = {
    'data_uid':uid,
    'Classifier':'SVC',
    'C':c,
    'kernal':kernel,
    'class_weight':class_weight,
    'random_state':random_state,
    'features':list(train_x.columns),
    'split_index':split
  }
  pred_uid = generate_uid(meta_dict)
  meta_dict['train_accuracy'] = training_accuracy
  end_time = dt.datetime.now()
  # with open(f'{meta_path}{pred_uid}_metadata.json', 'w') as outfile:
  #   json.dump(meta_dict, outfile)
  results_dict = {
    'data_uid':[uid],
    'pred_uid':[pred_uid],
    'Classifier':['SVC'],
    'C':[c],
    'kernal':[kernel],
    'class_weight':[class_weight],
    'random_state':[random_state],
    'n_features':[len(train_x.columns)],
    'training_accuracy':[training_accuracy],
    'test_accuracy':[test_accuracy],
    'split_index':[split],
    'runtime':[(end_time - start_time).total_seconds()]
  }
  for level in ['macro avg', 'weighted avg']:
    for k in classification_rep[level]:
      results_dict[f'{level}|{k}'] = [classification_rep[level][k]]
  for group in set(train_y):
    group_list = ['ADHD','BIPOLAR','SCHZ']
    results_dict[f'{group}|N'] = [np.sum(confusion_mat[group_list.index(group),])]
    results_dict[f'{group}|accuracy'] = [confusion_mat[group_list.index(group),group_list.index(group)]/np.sum(confusion_mat[group_list.index(group),])]
    for k in classification_rep[str(group)].keys():
      results_dict[f'{group}|{k}'] = [classification_rep[str(group)][k]]
    for group2 in set(train_y):
      results_dict[f'{group}|Predicted{group2}'] = [confusion_mat[group_list.index(group),group_list.index(group2)]]
    for group2 in set(train_y):
      results_dict[f'{group}|Predicted{group2}_percent'] = [confusion_mat[group_list.index(group),group_list.index(group2)]/np.sum(confusion_mat[group_list.index(group),])]
  res_df = pd.DataFrame(results_dict)
  # result_comparison = pd.DataFrame({'preds': y_pred, 'y_test': test_y})
  # result_comparison.to_csv(f'{pred_path}{pred_uid}_results.csv')
  return res_df


cv_splitdict_nc = {}
for ind in range(10):
  cv_splitdict_nc[ind] = (pd.read_csv(f'{project_path}NC-6_SplitInfo_Train_{ind}.csv'), pd.read_csv(f'{project_path}NC-6_SplitInfo_Test_{ind}.csv'))

data_split_dict = {}
for ind in range(10):
  # assign train and test values to the new dict with data instead of just indices
  data_split_dict[ind] = {
    'train':pd.merge(pd.read_csv(f'{project_path}NC-6_SplitInfo_Train_{ind}.csv'),full_data,how='left',on=['participant_id','Task','diagnosis']),
    'test':pd.merge(pd.read_csv(f'{project_path}NC-6_SplitInfo_Test_{ind}.csv'),full_data,how='left',on=['participant_id','Task','diagnosis'])
  }

c = 1
res_df_dict = {}
for i in range(10):
  for j in range(10):
    res_dfs = Parallel(n_jobs = 23)(
      delayed(svc_predgen)(
            train_x = shuffle(data_split_dict[ind]['train'], random_state=(k+(i*10)+(j*100)))[colnames],
            train_y = data_split_dict[ind]['train']['diagnosis'].values.ravel(),
            test_x = shuffle(data_split_dict[ind]['test'], random_state=(k+(i*10)+(j*100))+1)[colnames],
            test_y = data_split_dict[ind]['test']['diagnosis'].values.ravel(),
            split = f'{k+(i*10)+(j*100)}_{ind}'
        ) for ind in data_split_dict.keys() for k in range(10)
    )
    res_df_full = pd.concat(res_dfs)
    res_df_dict[f'{i}_{j}'] = res_df_full
    res_df_full.to_csv(f'{project_path}PermutationTesting_Results_{i}-{j}.csv', index=False)


res_dfs = []
for i in range(10):
  for j in range(10):
    res_df = pd.read_csv(f'{project_path}PermutationTesting_Results_{i}-{j}.csv')
    res_dfs.append(res_df)


res_df_full = pd.concat(res_dfs)
res_df_full.to_csv(f'{project_path}PermutationTesting_Results.csv', index=False)
