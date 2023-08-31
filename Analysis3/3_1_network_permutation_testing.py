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

def permute_subset(df, column_index, random_state = 812):
  column_index_list = [x for x in range(19900)]
  original_columns = list(df.columns)
  target_df = df[[original_columns[i] for i in column_index]]
  other_df = df[[original_columns[i] for i in column_index_list if i not in column_index]]
  premuted_target_df = shuffle(target_df, random_state=random_state)
  full_df = pd.concat([other_df, premuted_target_df.reset_index(drop=True)], axis=1,)
  # full_df.columns = original_columns
  full_df = full_df[original_columns]
  return full_df

#### Testing ####
ind = 1
df = data_split_dict[ind]['train'][colnames]
k = 'Vis_ALL' 
column_index = ind_dict[k]
random_state = 1




def svc_predgen(train_x, train_y, test_x, test_y, split, label='', C=1, random_state=812, kernel= 'linear', class_weight= None):
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
    'split_index':split,
    'label':label
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
    'label':[label],
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

# Generate column indices to determine feature groupings #
# read in index of edge identity
index_df = pd.read_csv(f'{project_path}edge_index.csv')
ind_dict = {}
## Networks ##
for network in index_df['Network1'].unique():
  between_network_connections = index_df[
    (index_df['Network1']==network) |
    (index_df['Network2']==network)
  ]
  ind_dict[f'{network}_ALL'] = list(between_network_connections.index)

for network_connection in index_df['Network_Connection'].unique():
  network_connections = index_df[index_df['Network_Connection']==network_connection]
  ind_dict[network_connection] = list(network_connections.index)

for k in ind_dict.keys():
  print(k, len(ind_dict[k]))



## Regions ##
# for name in index_df['Name1'].unique():
#   between_region_connections = index_df[
#     (index_df['Name1']==name) |
#     (index_df['Name2']==name)
#   ]
#   ind_dict[f'{name}_ALL'] = list(between_region_connections.index)

# This version does not consider lateralization
# for region_connection in index_df['Name_Connection'].unique():
#   region_connections = index_df[index_df['Name_Connection']==region_connection]
#   ind_dict[region_connection] = list(region_connections.index)

# This version includes lateralization
# for region_connection in index_df['Name_Connection2'].unique():
#   region_connections = index_df[index_df['Name_Connection2']==region_connection]
#   ind_dict[region_connection] = list(region_connections.index)

## Parcels ##
#Not doing this for now; it would take way too long


data_split_dict = {}
for ind in range(10):
  # asign train and test values to the new dict with data instead of just indices
  data_split_dict[ind] = {
    'train':pd.merge(pd.read_csv(f'{project_path}NC-6_SplitInfo_Train_{ind}.csv'),full_data,how='left',on=['participant_id','Task','diagnosis']),
    'test':pd.merge(pd.read_csv(f'{project_path}NC-6_SplitInfo_Test_{ind}.csv'),full_data,how='left',on=['participant_id','Task','diagnosis'])
  }

c = 1
res_df_dict = {}
length_list = []
for k in ind_dict.keys():
  # print(k, len(ind_dict[k]))
  length_list.append(len(ind_dict[k]))

print(length_list)
length_list = [
  66, 231, 264, 312, 
  325, 348, 360, 406, 
  420, 435, 552, 572, 
  595, 638, 660, 754, 
  770, 780, 870, 910, 
  1012, 1015, 1035, 1050, 
  1196, 1334, 1380, 1610, 
  2322, 4147, 4849, 5365, 
  5535, 6370, 8119
]

column_index_list = [x for x in range(19900)]
res_df_dict = {}
for k in ind_dict.keys(): # Iterate through subsets to permute
  print(k)
  for i in range(10): # run all CV splits with 10 permutations of the target set each
    print(f'\tRandomState set {i}')
    res_dfs = Parallel(n_jobs = 23)(
      delayed(svc_predgen)(
        train_x = permute_subset(data_split_dict[ind]['train'][colnames],column_index = ind_dict[k],random_state=i),
        train_y = data_split_dict[ind]['train']['diagnosis'].values.ravel(),
        test_x = permute_subset(data_split_dict[ind]['test'][colnames],column_index = ind_dict[k],random_state=i+1),
        test_y = data_split_dict[ind]['test']['diagnosis'].values.ravel(),
        split = f'{i}_{ind}',
        label = f'{k}'
      ) for ind in data_split_dict.keys()
    )
    res_df_full = pd.concat(res_dfs)
    res_df_dict[f'{k}_{i}'] = res_df_full
    res_df_full.to_csv(f'{project_path}FeatureImportance_Results_{k}-{i}.csv', index=False)


for k in ind_dict.keys(): # Iterate through subsets to permute
  print(f'{k} - random equivalent')
  for i in range(10): # run all CV splits with 10 permutations of the target set each
    print(f'\tRandomState set {i}')
    res_dfs = Parallel(n_jobs = 23)(
      delayed(svc_predgen)(
        train_x = permute_subset(data_split_dict[ind]['train'][colnames],column_index = np.random.choice(column_index_list, size=len(ind_dict[k]), replace=False),random_state=i),
        train_y = data_split_dict[ind]['train']['diagnosis'].values.ravel(),
        test_x = permute_subset(data_split_dict[ind]['test'][colnames],column_index = np.random.choice(column_index_list, size=len(ind_dict[k]), replace=False), random_state=i+1),
        test_y = data_split_dict[ind]['test']['diagnosis'].values.ravel(),
        split = f'{i}_{ind}',
        label = f'Rand-{len(ind_dict[k])}'
      ) for ind in data_split_dict.keys()
    )
    res_df_full = pd.concat(res_dfs)
    res_df_dict[f'Rand-{len(ind_dict[k])}_{i}'] = res_df_full
    res_df_full.to_csv(f'{project_path}FeatureImportance_Results_Rand-{len(ind_dict[k])}-{i}.csv', index=False) 

res_dfs = []
for k in res_df_dict.keys():
  res_dfs.append(res_df_dict[k])

# res_dfs = []
# for k in ind_dict.keys():
#   for i in range(10): 
#     res_df = pd.read_csv(f'{project_path}FeatureImportance_Results_{k}-{i}.csv')
#     res_dfs.append(res_df)


res_df_full = pd.concat(res_dfs)
res_df_full.to_csv(f'{project_path}FeatureImportance_Results.csv', index=False)

# oops I forgot to log the size of the set that was permuted
size_dict = {}
for k in ind_dict.keys(): # Iterate through subsets 
  size_dict[f'{k}'] = [len(ind_dict[k])]
  size_dict[f'Rand-{len(ind_dict[k])}'] = [len(ind_dict[k])]

size_index_df = pd.DataFrame.from_dict(size_dict, orient='index', columns = ['NumberPermuted'])
size_index_df['Label'] = size_index_df.index

size_index_df.to_csv(f'{project_path}FeatureImportance_index.csv', index=False)