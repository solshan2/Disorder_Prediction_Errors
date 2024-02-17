# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:58:02 2023

@author: sarah
"""
# Packages 
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import os
sep = os.path.sep
# Read in data 
predictions_metadata_df = pd.read_csv(f'S:{sep}UCLA_Decoding_v2{sep}predictions_metadat_confusion_v2.csv')
predictions_metadata_nc = predictions_metadata_df[predictions_metadata_df['subjects_used']== 'No CONTROL']
# Filter to use all sessions (removing just rest, and just individual tasks)
predictions_metadata_nc_all = predictions_metadata_nc[predictions_metadata_nc['sessions_used']== 'all']
#predictions_metadata_nc_all = predictions_metadata_nc_all[(predictions_metadata_nc_all['C']== 1.00) & (predictions_metadata_nc_all['class_weight']!= 'balanced') & (predictions_metadata_nc_all['kernel']== 'linear')]
predictions_metadata_nc_all = predictions_metadata_nc_all[(predictions_metadata_nc_all['C']== 1.00) & (predictions_metadata_nc_all['class_weight']!= 'balanced') & (predictions_metadata_nc_all['kernel']== 'linear')]

conf_mat1 = confusion_matrix(predictions_metadata_nc_all['y_test'], predictions_metadata_nc_all['preds'])

conf_mat_percent = conf_mat1.astype(float)
for i in range(conf_mat1.shape[0]):
    conf_mat_percent[i,] = conf_mat_percent[i,]/sum(conf_mat_percent[i])

print(conf_mat_percent)

predictions_metadata_df['sessions_used'].unique()

pred_df_2 = predictions_metadata_df[
    (predictions_metadata_df['sessions_used']!='all') &
    (predictions_metadata_df['sessions_used']!='rest') &
    (predictions_metadata_df['sessions_used']!='bht')
    ]
predictions_metadata_nc2 = pred_df_2[pred_df_2['subjects_used']== 'No CONTROL']
conf_mat1 = confusion_matrix(predictions_metadata_nc2['y_test'], predictions_metadata_nc2['preds'])
conf_mat_percent = conf_mat1.astype(float)
for i in range(conf_mat1.shape[0]):
    conf_mat_percent[i,] = conf_mat_percent[i,]/sum(conf_mat_percent[i])

print(conf_mat_percent)

# Filter out models to that same hyperparam used for all- linear, C = 1, no balanced 
predictions_metadata_nc2 = predictions_metadata_nc2[
    (predictions_metadata_nc2['C']== 1.00) #&
    #(predictions_metadata_nc2['class_weight']!= 'balanced') &
    #(predictions_metadata_nc2['kernel']== 'linear')
    ]

true_schz = predictions_metadata_nc2[predictions_metadata_nc2['y_test']=='SCHZ']
true_bipolar = predictions_metadata_nc2[predictions_metadata_nc2['y_test']=='BIPOLAR']

true_schz['schz_as_bipolar'] = 0
true_schz.loc[true_schz['preds']=='BIPOLAR','schz_as_bipolar']= 1

true_bipolar['bipolar_as_schz'] = 0
true_bipolar.loc[true_bipolar['preds']=='SCHZ','bipolar_as_schz']= 1

schz_as_bipolar_pivot = true_schz.pivot_table(values = ['schz_as_bipolar'], index = ['Participant_ID'], aggfunc = [np.mean])
bipolar_as_schz_pivot = true_bipolar.pivot_table(values = ['bipolar_as_schz'], index = ['Participant_ID'], aggfunc = [np.mean])

phen_data = 'C:\\Users\\kyle\\UWM\\SRP Paper 01-Group - Documents\\Analytics\\phenotype\\'
target_files = []
for filename in os.listdir(phen_data):
  if '.tsv' in filename:
    target_files.append(f'{phen_data}{filename}')

curfile = pd.read_csv( 'C:\\Users\\kyle\\UWM\\SRP Paper 01-Group - Documents\\Analytics\\participants.tsv', sep = '\t')
for filename in target_files:
  curfile = pd.merge(curfile, pd.read_csv(filename, sep  = '\t'), on = 'participant_id')


phen_acc_schz = pd.DataFrame(index=schz_as_bipolar_pivot.index)
phen_acc_schz['participant_id'] = phen_acc_schz.index
phen_acc_schz['rate_misclass']= schz_as_bipolar_pivot[('mean','schz_as_bipolar')] 
phen_acc_schz = pd.merge (phen_acc_schz,curfile, on = 'participant_id', how = 'left')

# Drop cols where all the vallues in col are Na so we get rid of places where they didn't do a certain questionnaire 
phen_acc_schz.dropna(axis=1, how = 'all', inplace=True)

phen_acc_bipolar= pd.DataFrame(index=bipolar_as_schz_pivot.index)
phen_acc_bipolar['participant_id'] = phen_acc_bipolar.index
phen_acc_bipolar['rate_misclass']= bipolar_as_schz_pivot[('mean','bipolar_as_schz')] 
phen_acc_bipolar = pd.merge(phen_acc_bipolar,curfile, on = 'participant_id', how = 'left')

phen_acc_bipolar.dropna(axis=1, how = 'all', inplace=True)

phen_acc_schz.to_csv('C:\\Users\\kyle\\UWM\\SRP Paper 01-Group - Documents\\UCLA_Decoding_v2\\phen_acc_schz-v2.csv', index=False)
phen_acc_bipolar.to_csv('C:\\Users\\kyle\\UWM\\SRP Paper 01-Group - Documents\\UCLA_Decoding_v2\\phen_acc_bipolar-v2.csv', index=False)


# Redo confusion matrix to replicate what is on the poster- did we use bht and rest tasks or just the other 6?


# Run regression again (done with all 8 tasks) 

## Redo with just 6 tasks 

## Run with demographics (gender, etc..)

