#Imports
import pandas as pd
import json
import os 

# Declare paths
output_path = '/data/hx-hx1/solshan2/UCLA_Decoding_v2/'
comparison_path = f'{output_path}comparison/' # Sheets with subject ID, predicted y, actual y
confusion_path = f'{output_path}confusion/' # Sheets with confusion matrices
metadata_path = f'{output_path}metadata/' # metadata json files with information about the models
probs_path = f'{output_path}probs/' # probabilities of each of the predictions? not totally sure

sep = os.path.sep
# Version to run locally
output_path = f'S:{sep}UCLA_Decoding_v2{sep}'
comparison_path = f'{output_path}comparison\\' # Sheets with subject ID, predicted y, actual y
confusion_path = f'{output_path}confusion\\' # Sheets with confusion matrices
metadata_path = f'{output_path}metadata\\' # metadata json files with information about the models
probs_path = f'{output_path}probs\\' # probabilities of each of the predictions? not totally sure

# Read in accuracies 
accuracy_df = pd.read_csv(f'{output_path}Diag_Accuracy_V2.csv')
accuracy_df.drop_duplicates(subset='runuid', inplace=True, keep='last')

# Read in confusion matrices and convert them to column format
confusion_dfs = []
# iterate through all model runs


# Read in meta-info about the models
metadata_dfs = []
# iterate through all model runs
for runuid in accuracy_df['runuid'].unique():
  # Read in the json files as dictionaries
  run_metadata_dict = json.load(open(f'{metadata_path}{runuid}_metadict.json'))
  # Convert the dictionaries to dataframes
  run_metadata_df = pd.DataFrame(run_metadata_dict, index=[0])
  # add a column to indicate the runuid
  run_metadata_df['runuid'] = runuid
  # add the run_dataframe to the list of dataframes
  metadata_dfs.append(run_metadata_df)

# Concatenate the list of metadata dataframes
metadata_df = pd.concat(metadata_dfs, ignore_index=True)
# Create a new column to indicate whether control subjects were included in the analysis
metadata_df['subjects_used'] = ''

### ! Removed this for the v2 version due to missing columns and the fact that we no longer have the problem this was supposed to solve
# accuracy_df['Task'].fillna('all', inplace=True)
# accuracy_df['sessions_used'].fillna(accuracy_df['Task'], inplace=True)
# accuracy_df['subjects_used'].fillna('all', inplace=True)



# Read in record of all test splits
test_split_dict = {
  'No_Control':{}#,
  # 'All_Subjects':{},
  # 'SCHZ_BIPOLAR':{}
}
for k in test_split_dict:
  for task in accuracy_df['sessions_used'].unique():
    test_split_dict[k][task] = {}

for task in accuracy_df['sessions_used'].unique():
  for split_ind in range(20):
    if task=='all':
      test_split_dict['No_Control'][task][split_ind] = pd.read_csv(f'{output_path}NC_SplitInfo_Test_{split_ind}.csv')
      test_split_dict['All_Subjects'][task][split_ind] = pd.read_csv(f'{output_path}SplitInfo_Test_{split_ind}.csv')
      test_split_dict['SCHZ_BIPOLAR'][task][split_ind] = pd.read_csv(f'{output_path}SchzBIPOLAR_SplitInfo_Test_{split_ind}.csv')
    elif task=='6:noRest-noBht':
      test_split_dict['No_Control'][task][split_ind] = pd.read_csv(f'{output_path}NC-6_SplitInfo_Test_{split_ind}.csv')
      # test_split_dict['SCHZ_BIPOLAR'][task][split_ind] = pd.read_csv(f'{output_path}SchzBIPOLAR-6_SplitInfo_Test_{split_ind}.csv')
    else:
      test_split_dict['No_Control'][task][split_ind] = pd.read_csv(f'{output_path}{task}_NC_SplitInfo_Test_{split_ind}.csv')
      test_split_dict['All_Subjects'][task][split_ind] = pd.read_csv(f'{output_path}{task}_SplitInfo_Test_{split_ind}.csv')

# Read in predictions on the subject level
diagnoses = ['CONTROL', 'SCHZ', 'BIPOLAR', 'ADHD']
prediction_dfs = []
confusion_dfs = []
# iterate through all model runs
for runuid in accuracy_df['runuid']:
  # Read in prediciton files
  run_predictions = pd.read_csv(f'{comparison_path}{runuid}_results.csv')
  # Drop the unneeded column
  run_predictions.drop(columns=['Unnamed: 0'], inplace=True)
  # add a column to indicate the runuid
  run_predictions['runuid'] = runuid
  # Identify the split number from the metadata_df
  split_number = list(metadata_df.loc[metadata_df['runuid']==runuid]['split'])[0]
  subjects_used = list(accuracy_df.loc[accuracy_df['runuid']==runuid]['subjects_used'])[0]
  sessions_used = list(accuracy_df.loc[accuracy_df['runuid']==runuid]['sessions_used'])[0]
  if subjects_used=='all':
    subjects_used_key = 'All_Subjects'
  else:
    subjects_used_key = 'No_Control'
  target_split_df = test_split_dict[subjects_used_key][sessions_used][split_number]
  metadata_df.loc[metadata_df['runuid']==runuid, 'subjects_used'] = subjects_used
  metadata_df.loc[metadata_df['runuid']==runuid, 'sessions_used'] = sessions_used
  # print('rp1',len(run_predictions))
  # print('sp_df',len(target_split_df))
  run_predictions['Task'] = target_split_df['Task']
  run_predictions['Participant_ID'] = target_split_df['participant_id']
  # print('rp2',len(run_predictions))
  prediction_dfs.append(run_predictions)
  # Create a dataframe with columns indicating the prediction percentage for each cell of a confusion matrix
  run_confusion_df = pd.DataFrame({'runuid':[runuid]})
  for diagnosis in run_predictions['y_test'].unique():
    for diagnosis_2 in run_predictions['y_test'].unique():
      # {True}_{Predicted}
      # _percent indicates the percent of the true diagnosis category which fell under the given prediction
      run_confusion_df[f'{diagnosis}_{diagnosis_2}'] = len(run_predictions[(run_predictions['y_test']==diagnosis) & (run_predictions['preds']==diagnosis_2)])
      run_confusion_df[f'{diagnosis}_{diagnosis_2}_percent'] = len(run_predictions[(run_predictions['y_test']==diagnosis) & (run_predictions['preds']==diagnosis_2)])/len(run_predictions[(run_predictions['y_test']==diagnosis)])
  confusion_dfs.append(run_confusion_df)

# Concatenate the list of prediction dataframes and confusion_dataframes
predictions_df = pd.concat(prediction_dfs, ignore_index=True)
predictions_df.to_csv(f'{output_path}predictions_v2.csv',index=False)
confusions_df = pd.concat(confusion_dfs, ignore_index=True)
confusions_df.to_csv(f'{output_path}confusion_v2.csv',index=False)

# Merge information to have it all in one place
accuracy_metadata_df = pd.merge(accuracy_df, metadata_df, on='runuid', how='left')
accuracy_metadata_df.to_csv(f'{output_path}accuracy_metadata_v2.csv',index=False)
accuracy_metadata_confusion_df = pd.merge(accuracy_metadata_df, confusions_df, on='runuid', how='left')
accuracy_metadata_confusion_df.to_csv(f'{output_path}accuracy_metadata_confusion_v2.csv',index=False)

predictions_metadata_df = pd.merge(predictions_df, metadata_df, on='runuid', how='left')
predictions_metadata_df.to_csv(f'{output_path}predictions_metadat_confusion_v2.csv',index=False)













# Check to see why task was not listed in the no-control splits
for k in test_split_dict.keys():
  for j in test_split_dict[k].keys():
    print(k, j, len(test_split_dict[k][j]))

# looks like only resting state was used for the no-control analyses

###CHOOSING MODELS FROM FIRST ANALYSIS###
# Looking at MISclassifications with different sets of models 

target_columns = ['testacc']
for col in confusions_df.columns:
  if 'CONTROL' not in col:
    if 'percent' in col:
      target_columns.append(col)

acc_df_nc = accuracy_metadata_confusion_df[accuracy_metadata_confusion_df['subjects_used_x']=='No CONTROL']
for session in acc_df_nc['sessions_used_x'].unique():
  acc_df_nc_session =  acc_df_nc[acc_df_nc['sessions_used_x']==session]
  print('No Control Subjects')
  print(session)
  print(acc_df_nc_session[target_columns].mean())
  
target_columns = ['testacc']
for col in confusions_df.columns:
  if 'percent' in col:
    target_columns.append(col)
  
acc_df_nc = accuracy_metadata_confusion_df[accuracy_metadata_confusion_df['subjects_used_x']=='all']
for session in acc_df_nc['sessions_used_x'].unique():
  acc_df_nc_session =  acc_df_nc[acc_df_nc['sessions_used_x']==session]
  print('All Subjects')
  print(session)
  print(acc_df_nc_session[target_columns].mean())


confusion_pivot_01 = accuracy_metadata_confusion_df.groupby(['sessions_used_x','subjects_used_x'])[target_columns].mean().transpose()
confusion_pivot_01.to_csv(f'{output_path}confusion_pivot_01.csv')

acc_df_nc[target_columns].mean()

# Filtering to test acc > or = .6
acc_df_nc_abv60 = acc_df_nc[acc_df_nc['testacc']>=.6]
# Filtering to test acc < .6
acc_df_nc_bel60 = acc_df_nc[acc_df_nc['testacc']<.6]

# NC and only models w/ acc 60% and ^ **first one is our results 
acc_df_nc_abv60[target_columns].mean()
acc_df_nc_bel60[target_columns].mean()

# Looking at it with controls 
acc_df_wc = accuracy_metadata_confusion_df[accuracy_metadata_confusion_df['subjects_used_x']=='all']
acc_df_wc[target_columns].mean()

# All models w/ and without controls 
accuracy_metadata_confusion_df[target_columns].mean()

print('This should be 128')
len(list(predictions_df['Participant_ID'].unique()))

