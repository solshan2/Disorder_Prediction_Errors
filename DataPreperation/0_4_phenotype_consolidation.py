import pandas as pd
import json

data_path = 'C:\\Users\\kyle\\OneDrive - UWM\\SRP 2022\\phenotype\\'
output_path = 'C:\\Users\\kyle\\OneDrive - UWM\\SRP 2022\\'
scale_names = ['sans','saps','bipolar_ii']

item_description_df_list = []
item_response_list = []
scale_metadata_list = []

scale = scale_names[0]


for scale in scale_names:
  description_dict = json.load(open(f'{data_path}{scale}.json'))
  for item_name in description_dict.keys():
    if item_name =='MeasurementToolMetadata':
      item_df = pd.DataFrame({
        "Scale":[scale],
        item_name:[description_dict[item_name]['Description']]
        })
      scale_metadata_list.append(item_df)
    else:
      item_df = pd.DataFrame({
        "Scale":[scale],
        "Item":[item_name],
        "Description":[description_dict[item_name]['Description']]
      })
      if "Levels" in description_dict[item_name].keys():
        item_df["Levels"] = len(description_dict[item_name]['Levels'].keys())
        item_df["Derivative"] = "FALSE"
        for level in description_dict[item_name]['Levels'].keys():
          response_df = pd.DataFrame({
            "Scale":[scale],
            "Item":[item_name],
            "Level":[level],
            "Response":[description_dict[item_name]['Levels'][level]]
          })
          item_response_list.append(response_df)
      else:
        item_df['Levels'] = pd.NA
        item_df["Derivative"] = "TRUE"
      item_description_df_list.append(item_df)

item_description_df = pd.concat(item_description_df_list)
response_df = pd.concat(item_response_list)
metadata_df = pd.concat(scale_metadata_list)

item_description_df.to_csv(f'{output_path}item_descriptions.csv', index=False)
response_df.to_csv(f'{output_path}response_descriptions.csv', index=False)
metadata_df.to_csv(f'{output_path}scale_metadata.csv', index=False)