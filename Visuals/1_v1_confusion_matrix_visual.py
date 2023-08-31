import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import plotly.figure_factory as ff
import plotly

# Read in data 
predictions_metadata_df = pd.read_csv('predictions_metadat_confusion_v2.csv')
visuals_path = f'C:\\Users\\kyle\\UWM\\SRP Paper 01-Group - Documents\\Analytics\\UpdatedVisuals\\'
print('sessions_used', predictions_metadata_df['sessions_used'].unique())
print('subjects_used', predictions_metadata_df['subjects_used'].unique())
predictions_metadata_nc = predictions_metadata_df[predictions_metadata_df['subjects_used']== 'No CONTROL']
task_df = predictions_metadata_df[predictions_metadata_df['sessions_used']== '6:noRest-noBht']

x = ['SCZ', 'BD', 'ADHD']
y = ['ADHD', 'BD', 'SCZ']

sessions_used = '6_noRest-noBht'
conf_mat1 = confusion_matrix(task_df['y_test'], task_df['preds'])
conf_mat2 = np.flipud(conf_mat1)
conf_mat_percent = conf_mat2.astype(float)
for i in range(conf_mat1.shape[0]):
    conf_mat_percent[i,] = conf_mat_percent[i,]/sum(conf_mat_percent[i])

averaged_conf_mat_percent = np.array([
  [
    .2026, # ADHD as SCZ
    .3895, # BD as SCZ
    .5243, # SCZ as SCZ
  ],
  [
    .4470, # ADHD as BD
    .4665, # BD as BD
    .3561 # SCZ as BD
  ],
  [
    .3504, # ADHD as ADHD
    .1439, # BD as ADHD
    .1196, # SCZ as ADHD    
  ],
]).transpose()
#############
conf_mat_percent = averaged_conf_mat_percent
#############
# setting text
confusion_matrix_text = [['{:.2%}'.format(y) for y in x] for x in conf_mat_percent]
# creates figure object we will use to display things 
fig = ff.create_annotated_heatmap(
  conf_mat_percent, 
  x=x, 
  y=y, 
  annotation_text=confusion_matrix_text, 
  colorscale='Tempo', 
  reversescale=False,
  showscale=False
)
# add title
# fig.update_layout(title_text=f'<i><b>Diagnosis Confusion Matrix ({sessions_used})</b></i>',
#                   #xaxis = dict(title='x'),
#                   #yaxis = dict(title='x')
#                   )
# add custom xaxis title
fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=0.5,
                        y=1.15,
                        showarrow=False,
                        text="Predicted",
                        xref="paper",
                        yref="paper"))

# add custom yaxis title
fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=-0.15,
                        y=0.5,
                        showarrow=False,
                        text="True",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))
# adjust margins to make room for yaxis title
fig.update_layout(
    margin=dict(t=50, l=90),
    # ,height=2000, 
    # width=2000, 
    )
# add colorbar
# fig['data'][0]['showscale'] = True
#fig.show()
fig.write_image(f'{visuals_path}confusion_{sessions_used}_v4.png')
fig.show()