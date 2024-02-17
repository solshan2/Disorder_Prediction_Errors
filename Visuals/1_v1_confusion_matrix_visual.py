import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import plotly.figure_factory as ff
import plotly

# Read in data 
predictions_metadata_df = pd.read_csv('C:\\Users\\kyle\\UWM\\SRP Paper 01-Group - Documents\\UCLA_Decoding_v2\\predictions_metadat_confusion_v2.csv')
visuals_path = f'C:\\Users\kyle\\UWM\\SRP Paper 01-Group - Documents\\OpenScience_Github\Visuals\\'
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
    0.220881551, # ADHD as SCZ
    0.357634751, # BD as SCZ
    0.474476766, # SCZ as SCZ
  ],
  [
    0.457639961, # ADHD as BD
    0.471273495, # BD as BD
    0.375964626 # SCZ as BD
  ],
  [
    0.321478488, # ADHD as ADHD
    0.171091755, # BD as ADHD
    0.149558607, # SCZ as ADHD    
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
  showscale=False,
  colorbar=dict(
    # title="Percent Classification/Misclassification Rate",
    
    tickvals=[.15, .2, .25, .3, .35, .4, .45],
    ticktext=[
        # .15:"15%", 
        # .2:"20%", 
        # .25:"25%",
        # .3:"30%", 
        # .35:"35%", 
        # .4:"40%",
        # .45:"45%"
        "15%", 
        "20%", 
        "25%",
        "30%", 
        "35%", 
        "40%",
        "45%"
      ]
    )
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
    height=800, 
    width=1000, 
    font=dict(family="Arial",
      size=14))
# add colorbar
fig['data'][0]['showscale'] = True

#fig.show()
fig.write_image(f'{visuals_path}confusion_{sessions_used}_v4.jpg', scale=4)
fig.show()