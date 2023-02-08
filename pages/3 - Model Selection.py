#############
## Imports ##
#############

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import joblib

# Regression Models
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Clustering Models
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import MeanShift

#########################
## Important Variables ##
#########################

model_selection = '''
Model selection is a crucial step in the machine learning process as it determines the performance
of the algorithm on a particular dataset. The choice of model can greatly impact the accuracy and 
efficiency of the predictions. Selecting an appropriate model involves evaluating various algorithms
and determining which one best fits the data characteristics and the problem to be solved. Proper model
selection can result in improved accuracy, reduced overfitting, and increased interpretability, making 
it an essential part of the machine learning workflow.
'''

find_model = '''
The choice of model depends on several different factors such as computational cost, bias-variance trade-off,
interpretability, and generalization ability. To determine the appropriate model for your dataset, it is 
important to consider the following questions based on the 
<a href='https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html'>sklearn model map</a>.
'''

lack_of_data = '''**IMPORTANT: The uploaded dataset seems to have no more than 50 data points. 
Due to the severe lack of data, consider collecting more data.**'''

uploading_model = '''
The files you upload must have the extension ".pkl". This is a specific file format that is used for storing
python objects including trained machine learning models. Failure to follow this format may result in errors
or an inability to process the files.
'''

######################
## Helper Functions ##
######################

def recommend_models():
  '''
  Returns a list of recommended models based on the chosen dataset and other inputs provided by the user.
  '''
  target = pd.read_csv('data/target.csv')
  target_len = len(target)
  
  # If the user has less than or equal to 50 samples, warn them about collecting more data
  if (target_len <= 50):
    st.write(lack_of_data)
  
  predicting_category = st.selectbox('Are you trying to predict a category?', ['Yes', 'No'])
  
  recommended_models = []
  if (predicting_category == 'Yes'):
    labeled_data = st.selectbox('Do you have labeled data?', ['Yes' ,'No'])
    
    # Classification
    if (labeled_data == 'Yes'):
      if (target_len < 100000):
        textual_data = st.selectbox('Do you have textual data in your dataset?', ['Yes', 'No'])
        if (textual_data == 'Yes'):
          recommended_models += [GaussianNB()]
        else:
          recommended_models += [KNeighborsClassifier(), SVC(), LinearSVC(), RandomForestClassifier()]
      else:
        recommended_models += [SGDClassifier()]
    # Clustering
    else:
      categories_known = st.selectbox('Do you know the number of categories you want to structure your data in?', ['Yes', 'No'])
      if (categories_known == 'Yes'):
        recommended_models += [MeanShift(), BayesianGaussianMixture()]
      else:
        recommended_models += [KMeans(), MiniBatchKMeans(), SpectralClustering(), GaussianMixture()]
  else:
    predicting_quantity = st.selectbox('Are you trying to predict a quantity?', ['Yes', 'No'])
    if (target_len < 100000):
      feature_importance = st.selectbox('How many of the features in your dataset are important?', ['Most', 'Few'])
      if (feature_importance == 'Most'):
        recommended_models += [Ridge(), SVR(), RandomForestRegressor()]
      else:
        recommended_models += [Lasso(), ElasticNet()]
    else:
      recommended_models += [SGDRegressor()]
      
  return recommended_models

def submission(model_names):
  '''
  Function that submits models and produces aesthetically pleasing displays. 
  '''
  if (st.button('Submit Models')):
    placeholder = st.empty()
    
    with placeholder.container():
      st.info('Submitting models...')
      time.sleep(2)
      
    placeholder.empty()
        
    for model in model_names:
      model_name = type(model).__name__
      joblib.dump(model, f'models/{model_name}.pkl')
      st.success(f'{model_name} successfully submitted!', icon='✅')
  
    st.balloons()
    
    

#################
## Application ##
#################

st.title('Model Selection')

with st.expander('What is Model Selection?'):
  st.write(model_selection, unsafe_allow_html=True)
  
if (len(os.listdir('data')) == 0):
  st.error('Dataset not yet selected!', icon='🚨')
  st.stop()

model_choice = st.selectbox('Choose one', ['Find Models', 'Upload Models'])

if (model_choice == 'Find Models'):
  st.header('Find Models')
  
  with st.expander('Important Instructions'):
    st.write(find_model, unsafe_allow_html=True)
  
  recommended_models = recommend_models() 
    
  st.subheader('Recommended Models')
  for model in recommended_models:
    model_name = type(model).__name__
    st.button(model_name)

  model_options = st.multiselect('Choose the models you want to later train and evaluate', recommended_models)
  
  # Submission
  submission(model_options)
else:
  st.header('Upload Models')
  
  with st.expander('Important Instructions'):
    st.write(uploading_model, unsafe_allow_html=True)
  
  model_files = st.file_uploader('Upload Models', accept_multiple_files=True)
  
  # Submission
  if (st.button('Submit Models') and model_files is not None):
    submission_bar = st.progress(0)
    for percent_complete in range(100):
      time.sleep(0.01)
      submission_bar.progress(percent_complete + 1)
    
    for model_file in model_files:
      model = joblib.load(model_file)
      model_name = type(model).__name__
      joblib.dump(model, f'models/{model_name}.pkl')
      st.success(f'{model_name} successfully submitted!', icon='✅')
    
    st.balloons()  