#############
## Imports ##
#############

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
import time

#########################
## Important Variables ##
#########################

choose_dataset = '''
Choosing the right dataset is an important step in any data analysis or machine learning project.
The dataset you choose will dictate the type of questions you can answer and the quality of your results. 
When selecting a dataset, it's important to consider factors such as the size and complexity of the data, the
quality of the data, and the relevance of the data to your project. You should also consider whether the data
has been preprocessed or if it requires additional cleaning and preparation. Ultimately, the best dataset for
your project is one that aligns with your goals and provides enough information to effectively train your
models and answer your research questions.'''

standard_dataset = '''
The standard datasets used in the program are sourced from the scikit-learn (sklearn) library, which is a popular 
machine learning library in Python. Sklearn provides a variety of datasets, including both real-world and 
artificial datasets, to help users experiment with and evaluate different machine learning algorithms. The 
toy datasets in the program are a subset of these datasets and are specifically chosen to be simple and small 
in size, making them ideal for learning and testing purposes. By using these datasets, users can gain a better 
understanding of how the algorithms work and how they can be applied to real-world problems.

Want to learn more about the sklearn toy datasets? Visit this 
<a href='https://scikit-learn.org/stable/datasets/toy_dataset.html'>website</a>.
'''

own_dataset = '''
Please ensure that you upload a **cleaned version of your CSV file** as the program will not perform any additional
manipulations on your data. It is equally important to **upload the feature and target data separately**, so the
program can determine what you aim to predict and on which data.
'''

######################
## Helper Functions ##
######################

def store_standard_dataset(dataset_name):
  '''
  Loading the `dataset_name` toy dataset from the sklearn library and splitting the data into the features and
  the target datasets. Returns the features and target datasets with a description of the dataset.
  '''
  if (dataset_name == 'Iris'):
    dataset = datasets.load_iris()
  elif (dataset_name == 'Diabetes'):
    dataset = datasets.load_diabetes()
  elif (dataset_name == 'Wine'):
    dataset = datasets.load_wine()
  else:
    dataset = datasets.load_breast_cancer()
  
  X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
  y = pd.DataFrame(dataset.target, columns=['target'])
  description = dataset.DESCR
  return X, y, description

def store_own_dataset(csv_file):
  '''
  Returning `csv_file` as a dataframe.
  '''
  if (csv_file is not None):
    dataset = pd.read_csv(csv_file)
  else:
    return
  return dataset

def submission(X, y, dataset_name):
  '''
  Function that submits a dataset and produces aesthetically pleasing displays. 
  '''
  
  if (st.button('Submit Dataset')):
    submission_bar = st.progress(0)
    for percent_complete in range(100):
      time.sleep(0.05)
      submission_bar.progress(percent_complete + 1)
    
    X.to_csv('data/features.csv', index=False)
    y.to_csv('data/target.csv', index=False)
  
    st.balloons()
    st.success(f'{dataset_name} dataset successfully submitted!', icon='✅')
  
def standard_dataset_work():
  '''
  Creates the layout of the page for if the user decides to use one of the standard sklearn toy datasets.
  '''
  # Header
  st.header('Choose Standard Dataset')
  st.write(standard_dataset, unsafe_allow_html=True)
  
  # Selecting and showing the dataset
  dataset_name = st.selectbox('', ['Iris', 'Diabetes', 'Wine', 'Breast Cancer'])
  X, y, description = store_standard_dataset(dataset_name)
  st.dataframe(pd.concat([X, y], axis=1))
  
  # Toy dataset description for the user
  with st.expander('Dataset Description'):
    st.write(description)
  
  # Submission
  submission(X, y, dataset_name) 
  
def own_dataset_work():
  '''
  Creates the layout of the page for if the user wants to upload their own dataset.
  '''
  # Header
  st.header('Upload Own Dataset')
  st.write(own_dataset, unsafe_allow_html=True)
  
  # Giving options to upload the dataset
  file_X = st.file_uploader('Upload a CSV (Only Features)')
  file_y = st.file_uploader('Upload a CSV (Only Target)')
  X, y = store_own_dataset(file_X), store_own_dataset(file_y)
  
  # Showing the dataset if both files uploaded
  if (file_X and file_y is not None):
    st.dataframe(pd.concat([X, y], axis=1))
    
    # Submission
    dataset_name = file_X.name + ' and ' + file_y.name
    submission(X, y, dataset_name) 
  
#################
## Application ##
#################

st.title('Choose Dataset')
st.write(choose_dataset, unsafe_allow_html=True)

# Seeing if user wants to enter own dataset or use one of the standard datasets provided by sklearn
dataset_choice = st.selectbox('Choose one option', ['Work with standard dataset', 'Upload own dataset'])

# Different actions taken depending on the dataset the user decides to use
if (dataset_choice == 'Work with standard dataset'):
  standard_dataset_work()
else:
  own_dataset_work()
