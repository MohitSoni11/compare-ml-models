#############
## Imports ##
#############

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets

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

own_dataset = '''
Please ensure that you upload a **cleaned version of your CSV file** as the program will not perform any additional
manipulations on your data. It is equally important to **upload the feature and target data separately**, so the
program can determine what you aim to predict and on which data.
'''

######################
## Helper Functions ##
######################

def store_standard_dataset(dataset_name):
  if (dataset_name == 'Iris'):
    dataset = datasets.load_iris()
  elif (dataset_name == 'Diabetes'):
    dataset = datasets.load_diabetes()
  elif (dataset_name == 'Digits'):
    dataset = datasets.load_digits()
  elif (dataset_name == 'Wine'):
    dataset = datasets.load_wine()
  else:
    dataset = datasets.load_breast_cancer()
  
  X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
  y = pd.DataFrame(dataset.target, columns=['target'])
  description = dataset.DESCR
  return X, y, description

def store_own_dataset(csv_file):
  if (csv_file is not None):
    dataset = pd.read_csv(csv_file)
  else:
    return
  return dataset
  
#################
## Application ##
#################

st.title('Choose Dataset')
st.write(choose_dataset)

# Seeing if user wants to enter own dataset or use one of the standard datasets provided by sklearn
dataset_choice = st.selectbox('Choose one option', ['Work with standard dataset', 'Upload own dataset'])

# Different actions taken depending on the dataset the user decides to use
if (dataset_choice == 'Work with standard dataset'):
  st.header('Choose Standard Dataset')
  dataset_name = st.selectbox('', ['Iris', 'Diabetes', 'Digits', 'Wine', 'Breast Cancer'])
  X, y, description = store_standard_dataset(dataset_name)
  st.write(pd.concat([X, y], axis=1))
  
  # User decides if they want to see the description of the dataset
  if (st.button('Dataset Description')):
    st.write(description)
else:
  st.header('Upload Own Dataset')
  st.write(own_dataset)
  file_X = st.file_uploader('Upload a CSV (Only Features)')
  file_y = st.file_uploader('Upload a CSV (Only Target)')
  X, y = store_own_dataset(file_X), store_own_dataset(file_y)
  
  if (X and y is not None):
    st.write(pd.concat([X, y], axis=1))
  
