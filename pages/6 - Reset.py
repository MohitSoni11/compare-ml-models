#############
## Imports ##
#############

import streamlit as st
import pandas as pd
import numpy as np
import os

#########################
## Important Variables ##
#########################

reset = '''
The reset page allows you to restart your process by clearing the current dataset and model selections,
allowing you to start fresh with a new dataset and model choices. Keep in mind that resetting will 
permanently delete any progress you have made, so make sure to reset only when necessary. Once you have confirmed 
the reset, simply return to the "Choose Dataset" page and restart your process anew by entering data and
choosing models.
'''

######################
## Helper Functions ##
######################

def remove_dir_files(dir_path):
  for file in os.scandir(dir_path):
    if (not file.name == 'placeholder.txt'):
      os.remove(file.path)

#################
## Application ##
#################

st.title('Reset')
with st.expander('Submitted something you didn\'t want to?'):
  st.write(reset)
  
confirm_reset = st.text_input('Type "Reset" and press enter to confirm your choice of resetting.')

if (confirm_reset == 'Reset'):
  remove_dir_files('data')
  remove_dir_files('data_test')
  remove_dir_files('models')
  remove_dir_files('models_trained')
  st.success('Reset Successful!', icon='✅')