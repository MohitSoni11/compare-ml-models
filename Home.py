#############
## Imports ##
#############

import streamlit as st
import pandas as pd
import numpy as np
from streamlit.components.v1 import html

#########################
## Important Variables ##
#########################

intro = '''
Welcome to the ML Model Comparison App, the perfect tool for anyone looking to compare and evaluate
the performance of different machine learning models. With this app, you can easily compare models based
on key metrics such as Accuracy, Precision, Recall, MSE, MAE, etc. -- as well as compare training time. 
Whether you're a data scientist looking for the best model for a specific task or a
machine learning enthusiast exploring the field, this app is designed to provide you with valuable
insights and make your model selection process faster and more efficient.'''

#################
## Application ##
#################

st.title('ML Model Comparison App')
st.write(intro, unsafe_allow_html=True)

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['KNN', 'Random Forest', 'Logistic Regression'])

st.area_chart(chart_data, height = 500)