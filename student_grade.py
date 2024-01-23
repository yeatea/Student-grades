# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:03:07 2024

@author: yetun
"""

# Import necessary libraries
import streamlit as st  # For creating the web app interface
import pandas as pd  # For data manipulation and handling
from joblib import load  # For loading the pre-trained machine learning model
from sklearn.preprocessing import LabelEncoder  # For data preprocessing

from PIL import Image

# Load our model data
model = load('student_data.joblib')

# Function to preprocess categorical columns

def load_model(df):
    return model.predict(df.values)    


def preprocess_cat(df):
    lb = LabelEncoder()
    for i in df.columns:
        if df[i].dtype == 'object':
            df[i] = lb.fit_transform(df[i])
            
    return df
    #df['famsize','Pstatus','Fjob','Mjob','sex','reason','guardian','schoolsup','famsup','paid','activities','internet','romantic']= lb.fit_transform(df['famsize','Pstatus','Fjob','Mjob','sex','reason','guardian','schoolsup','famsup','paid','activities','internet','romantic'])





# Main function to create the web app interface
def main():
    st.title('Student Performance App')
    st.write('This App is designed to know each student grades and to know how well they perform.')
    img = Image.open('pics_grade.PNG')
    st.image(img, width=500)
    # Create a dictionary to store input data
    input_data = {}
    col1, col2, col3 = st.columns(3)  # Split the interface into three columns
    
    with col1:
        # Collect user inputs for Students and Parent
        input_data['schoolsup'] = st.selectbox('schoolsup', ['Yes', 'No'])
        input_data['sex'] = st.selectbox('sex', ['F','M'])
        input_data['age'] = st.number_input('Age', min_value=15, max_value=21, step= 1)
        input_data['famsize'] = st.selectbox('famsize', ['GT3','LE3'])
        input_data['Pstatus'] = st.selectbox('Pstatus', ['A','T'])
        input_data['health'] = st.number_input('health', min_value=1, max_value=5, step= 1)
        
    with col2:
        # Collect user inputs for other indicators
        input_data['Medu'] = st.number_input('Mother Education', min_value=0, max_value=14, step= 1)
        input_data['Fedu'] = st.number_input('Father Education', min_value=0, max_value=15, step= 1)
        input_data['Mjob'] = st.selectbox('Mjob', ['at_home','health','other','services','teacher'])
        input_data['Fjob'] = st.selectbox('Fjob', ['at_home','health','other','services','teacher'])
        input_data['reason'] = st.selectbox('reason', ['course','other','home','reputation'])
        input_data['freetime'] = st.number_input('freetime', min_value=1, max_value=5, step= 1)
        
        
    with col3:
        # Collect user inputs for other indicators
        input_data['guardian'] = st.selectbox('guardian', ['mother','father','other'])
        input_data['traveltime'] = st.number_input('Travel Time', min_value=1, max_value=4, step= 1)
        input_data['studytime'] = st.number_input('Study Time', min_value=1, max_value=5, step= 1)
        input_data['failures'] = st.number_input('Failures', min_value=0, max_value=2, step= 1)
        input_data['absences'] = st.number_input('absences', min_value=0, max_value=20, step= 1)
        input_data['famrel'] = st.number_input('famrel', min_value=1, max_value=5, step= 1)

    
            # collect input data into a dataframe and display on screen
    input_df = pd.DataFrame([input_data])
    st.write(input_df)
    
    if st.button('Predict'):
        final_df = preprocess_cat(input_df)
        st.write(final_df)
        predictions = load_model(final_df)
        
        st.write(predictions)
    

# Run the app
if __name__ == '__main__':
    main()