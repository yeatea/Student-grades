# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:03:07 2024

@author: yetun
"""

# Import necessary libraries
import streamlit as st  # For creating the web app interface
import pandas as pd  # For data manipulation and handling
from joblib import load  # For loading the pre-trained machine learning model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder  # For data preprocessing
from PIL import Image