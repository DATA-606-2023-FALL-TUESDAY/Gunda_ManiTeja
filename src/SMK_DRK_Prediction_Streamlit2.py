import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
import pickle
import streamlit as st

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error,f1_score,classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV

import warnings
warnings.filterwarnings("ignore")

#mean and standard deviation, calculated for drinking
mean_drk = joblib.load('mean_drk.pkl')
std_drk = joblib.load('std_drk.pkl')

#model for drinking
rf_drk = joblib.load('model_drk.pkl')



#mean and standard deviation, calculated for smoking
mean_smk = joblib.load('mean_smk.pkl')
std_smk = joblib.load('std_smk.pkl')

#model for smoking
lr_smk = joblib.load('model_smk.pkl')








st.title("Smoking & Drinking habit Identification from Body Signals")


# Text input for 'sex'
sex = st.radio("Select sex:", ["Male", "Female"])

# Numeric inputs
age = st.slider("Age:", 1, 100, 25)
height = st.slider("Height (cm):", 100, 250, 170)
weight = st.slider("Weight (kg):", 30, 200, 70)
waistline = st.number_input("Waistline:", value=20.0)
# Other numeric inputs
sight_left = st.number_input("Sight Left:", value=20.0)
sight_right = st.number_input("Sight Right:", value=20.0)
hear_left = st.number_input("Hearing Left:", value=20.0)
hear_right = st.number_input("Hearing Right:", value=20.0)
SBP = st.number_input("Systolic Blood Pressure:", value=120.0)
DBP = st.number_input("Diastolic Blood Pressure:", value=80.0)
BLDS = st.number_input("Blood Sugar Level:", value=100.0)
tot_chole = st.number_input("Total Cholesterol:", value=200.0)
HDL_chole = st.number_input("HDL Cholesterol:", value=50.0)
LDL_chole = st.number_input("LDL Cholesterol:", value=100.0)
triglyceride = st.number_input("Triglyceride:", value=150.0)
hemoglobin = st.number_input("Hemoglobin:", value=12.0)
urine_protein = st.number_input("Urine Protein:", value=0.0)
serum_creatinine = st.number_input("Serum Creatinine:", value=1.0)
SGOT_AST = st.number_input("SGOT (AST):", value=30.0)
SGOT_ALT = st.number_input("SGPT (ALT):", value=30.0)
gamma_GTP = st.number_input("Gamma-GTP:", value=20.0)

smk_drk = st.radio("What do you want to Predict:", ["Smoking Rate", "Drinking Identification"])

# Submit button
if st.button("Submit"):

    if sex=="Male":
        sex=1
    elif sex=="Female":
        sex=0


    if smk_drk=="Drinking Identification":
        input_data = [
        sex,
        age,
        sight_left,
        sight_right,
        hear_left,
        hear_right,
        DBP,
        HDL_chole,
        hemoglobin,
        serum_creatinine,
        SGOT_ALT,
        gamma_GTP
    ]

        standardized_input = (np.array(input_data) - mean_drk) / std_drk

    # Make a prediction using the loaded model
        prediction = rf_drk.predict([standardized_input])

        if prediction==0:
            prediction="NO"
        if prediction==1:
            prediction="YES"

    # Display the prediction result
        st.write("Prediction Result: (Drinking) ", prediction)

    if smk_drk=="Smoking Rate":
        input_data = [
        sex,
        age,
        height,
        weight,
        waistline,
        hear_left,
        hear_right,
        SBP,
        HDL_chole,
        hemoglobin,
        serum_creatinine,
 
    ]

        standardized_input = (np.array(input_data) - mean_smk) / std_smk

    # Make a prediction using the loaded model
        prediction = lr_smk.predict([standardized_input])
        if prediction==1:
            text="Never Smoked"
        elif prediction==2:
            text="Used to Smoke"
        else:
            text="Still Smoking"


        text=str(int(prediction[0])) + "( "+text+" )"
    # Display the prediction result
        st.write("Prediction Result: (SMOKING) : ",text)

