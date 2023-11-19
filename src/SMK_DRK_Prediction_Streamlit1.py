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

df = pd.read_csv("processed_data.csv")

df_drk = df.copy()
df_smk = df.copy()

#train test split for drinking
X_drk, y_drk = df_drk.drop(columns=['height', 'weight', 'waistline', 'SBP', 'BLDS', 'tot_chole', 'LDL_chole',
                                  'triglyceride','urine_protein', 'SGOT_AST', 'SMK_stat_type_cd', 'DRK_YN']
                         , axis = 1), df_drk.DRK_YN

X_train_drk, X_test_drk, y_train_drk, y_test_drk = train_test_split(X_drk, y_drk, random_state = 42, train_size = 0.70)

# Scaling for drinking
scaler_drk = StandardScaler()
X_train_drk = scaler_drk.fit_transform(X_train_drk)
X_test_drk = scaler_drk.transform(X_test_drk)

#mean and standard deviation, calculated for drinking
mean_drk = scaler_drk.mean_
std_drk = scaler_drk.scale_

#model for drinking
rf_drk=RandomForestClassifier(random_state=0,n_jobs=-1)
rf_drk.fit(X_train_drk,y_train_drk)
y_pred_rf=rf_drk.predict(X_test_drk)





#train test split for smoking
X_smk, y_smk = df_smk.drop(columns=['sight_left', 'sight_right', 'DBP', 'BLDS', 'tot_chole', 'LDL_chole',
                                  'triglyceride', 'urine_protein', 'SGOT_AST', 'SGOT_ALT', 'gamma_GTP',
                                  'SMK_stat_type_cd', 'DRK_YN'], axis = 1), df_smk.SMK_stat_type_cd

X_train_smk, X_test_smk, y_train_smk, y_test_smk = train_test_split(X_smk, y_smk, random_state = 42, train_size = 0.70)

#scaling for smoking
scaler_smk = StandardScaler()
X_train_smk = scaler_smk.fit_transform(X_train_smk)
X_test_smk = scaler_smk.transform(X_test_smk)

#mean and standard deviation, calculated for smoking
mean_smk = scaler_smk.mean_
std_smk = scaler_smk.scale_

#model for smoking
lr_smk=LogisticRegression(max_iter=100, C=0.01, n_jobs=-1)
lr_smk.fit(X_train_smk,y_train_smk)
y_pred_lr_smk=lr_smk.predict(X_test_smk)
y_prob_lr_smk = lr_smk.predict_proba(X_test_smk)








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

