# Predicting smokers and drinkers using body signal data

- Author - Mani Teja Gunda
- Semester - Fall'23
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- GitHub - <a href="https://github.com/gunda18"> gunda18 </a>
- LinkedIn - <a href="https://www.linkedin.com/in/mani-teja-gunda-78a1bb137"> Mani Teja Gunda </a>
- <a href="https://github.com/DATA-606-2023-FALL-TUESDAY/Gunda_ManiTeja/blob/main/docs/Presentation.pptx"> <img src="https://img.shields.io/badge/-PowerPoint Presentation Download-B7472A?logo=microsoftpowerpoint&style=flat" alt="icon | PPT"/> </a>
- <a href="https://youtu.be/Fb1bQOorQAw"> <img src="https://img.shields.io/badge/-YouTube Presentation-FF0000?logo=youtube&style=flat" alt="icon | YouTube"/></a> 
    
## Background
### Problem Statement

This project aims to predict individual’s smoking and drinking habits using their health metrics. The dataset is provided by the 
National Health Insurance Service in Korea and contains a wide range of health indicators, including height, weight, waistline, 
blood pressure readings, cholesterol levels, and liver enzyme readings. By analyzing this data, we aim to determine the impact 
of smoking/drinking on these health metrics of the individual.


### Potential Real-world applications

**Health Risk Assessment:**  By analyzing the correlation between health metrics and smoking/drinking habits, healthcare professionals can 
develop more precise risk assessment tools. These tools can predict the likelihood of certain health complications based on an individual's 
habits and physiological measurements. 

**Public Health Campaigns:** Governments and health organizations can utilize the findings to design personalized health interventions and 
public health campaigns. For instance, those who smoke and show specific detrimental health metrics can receive targeted advice and resources 
to quit smoking. 

**Insurance Premium Calculation:** Insurance companies can use this data to adjust health insurance premiums based on the risk associated with 
smoking and drinking habits. Those who smoke or drink might face higher premiums, incentivizing healthier lifestyle choices.

**Health tracking wearables:** Companies developing wearable health technologies, like smartwatches that monitor various health metrics, 
can integrate these models trained from these datasets to provide users with real-time feedback on the potential health impacts of their habits.


### Research Questions
- Is smoking and drinking really independent of each other? Do all drinkers smoke or Do all smokers drink?
- Is it true that the majority of drinkers/smokers are men? What percent of smokers or drinkers are women?
- What age groups do the majority of smokers/drinkers belong to?
- What are some key health indicators (features) that correlate with smoking and/or drinking habits?
- How do different machine learning techniques (linear vs. non-linear) compare in the identification of smoking & drinking habits based on their health parameters and which models can most accurately classify individuals?
- As per the best performing models, which features are most important for identifying smoking status or drinking status?


## Data 

- **Source:** https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset?resource=download
- **Size:** - 109.6 MB
- **Shape:** -
  - Rows - 991,346
  - Columns - 24 
- **Time period** - 2022
- **Each row describes** - A person
- **Data Dictionary**

  | Column Name     | Data Type         | Description                                                                                             | Potential Values                                 |
  |-----------------|-------------------|---------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
  | Sex             | Categorical (str) | The sex of the individual, categorized as either male or female.                                       | 'Male', 'Female'                                 |
  | Age             | Numerical (int)   | The age of the individual, categorized into 5-year age intervals.                                       | 35, 40, 45, 50, |
  | Height          | Numerical (int)   | The height of the individual, measured in 5cm increments.                                              | 165, 170, 175, 180                    |
  | Weight          | Numerical (int)   | The weight of the individual, measured in 5kg increments.                                              | 55, 60, 65, 70                      |
  | Waist           | Numerical (int) | The circumference of the individual's waist.                                                           | 89, 90, 91, 92                    |
  | sight_left      | Numerical (float) | The visual acuity of the individual's left eye. | 0.9, 1.0, 1.2, 1.5                      |
  | sight_right     | Numerical (int) | The visual acuity of the individual's right eye.     | 0.9, 1.0, 1.2, 1.5                      |
  | hear_left       | Categorical (int) | The hearing status in the left ear of the individual, with 1 representing normal hearing and 2 representing abnormal hearing. | 1, 2                      |
  | hear_right      | Categorical (int) | The hearing status in the right ear of the individual, using the same classification system as hear_left. | 1, 2                      |
  | SBP             | Numerical (int)   | The highest systolic blood pressure measured by the individual, measured in mmHg (millimeters of mercury). | 67, 80, 128, 136                         |
  | DBP             | Numerical (int)   | The diastolic blood pressure measured by the individual, measured in mmHg.                             | 38, 76, 105, 180                         |
  | BLDS            | Numerical (int)   | The individual's fasting blood glucose level, measured in mg/dL (milligrams per deciliter).             | 74, 141, 186, 63                         |
  | tot_chole       | Numerical (int)   | The concentration of total cholesterol in the individual's blood, measured in mg/dL.                    | 74, 141, 400, 500                         |
  | HDL_chole       | Numerical (int)   | The concentration of high-density lipoprotein (HDL) cholesterol in the individual's blood, measured in mg/dL. | 74, 141, 186, 63                       |
  | LDL_chole       | Numerical (int)   | The concentration of low-density lipoprotein (LDL) cholesterol in the individual's blood, measured in mg/dL. | 74, 141, 186, 63                       |
  | triglyceride    | Numerical (int)   | The concentration of triglycerides in the individual's blood, measured in mg/dL.                         | 74, 141, 186, 63                         |
  | hemoglobin      | Numerical (float) | The concentration of hemoglobin in the individual's blood, measured in g/dL (grams per deciliter).       | 15.8, 17.6, 14.5, 19.3                          |
  | urine_protein   | Numerical (int)   | The amount of protein in the individual's urine, where high levels can indicate health problems like heart failure and kidney issues. | 1, 2, 3, 4                |
  | serum_creatinine  | Numerical (float)   | The concentration of creatinine in the individual's serum (blood), measured in mg/dL. Creatinine is a waste product that can indicate kidney function. | 11.8, 7.6, 16.9, 8.2        |
  | SGOT_AST        | Numerical (int)   | The SGOT (Glutamate-oxaloacetate transaminase) - AST (Aspartate transaminase) value in IU/L (International Units per Liter), which measures liver, heart, and other organ performance. | 20, 21, 40, 39          |
  | SGOT_ALT        | Numerical (int)   | The SGOT (Glutamate-oxaloacetate transaminase) - ALT (Alanine transaminase) value in IU/L, which specifically measures liver performance. | 20, 21, 40, 39                |
  | gamma_GTP       | Numerical (int)   | The gamma-GTP (y-glutamyl transpeptidase) value in IU/L, which quantifies liver function in the bile duct. | 20, 21, 40, 39                |
  | SMK_STAT_TYPE_CD| Categorical (int) | The individual's smoking status, where 1 indicates they have never smoked, 2 indicates they used to smoke but quit, and 3 indicates they are currently smoking. | 1, 2, 3 |
  | DRK_YN          | Categorical (int) | A flag indicating whether the individual is a drinker (1 for yes) or not (0 for no).                    | Y, N                    |


- **Target Variable(s)** - SMK_STAT_TYPE_CD and DRK_YN (2 different models will be developed)
- The remaining columns are predictors


## Exploratory Data Analysis (EDA)

EDA is performed in order to understand the data, answer the research questions and get some insights

- df.shape : Shape of the data - 991346 rows and 24 columns
- df.dtypes : To check the data types of all columns
- df.duplicated() : There are 26 duplicate rows
- df.isna() : To check the null values. There are no null values in any column

### Distribution Histogram

This is a Histogram for all the columns of the dataset

<img width="600" alt="image" src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/histogram_all.png">

### Drinking State

<img width="600" alt="image" src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/drinking_count.png">

The data has almost equal number of drinkers and non-drinkers

### Smoking State

<img width="600" alt="image" src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/smoking_count.png">

Majority of the data shows non-smokers, followed by smokers and then those who quit smoking

### Is smoking and drinking really independent of each other? Do all drinkers smoke or Do all smokers drink?

<img width="600" alt="image" src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/smoking_vs_drinking.png">

From the above plot, we can see that:
- Majority of the non-drinkers are non-smokers
- While majority of drinkers are non smokers, the gap between each of the smoking state is less
- The number of drinkers who smoke are approximately three times the number of non-drinkers who smoke

### Is it true that the majority of drinkers/smokers are men? What percent of smokers or drinkers are women?

<img width="600" alt="image" src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/gender_vs_drinking.png">

<img width="600" alt="image" src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/gender_vs_smoking.png">

From the above plot, we can see that:

- Approximately 70% of the drinkers are male
- Non-drinking female is almost twice the number of non-drinking male
- Majority of the non-smokers are female
- Male population dominates in both the states of still smoking and quit smoking

### What age groups do the majority of smokers/drinkers belong to?¶

<img width="1000" alt="image" src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/age_drinking.png">

<img width="1000" alt="image" src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/age_smoking.png">

- Most of the smokers are between 35 and 50 years
- Most of the drinkers are between 30 and 50 years

### What are some key health indicators (features) that correlate with smoking and/or drinking habits?¶

To answer this question, we need to see the correlation matrix to see the inter relation between columns.

<img width="600" alt="image" src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/correlation.png">

Using "Recursive Feature Elemination" method to select the features for smoking and drinking seperately. These are the selected features:


Smoking:
- age
- sex
- HDL_chole
- hear_left
- hear_right
- serum_creatinine
- hemoglobin
- waistline
- SBP
- weight
- height

Drinking: 
- age
- sex
- HDL_chole
- hear_left
- hear_right
- serum_creatinine
- hemoglobin
- SGOT_ALT
- DBP
- gamma_GTP
- sight_right

Common features for both smoking and drinking:
- age
- sex
- HDL_chole
- hear_left
- hear_right
- serum_creatinine
- hemoglobin


## Model Training

- As the objective is to predict if a person is smoking or not and if a person is drinking or not, we are going to use classification models
- For model training, the data is split into two parts:
  - 70% of the data is used for training the models - 693,924 rows
  - 30% of the data is used for testing the models - 297,396 rows
- As two kinds of predictions are done here, models are trained seperately for drinking and smoking
- The models to be trained are:
  - K Nearest Neighbors (KNN)
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting

## Model Evaluation - Drinking

### KNN
- Accuracy: The accuracy score of the model is 0.669101. This means that approximately 66.91% of the predictions were correct.
- Precision: The precision score of the model is 0.666609. This indicates that about 66.66% of the positive predictions made by the model were correct.
- A recall of 0.675707 means that the model correctly identified approximately 67.57% of all actual positives.
- An F1 score of 0.671127 suggests a fairly balanced performance between precision and recall.
- A score of 0.669105 indicates the model's ability to distinguish between classes.

<p float="left">
  <img src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/d_knn_1.png" width="500" height="500" />
  <img src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/d_knn_2.png" width="500" height="500" /> 
</p>


### Logistic Regression
- Accuracy: The accuracy score of the model is 0.719384. This means that approximately 71.93% of the predictions were correct.
- Precision: The precision score of the model is 0.718072. This indicates that about 71.80% of the positive predictions made by the model were correct.
- A recall of 0.721783 means that the model correctly identified approximately 72.17% of all actual positives.
- An F1 score of 0.719923 suggests a fairly balanced performance between precision and recall but better than the KNN.
- A score of 0.719386 indicates the model's ability to distinguish between classes.

<p float="left">
  <img src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/d_lr_1.png" width="500" height="500" />
  <img src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/d_lr_2.png" width="500" height="500" /> 
</p>

### Decision Tree
- Accuracy: The accuracy score of the model is 0.635570. This means that approximately 63.55% of the predictions were correct.
- Precision: The precision score of the model is 0.635321. This indicates that about 63.53% of the positive predictions made by the model were correct.
- A recall of 0.635363 means that the model correctly identified approximately 63.53% of all actual positives.
- An F1 score of 0.635342 suggests a moderately fair balanced performance between precision and recall.
- A score of 0.635570 indicates the model's ability to distinguish between classes.

<p float="left">
  <img src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/d_dt_1.png" width="500" height="500" />
  <img src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/d_dt_2.png" width="500" height="500" /> 
</p>


### Random Forest
- Accuracy: The accuracy score of the model is 0.720733. This means that approximately 72.07% of the predictions were correct.
- Precision: The precision score of the model is 0.716712. This indicates that about 71.67% of the positive predictions made by the model were correct.
- A recall of 0.729401 means that the model correctly identified approximately 72.94% of all actual positives.
- An F1 score of 0.723001 suggests a fair balanced performance between precision and recall.
- A score of 0.720738 indicates the model's ability to distinguish between classes.

<p float="left">
  <img src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/d_rf_1.png" width="500" height="500" />
  <img src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/d_rf_2.png" width="500" height="500" /> 
</p>


### Gradient Boosting
- Accuracy: The accuracy score of the model is 0.712084. This means that approximately 71.20% of the predictions were correct.
- Precision: The precision score of the model is 0.696168. This indicates that about 69.61% of the positive predictions made by the model were correct.
- A recall of 0.751978 means that the model correctly identified approximately 75.19% of all actual positives.
- An F1 score of 0.722998 suggests a fair balanced performance between precision and recall.
- A score of 0.712110 indicates the model's ability to distinguish between classes.

<p float="left">
  <img src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/d_gb_1.png" width="500" height="500" />
  <img src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/d_gb_2.png" width="500" height="500" /> 
</p>


### Model Comparision 

<img width="600" alt="image" src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/d_compare.png">

- Random Forest has highest accuracy overall.
- Though Random Forest does not top in all other scores, it showed consistent performance overall.


## Model Evaluation - Smoking

### KNN
- Accuracy: The accuracy score of the model is 0.641004. This means that approximately 64.10% of the predictions were correct.
- Precision: The precision score of the model is 0.612603. This indicates that about 61.23% of the positive predictions made by the model were correct.
- A recall of 0.641004 means that the model correctly identified approximately 64.10% of all actual positives.
- An F1 score of 0.622709 suggests a fairly balanced performance between precision and recall.
- A score of 0.739050 indicates the model's ability to distinguish between classes.

<img src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/s_knn_1.png" width="500" height="500" />


### Logistic Regression
- Accuracy: The accuracy score of the model is 0.670345. This means that approximately 67.03% of the predictions were correct.
- Precision: The precision score of the model is 0.702312. This indicates that about 70.23% of the positive predictions made by the model were correct.
- A recall of 0.670345 means that the model correctly identified approximately 67.03% of all actual positives.
- An F1 score of 0.679611 suggests a fairly balanced performance between precision and recall.
- A score of 0.818277 indicates the model's ability to distinguish between classes.

<img src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/s_lr_1.png" width="500" height="500" />


### Decision Tree
- Accuracy: The accuracy score of the model is 0.605465. This means that approximately 60.54% of the predictions were correct.
- Precision: The precision score of the model is 0.608824. This indicates that about 60.88% of the positive predictions made by the model were correct.
- A recall of 0.605465 means that the model correctly identified approximately 60.54% of all actual positives.
- An F1 score of 0.607108 suggests a moderately fair balanced performance between precision and recall.
- A score of 0.635952 indicates the model's ability to distinguish between classes.

<img src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/s_dt_1.png" width="500" height="500" />


### Random Forest
- Accuracy: The accuracy score of the model is 0.668711. This means that approximately 66.87% of the predictions were correct.
- Precision: The precision score of the model is 0.670010. This indicates that about 67% of the positive predictions made by the model were correct.
- A recall of 0.668711 means that the model correctly identified approximately 66.87% of all actual positives.
- An F1 score of 0.679611 suggests a fair balanced performance between precision and recall.
- A score of 0.818277 indicates the model's ability to distinguish between classes.

<img src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/s_rf_1.png" width="500" height="500" />


### Gradient Boosting
- Accuracy: The accuracy score of the model is 0.634649. This means that approximately 63.46% of the predictions were correct.
- Precision: The precision score of the model is 0.680251. This indicates that about 68.02% of the positive predictions made by the model were correct.
- A recall of 0.634649 means that the model correctly identified approximately 63.46% of all actual positives.
- An F1 score of 0.540130 suggests a fair balanced performance between precision and recall.
- A score of 0.827955 indicates the model's ability to distinguish between classes.

<img src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/s_gb_1.png" width="500" height="500" />


### Model Comparision 

<img width="600" alt="image" src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/s_compare.png">

- Linear Regression has highest accuracy overall.
- Except for ROC scores, Linear Regression has topped in all the scores, the ROC score is pertty close to the scores of Random Forest and Gradient Boosting models


## Application of the Trained Models - Streamlit Web Application

- Developed a web application using streamlit.
- The streamlit web page contains several areas of input where the user is asked to enter the values like height, weight, eye sight, cholestrol etc.
- Once the input is given, the user should select if they want to predict smoking or drinking.
- When submitted, the result will be displayed below the submit button.

<p float="left">
  <img src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/streamlit_1.png" width="500" height="1300" />
  <img src="https://github.com/gunda18/UMBC-DATA606-FALL2023-TUESDAY/blob/main/ref_pics/streamlit_2.png" width="500" height="1300" /> 
</p>


## 7. Conclusion

### Summary of Project
In summary, this project embarked on the journey of predicting smoking and drinking habits based on health metrics. Leveraging a diverse dataset, we delved into the intricacies of the relationships between various health indicators and these habits. Through meticulous exploratory data analysis and model training, we aimed to provide actionable insights for various stakeholders.

### Potential Applications
The application of our predictive models spans across multiple domains. From healthcare professionals using more precise risk assessment tools to governments designing targeted public health campaigns, the implications are far-reaching. Insurance companies can benefit from adjusting premiums based on accurate risk assessments, and individuals can receive personalized feedback through health tracking wearables.

### Limitations
It's essential to acknowledge the limitations of our work. The predictions provided by the models are based on correlations observed in the dataset and may not imply causation. Furthermore, the accuracy of predictions relies heavily on the quality and representativeness of the dataset.

### Lessons Learned
Throughout the project, we learned valuable lessons in data preprocessing, model selection, and the importance of interpretability. The agile approach allowed us to adapt to challenges and refine our models continuously.


## 8. References 

- Kaggle datasets: Smoking and Drinking Dataset with body signal
  - **Source:** https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset?resource=download
- Streamlit documentation:
  - **Source:** https://docs.streamlit.io/
- Scikit-Learn documentation:
  - **Source:** https://scikit-learn.org/stable/index.html
- Liver Disease Prediction using Machine learning Classification Techniques by Ketan Gupta; Nasmin Jiwani; Neda Afreen; Divyarani D
  - **Source:** https://ieeexplore.ieee.org/document/9787574
