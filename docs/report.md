# Predicting smokers and drinkers using body signal data

- Author - Mani Teja Gunda
- Semester - Fall'23
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- GitHub - <a href="https://github.com/gunda18"> gunda18 </a>
- LinkedIn - <a href="https://www.linkedin.com/in/mani-teja-gunda-78a1bb137"> Mani Teja Gunda </a>
- PowerPoint Presentation - TBA
- YouTube Video - TBA 
    
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
- hear_left
- SBP
- hemoglobin
- age
- hear_right
- sex
- height
- HDL_chole
- waistline
- serum_creatinine
- weight

Drinking: 
- hear_left
- urine_protein
- sight_right
- age
- hear_right
- SGOT_ALT
- gamma_GTP
- DBP
- sex
- HDL_chole
- serum_creatinine

Common features for both smoking and drinking:
- DBP
- HDL_chole
- SBP
- SGOT_ALT
- age
- gamma_GTP
- hear_left
- hear_right
- height
- hemoglobin
- serum_creatinine
- sex
- sight_right
- urine_protein
- waistline
- weight


## Model Training 

- As data size is moderately large, I plan to use google colab/kaggle to train the models.
- Python packages to be used: scikit-learn, pandas, matplotlib, seaborn.
- Train/Test data: Split the data to create train (70%) & test (30%) splits.
- Feature Creation/transformation: Write code to transform categorical features into numerical features.
- Modeling: start with basic models like KNN, logistic regression, decision tree and move to more complex ones such as random forest and gradient boosting.
- Metrics & Evaluation:  As this is a classification task, Accuracy scores, f1-score and confusion matrices will be used for performance measurement.


## Application of the Trained Models

Develop a web app for people to interact with your trained models using Streamlit.
- TBA

## 7. Conclusion

- TBA

## 8. References 

- TBA
