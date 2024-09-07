# BiasDetection-IntepretabilityAI
This repository contains several notebooks exploring bias detection techniques in artificial intelligence models and applying various interpretability algorithms to analyze the decisions made by the models.

## Notebook 1: Analysis of COMPASS Algorithm Predictions

### Use Case:
This notebook analyzes the COMPASS algorithm's predictions for the likelihood of recidivism among criminals, focusing on how demographic factors such as race, gender, and age category affect these predictions. The goal is to evaluate potential biases in the algorithm's predictions by comparing them with actual recidivism rates measured two years after the original offense.

### Objective:
The main objective is to assess the influence of racial, gender, and age factors on the COMPASS algorithm's risk prediction and to determine whether these factors contribute to biased predictions. This will be achieved through data exploration and visualization, with a focus on understanding discrepancies between predicted and actual recidivism rates.
The second objective of this notebook is to evaluate the performance of the COMPASS model in predicting recidivism across different demographic groups. This includes identifying potential biases by comparing the model's predictions with actual recidivism rates. The evaluation will be conducted using precision, recall, F1 score, and differences in mean values, segmented by race, gender, and age. Additionally, the impact of these biases will be assessed using confusion matrices to better understand discrepancies in the model's performance across different groups.

### Steps:

#### Preparation of the Working Environment:
Downloaded the dataset from a Google Drive repository using a public link.
Loaded the dataset into a DataFrame using Pandas.
Installed and imported necessary libraries for data analysis and visualization.

#### Initial Data Exploration:
Conducted exploratory data analysis (EDA) using a custom EDAModule.
Evaluated the dimensions of the DataFrame, which contains 40 variables and 18,316 entries.
Identified numerical and categorical variables.
Analyzed basic statistics of numerical columns, noting average age, prediction scores, and actual recidivism rates.
Handling Missing Values:
Identified columns with missing values and evaluated their impact on the dataset.
Found that some columns had a significant number of missing values, while others were nearly empty and deemed unusable for further analysis.
Detected inconsistencies, such as erroneous values in is_recid and decile_score.
Outlier Detection:
Used boxplots to identify potential outliers in numerical variables.
Noted that outliers could be due to various factors like extreme ages or long judicial processes.
Histogram Visualization:
Generated histograms to observe the distribution of numerical variables.
Observed that crimes are most frequent among younger individuals and that the COMPASS algorithm tends to predict extreme risk values.

#### Descriptive Analysis:
Conducted a descriptive analysis of the data, focusing on the correlation between personal characteristics and the COMPASS risk score.
Created a correlation matrix to assess relationships between variables and the decile_score.
Analysis by Race:
Examined how the COMPASS risk score varies by race.
Found that certain racial groups, particularly African American and Hispanic, had higher average risk scores compared to others.
Analysis by Gender:
Investigated the impact of gender on the COMPASS risk score.
Observed that men generally had a higher average risk score compared to women, with a greater range of values.
Analysis by Age Category:
Analyzed the relationship between age categories and risk scores.
Found a trend where the risk score decreases with increasing age categories, potentially indicating age-related bias.

#### Combined Analysis of Race, Gender, and Age Category:
Performed a combined analysis to explore interactions between race, gender, and age categories.
Identified notable trends such as higher recidivism predictions for certain racial and gender combinations in specific age groups.

#### Identification of Biases:
Compare the model’s predictions with actual recidivism values by grouping subjects based on their race, gender, and age.
Analyze how these demographic factors impact the differences between predicted and actual recidivism rates.

#### Evaluation Metrics:
Precision: Proportion of correctly predicted positive recidivism cases among all predicted positives.
Recall: Proportion of actual positive recidivism cases correctly identified by the model among all actual positives.
F1 Score: Harmonic mean of precision and recall.
Difference in Means: Compare the average predicted recidivism score with the actual recidivism rate.

#### Data Transformation for Metrics:
Transform the COMPASS risk scores from a range of 1-10 to binary values (0 or 1) for calculating precision, recall, and F1 score.
Adjust actual recidivism values by multiplying by 10 to align with the transformed prediction range for mean differences.

#### Performance by Demographic Groups:
Evaluate model performance by demographic group (race, gender, and age) using the defined metrics.
Identify any significant biases or discrepancies in performance.

#### Confusion Matrix Analysis:
Generate and analyze confusion matrices for each demographic group to assess the accuracy of the model’s predictions.
Identify any patterns or imbalances in the model's ability to predict recidivism correctly.

#### Recommendations:
Suggest improvements to reduce or eliminate identified biases in the COMPASS model.
Recommendations include improving data preprocessing, employing fairness-aware algorithms, continuous evaluation and re-training, and ensuring transparency in the model's development and deployment.

### Conclusion:
The analysis reveals that the COMPASS algorithm exhibits potential biases based on demographic factors. Specifically, racial minorities and younger individuals tend to receive higher risk scores, suggesting a possible bias in the algorithm. Gender also plays a role, with men generally receiving higher risk scores than women. Age shows a clear trend where older individuals are predicted to have a lower risk of recidivism.
The combined analysis of race, gender, and age categories highlights significant disparities in risk predictions, emphasizing the need for a closer examination of how these factors influence the algorithm's outputs. The findings suggest that the COMPASS algorithm may benefit from adjustments to mitigate demographic biases and improve fairness in its predictions.
The analysis of the COMPASS risk assessment model reveals significant variations in model performance across different demographic groups. The model tends to underpredict recidivism rates for certain groups, such as African-Americans and younger individuals, while showing varying performance across other groups. Confusion matrix analysis further highlights the model's struggles with accurately predicting recidivism, particularly in certain demographic segments.
To address these biases, it is recommended that the COMPASS model undergoes thorough data preprocessing to ensure representativeness, employs fairness-aware algorithms during training, and undergoes continuous evaluation and re-training with updated data. Transparency in the model’s development process will also be crucial for maintaining trust and accountability. Implementing these measures can help create a more equitable risk assessment tool, reducing bias and improving fairness across all demographic groups.

## Notebook 2: Understanding COMPASS Risk Assessment Model Using SHAP

### Use Case:
This notebook focuses on evaluating the COMPASS risk assessment model, which predicts the likelihood of recidivism among criminals. The aim is to assess the influence of demographic factors—such as race, gender, and age—on the model’s risk predictions. The analysis uses SHAP (SHapley Additive exPlanations) to interpret the model’s predictions and to identify potential biases based on these factors.

### Objective:
The objective is to examine the impact of various demographic factors on the COMPASS model’s predictions by employing SHAP values. 

### Steps:

#### Data Loading and Preprocessing:
Load the dataset from a Google Drive repository into a Pandas DataFrame.
Install and import necessary libraries.
Perform initial exploratory data analysis (EDA) to understand the dataset’s structure.
Clean and preprocess the data, including handling missing values, standardizing numerical variables, and encoding categorical variables.

#### Dataset Splitting:
Split the dataset into training and test sets using train_test_split from Scikit-learn, with 70% for training and 30% for testing.
Ensure stratification by race to maintain proportional representation of racial groups in both sets.
Set a random_state for reproducibility of the results.

#### Model Training:
Train an XGBoost model on the training set.
Evaluate the model using metrics such as Precision, Recall, F1-Score, and Confusion Matrix to assess its performance.

#### Generating SHAP Values:
Apply SHAP to the trained model to generate SHAP values, which quantify the contribution of each feature to the model’s predictions.
SHAP values are organized in a 2D array, where the first dimension represents instances, and the second represents features.

#### Analysis with SHAP:
Global Feature Importance:
Use SHAP’s summary plot to visualize the global importance of each feature.
Interpret the plot to understand which features have the most influence on the model’s predictions.
Individual Prediction Analysis:
Create waterfall plots for selected individuals from different racial groups.
Examine how individual features contribute to predictions and compare across different races.

### Conclusion:
Summarize findings from SHAP analysis, including feature importance and any observed biases.
Reflect on the overall performance of the model and its interpretability.
The analysis using SHAP values has provided a clear understanding of the COMPASS model’s behavior and feature importance. Key findings include:
The feature r_charge_degree has the highest impact on the model’s predictions.
Features related to recidivism history and crime charges are more influential than demographic factors like age, gender, and race.
The model appears to have lower sensitivity to demographic variables, suggesting minimal bias related to these factors.
The notebook has successfully demonstrated how SHAP can be used to interpret and understand machine learning models, providing insights into feature contributions and potential biases. This exercise enhances the ability to develop fairer and more transparent models in future work.
