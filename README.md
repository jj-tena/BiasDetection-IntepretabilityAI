# BiasDetection-IntepretabilityAI
This repository contains several notebooks exploring bias detection techniques in artificial intelligence models and applying various interpretability algorithms to analyze the decisions made by the models.

## Notebook X: Analysis of COMPASS Algorithm Predictions

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
