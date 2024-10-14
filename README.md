# Diabetes-Medication-and-Patient-Readmision-Prediction.
This project aims to analyze a large clinical database to explore historical patterns of diabetes care in patients admitted to a US hospital, predicting both the likelihood of hospital readmission and the necessity of receiving diabetes medication.


Ø  WHY THIS PROJECT?

Diabetes is a significant public health concern in the United States, affecting over 37 million people (approximately 11.3% of the population) as of 2022 [1], and contributing to numerous complications that greatly increase healthcare costs. The estimated cost of diagnosed diabetes in the U.S. in 2022 is $412.9 billion, including $306.6 billion in direct medical costs [2]. These statistics highlight the urgent need for effective management strategies. Hospitals frequently face challenges in managing the care of diabetic patients due to persistent shortages of medical resources, such as hospital beds, medications, and equipment. This project aims to :

Analyze a large clinical database to explore historical patterns of diabetes care in patients admitted to a US hospital
Predict whether the patient should be treated with diabetic medication or not.
Predict if the patient will be required to be readmitted to the hospital within 30 days, after 30 days, or never.

By addressing these issues, we seek to inform future directions that could significantly improve patients' medical conditions while optimizing resource utilization.


Ø  METHODS

Data

The dataset used is from the UCI Machine Learning Repository [3]. It contains 50 variables characterized as multivariate, with a mix of quantitative and qualitative. It also has more than 100,000 observations.

Analysis

We performed exploratory data analysis by presenting key variables and their frequency distributions and visualizing using Power BI.
Before initiating the mining process, we checked correlations between variables: using Pearson for quantitative-quantitative variables, Chi-square for qualitative-qualitative, and Cramér's V for quantitative-qualitative pairs. The correlations were visualized via a heatmap in Rstudio.
The variables "diabetesMed" and "readmitted" represent respectively Diabetes medications and Patient readmission 
We built and tested four predictive models through data mining: Random Forest, Gradient Boosting, and Decision Tree in Rstudio.
Using confusion matrices, we evaluated model performance and selected the most appropriate one for predicting both hospital readmissions and the necessity of diabetes medications.


Ø  RESULTS

Data cleaning

Single-value columns: The columns “Examide” and “Citoglipton” contained the same value for all entries in the dataset. Since such columns do not contribute to the analysis, they were removed.
Unique-value columns: “encounter_id” and “patient_number” had unique values for each record, offering no analytical significance. As a result, they were excluded from further analysis.
Handling Duplicates and Missing Values: First, we removed duplicate entries in the dataset to ensure data integrity. Additionally, non-informative characters such as “N/A,” “?”, “null,” “NA,” and other placeholders were replaced with NA to identify missing values more effectively. After identifying missing values, we observed that certain columns like “Weight,” “Medical_Specialty,” and “Payer_code” had almost complete missing data, making them unsuitable for imputation. These columns were therefore removed from the dataset.

For columns with missing values that had a relatively low percentage of missed entries "Race", "Diag_1", "Diag_2" and "Diag_3", we used imputation strategies. Quantitative variables were imputed using the mean value, while categorical variables were imputed with their mode (the most frequent value in the column). This method helps maintain the data's consistency without introducing bias from arbitrary replacements.


Data integration

In this step, we transformed qualitative nominal diagnosis data Diag_1", "Diag_2" and "Diag_3" into categorical codes corresponding to their relevant diagnosis groups based on the ICD-9 classification system [4].

Data transformation

Categorical Data Transformation: We transformed all the qualitative nominal variables, such as race, gender, age, and various medication-related variables, into factors using the mutate() function in R.
Quantitative Data Transformation: We performed min-max normalization on the quantitative variables to standardize the range of these features between 0 and 1. Normalization ensures that variables with larger ranges do not disproportionately influence the model, which can be crucial when combining multiple predictors in machine learning models. The following quantitative columns were normalized: time_in_hospital, num_lab_procedures, num_procedures, num_medications, number_outpatient, number_inpatient, number_diagnoses and number_emergency


Explanatory Data Analysis (EDA)

Patient repartition by Age and Gender.
The bar chart presents the distribution of patients by age class and gender, with categories for female, male, and unknown/invalid. The age groups span from 0 to 100 years, with the highest patient count observed in the [70-80] age class, where 14,000 females and 12,100 males are reported. Notably, females consistently outnumber males in the older age groups, particularly in the [70-80] and [80-90] ranges, with a significant female majority in both categories. In contrast, males show a slight majority in the [50-60] age class, though the difference remains minimal. The youngest group, aged 0 to 10, has the lowest patient numbers, with only 100 patients recorded for males and females.


Patient repartition by Race.
The pie chart illustrates the distribution of patients based on race, showing that the majority are Caucasian, representing 77.01% of the total population. African Americans account for 18.88%, making them the second-largest racial group. Hispanic patients represent a smaller portion, with 2% of the total, followed by Asian patients, who comprise 0.63%. The "Other" category includes 1.48% of the patients. 


Patient distribution by Diagnosis.
The bar chats provide a comprehensive overview of patient diagnoses using ICD-9 codes, showcasing the distribution across primary (diag_1), secondary (diag_2), and tertiary (diag_3) diagnostic categories. Circulatory issues consistently dominate across all three categories, affecting approximately 30,000 patients in each, highlighting the prevalence of cardiovascular problems both as primary conditions and comorbidities. Respiratory problems rank second in primary diagnoses but decrease in prominence in secondary and tertiary categories, suggesting they often drive initial hospital visits. Conversely, diabetes shows an interesting pattern, rising from fourth place in primary diagnoses to second in both secondary and tertiary categories, indicating its frequent role as a complicating factor or comorbidity. Digestive issues and injuries feature prominently in primary diagnoses but less so in subsequent categories, reflecting their nature as common reasons for seeking immediate care.


Patient Distribution based on Admission and Discharge types
The horizontal bar charts illustrate patient distribution based on admission and discharge types. Emergency admissions account for the largest proportion with 54,000 patients, while elective and urgent admissions are reported for 19,000 and 18,000, respectively. Regarding discharge types, the majority of patients, 74,000 in total, were discharged to home care. Meanwhile, 19,000 patients were discharged to skilled nursing, rehabilitation, or psychiatric care facilities. These patterns highlight the predominant reliance on emergency services for admissions and home care as the most frequent discharge outcome.



The bar charts  display key trends regarding diabetes medication and hospital readmission among patients. Of the total cohort, 78,000 patients (77%) were receiving diabetes medication, while 23,000 (23%) were not. Regarding hospital readmissions, 55,000 patients (55%) did not experience a readmission, 36,000 (36%) were readmitted more than 30 days post-discharge, and 11,000 (11%) were readmitted within 30 days. The high rate of readmissions, particularly in the >30-day group, suggests potential gaps in post-discharge care or disease management.



Correlation Analysis

Correlation analysis is a statistical technique that evaluates the strength and direction of the relationship between two variables. A strong correlation indicates that as one variable changes, the other variable tends to change predictably, while a weak correlation suggests that the variables do not exhibit a meaningful relationship. 

The diagonal elements, highlighted in yellow, represent the self-correlation of each variable and are expectedly 1, reflecting a perfect correlation.  Off-diagonal elements are either close to zero or exhibit weak negative correlations (dark blue shades), indicating limited relationships among variables.
This lack of significant correlation implies that the dataset is diverse, with many variables providing unique information. Such a characteristic is advantageous for subsequent predictive modeling, as it minimizes the risk of multicollinearity, allowing each variable to contribute independently to the analysis.


Data Splitting and Balancing 

In this analysis, we aimed to prepare the dataset for predictive modeling by splitting the data into training and testing sets and addressing class imbalance in the target variables. Below are the steps taken and the results obtained.

1. Data Splitting

To begin with, we split the dataset into training and testing sets. We employed a random sampling technique, ensuring reproducibility by setting a seed value. The training dataset comprised 80% of the data, while the remaining 20% was allocated to the testing set.

2. Data balancing

We then examined the distribution of the target variables, readmitted and diabetesMed, within the training dataset to identify imbalances. The initial assessment revealed a significant class imbalance in both variables which constitutes a risk of developing biased predictive models that could favor the majority class. To mitigate this issue, we applied the UpSampling, which generates synthetic samples for the minority and intermediate courses. 


Data Mining Methods

Classification is the most common task involved in data mining. To predict the readmissions in the hospital and diabetes medication administration, various classification tasks were performed. 


Ø  CONCLUSION

The models demonstrated strong performance in predicting diabetes medication usage in hospital settings, particularly with the Random Forest and Decision Tree models achieving high accuracy and precision.
However, predicting readmissions was more challenging, as most models exhibited low accuracy, indicating a need for further model optimization and exploration of additional data sources to enhance predictive performance.



REFERENCES

1. https://www.cdc.gov/diabetes/php/data-research/index.html

2. https://diabetesjournals.org/care/article/47/1/26/153797/Economic-Costs-of-Diabetes-in-the-U-S-in-2022

3. http://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+19992008

4. http://www.icd9data.com/2015/Volume1/default.htm
