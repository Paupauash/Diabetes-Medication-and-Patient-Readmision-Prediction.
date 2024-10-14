## Project Title: Predicting Diabetes medication and patient readmission based on personal profile and clinical characteristics
##Let's upload the data set
Data <- read.csv("C:/Users/Hp/Desktop/Nouveau dossier/healthcare data/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv")


#Exploring the data 
summary(Data)
nrow(Data) #Finding the numbers of row
ncol(Data) # Finding the numbers or columns

##Data cleaning
##Checking if there are columns with the same values for all the entry
library(dplyr)
same_value_columns <- sapply(Data, function(col) length(unique(col)) == 1)

## The columns examide AND citoglipton have been identified, so we will remove them 
CData <- Data   %>% 
  select(-all_of(c("examide","citoglipton")))

#Checking the columns that have unique distinct values and removing them since they won't 
#have any effect in the modeling process
unique_columns <- sapply(CData, function(col) length(unique(col)) == nrow(CData))

#Removing the columns "encounter_id" and "patient_nbr"
CData <- CData  %>%
  select(-all_of(c("encounter_id", "patient_nbr")))

#Removing duplicates 
CData <- CData   %>%   
distinct()


#Replacing any of the following character "N/A", "?", " ","null", "NA", "UNKNOWN", "-",  "*" with NA
CData <- CData %>% 
  mutate(across(where(is.character), 
                ~ ifelse(. %in% c("", "?", "N/A", "null", "NULL", "NA", "UNKNOWN", "-", "*"), NA, .))) %>% 
  mutate(across(where(is.numeric), 
                ~ suppressWarnings(as.numeric(.))))

#Let's check if there are  missing  "NA" Values
colSums(is.na(CData)) 

#Since there are  missing values, we will try to handle them 
#Columns such as “Weight”, “Medical_Speciality”, “Payer_code” have almost 100 percent missing values 
#so we will remove these columns since we can not replace them

CData <-CData   %>% 
 select(-all_of(c("weight","medical_specialty", "payer_code")))

#There is also  missing values in other column but,there are low. It's preferable not to remove them. 
#A way to proceed is to replace them by the mean value if it's a quantitative variable and the mode 
#if it's a categorical variable.

CData <- CData %>%
mutate(race = if_else(is.na(race), 
                          { mode_value2 <- names(sort(table(race), decreasing = TRUE))[1] 
                          mode_value2 }, 
                          race),
         diag_1 = if_else(is.na(diag_1), 
                        { mode_value4 <- names(sort(table(diag_1), decreasing = TRUE))[1] 
                        mode_value4 }, 
                        diag_1),
         diag_2 = if_else(is.na(diag_2), 
                          { mode_value5 <- names(sort(table(diag_2), decreasing = TRUE))[1] 
                          mode_value5 }, 
                          diag_2),
         diag_3 = if_else(is.na(diag_3), 
                          { mode_value6 <- names(sort(table(diag_3), decreasing = TRUE))[1] 
                          mode_value6 }, 
                          diag_3))

#Let's verify if the missing NA values are gone.
colSums(is.na(CData)) 


##Qualitative Data transformation
#Let's transform diag_1 diag_2 and diag_3 into their corresponding diagnosis group codes (refer to http://www.icd9data.com/2015/Volume1/default.htm)

# Step 2: Define a function to classify diag codes based on the ranges
classify_icd9 <- function(diag_code) {
  
  if (grepl("^V", diag_code)) { 
    
  return("Others" )}
  
  else if (grepl("^E", diag_code)) {return("Others")} 
  
  else { diag_num <- as.numeric(diag_code) # For purely numeric codes, extract the numeric portion
    
    if (is.na(diag_num)) {
      return("UNKNOWN")}
  
   else if (diag_num >= 390 & diag_num <= 459 | diag_num == 785) {
    return("Circulatory")
  } else if (diag_num >= 460 & diag_num <= 519 | diag_num == 786) {
    return("Respiratory")
  } else if (diag_num >= 520 & diag_num <= 579 | diag_num == 787) {
    return("Digestive")
  } else if (grepl("^250", diag_code)) {
    return("Diabetes")
  } else if (diag_num >= 800 & diag_num <= 999) {
    return("Injury")
  } else if (diag_num >= 710 & diag_num <= 739) {
    return("Musculoskeletal ")
  } else if (diag_num >= 580 & diag_num <= 629 | diag_num == 788) {
    return("Genitourinary")
  } else if (diag_num >= 140 & diag_num <= 239) {
    return("Neoplasms")
  } else if (diag_num == 780 | diag_num == 781 | diag_num == 784 | (diag_num >= 790 & diag_num <= 799)) {
    return("Others")
  } else if (diag_num >= 240 & diag_num <= 279 & !grepl("^250", diag_code)) {
    return("Endocrine, nutritional, and metabolic diseases without diabetes")
  } else if (diag_num >= 680 & diag_num <= 709 | diag_num == 782) {
    return("Skin")
  } else if (diag_num >= 1 & diag_num <= 139) {
    return("Infectious")
  } else if (diag_num >= 290 & diag_num <= 319) {
    return("Mental")
  } else if (grepl("^(E|V)", diag_code)) {
    return("External")
  } else if (diag_num >= 280 & diag_num <= 289) {
    return("Blood")
  } else if (diag_num >= 320 & diag_num <= 359) {
    return("Nervous")
  } else if (diag_num >= 630 & diag_num <= 679) {
    return("Pregnancy")
  } else if (diag_num >= 360 & diag_num <= 389) {
    return("Sense Organs")
  } else if (diag_num >= 740 & diag_num <= 759) {
    return("Congenital")
  } else {
    return("Other")
  }
}
}


# Applying the classification function to the diag column
CData$diag_1 <- sapply(CData$diag_1, classify_icd9)
CData$diag_2 <- sapply(CData$diag_2, classify_icd9)
CData$diag_3 <- sapply(CData$diag_3, classify_icd9)

write.csv(CData, "C:/Users/Hp/Desktop/Nouveau dossier/healthcare data/diabetes+130-us+hospitals+for+years+1999-2008/Cleaned_Data.csv")

##Some Explanatory statistics
#Race repartition

RepRace <- table(CData$race)%>%
  as.data.frame()%>%
  mutate(Percentage=(Freq/sum(Freq))*100)%>%
  arrange(desc(Percentage))

#Gender repartition
RepGender <- table(CData$gender)%>%
  as.data.frame()%>%
  mutate(Percentage=(Freq/sum(Freq))*100)%>%
  arrange(desc(Percentage))        


# Readmission repartition
Rep_adm <- table(CData$readmitted)%>%
  as.data.frame()%>%
  mutate(Percentage=(Freq/sum(Freq))*100)%>%
  arrange(desc(Percentage))

# Age repartition
Rep_age <- table(CData$age)%>%
  as.data.frame()%>%
  mutate(Percentage=(Freq/sum(Freq))*100)%>%
  arrange(desc(Percentage))

#Admission_type_repartition
Rep_adm_type <-table(CData$admission_type_id)%>%
  as.data.frame()%>%
  mutate(Percentage=(Freq/sum(Freq))*100)%>%
  arrange(desc(Percentage))

#Average lenght_of stay in hospital
Avg_stay <-  mean(CData$time_in_hospital)


#Let's transform categorical variables from character to factor
CData <- CData%>%
 mutate(race=as.factor(race),
     gender=as.factor(gender),
    age=as.factor(age), admission_type_id=as.factor(admission_type_id),
  discharge_disposition_id=as.factor(discharge_disposition_id), 
  admission_source_id=as.factor(admission_source_id),
  diag_1=as.factor(diag_1),   diag_2=as.factor(diag_2),   diag_3=as.factor(diag_3), readmitted=as.factor(readmitted),
  max_glu_serum = as.factor(max_glu_serum), A1Cresult=as.factor(A1Cresult), metformin = as.factor(metformin),
  repaglinide=as.factor(repaglinide), nateglinide=as.factor(repaglinide), chlorpropamide =as.factor(chlorpropamide),
  glimepiride=as.factor(glimepiride), acetohexamide =as.factor(acetohexamide), glipizide=as.factor(glipizide), 
  glyburide=as.factor(glyburide), tolbutamide =as.factor(tolbutamide), pioglitazone=as.factor(pioglitazone),  
  rosiglitazone =as.factor(rosiglitazone), acarbose =as.factor(acarbose), miglitol=as.factor(miglitol), 
  troglitazone=as.factor(troglitazone), tolazamide=as.factor(tolazamide),  insulin=as.factor(insulin), 
  glyburide.metformin=as.factor( glyburide.metformin),  glipizide.metformin=as.factor(glipizide.metformin), 
  glimepiride.pioglitazone =as.factor(glimepiride.pioglitazone),  metformin.rosiglitazone=as.factor(metformin.rosiglitazone), 
  metformin.pioglitazone=as.factor(metformin.pioglitazone), change=as.factor(change), diabetesMed=as.factor(diabetesMed) )


## Before predicting diabetes medication, lets' transform the quantitative data using min-max transformation
Quantitative_columns <- c(CData$time_in_hospital, CData$num_lab_procedures, 
                          CData$num_procedures, CData$num_medications, CData$number_outpatient,
                          CData$number_inpatient, CData$number_diagnoses, CData$number_emergency) #Putting every quantitative columns into one

#Creating the min max normalization
min_max_normalization <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

# Applying the min max normalization to the quantitative columns
CData <- CData %>%
  mutate(across(c(7, 8, 9, 10, 11, 12, 13, 17), min_max_normalization)) 


##Let's check the correlation between the variables and remove correlated variable to avoid redundancy
# Load required libraries
library(ggplot2)
library(corrplot)


#Since we have two types of variables, we will proceed step by step

# Function to calculate Cramer's V that will compute the correlation between two categorical variables
cramers_v <- function(x, y) {
  confusion_matrix <- table(x, y)
  
  # Check for any zero frequencies
  if (any(confusion_matrix == 0)) {
    return(0)  # If there are zero frequencies, return 0 instead of NA
  }
  
  # Perform chi-square test and calculate Cramér's V
  chi2_test <- suppressWarnings(chisq.test(confusion_matrix))
  chi2 <- chi2_test$statistic
  
  # Calculate Cramér's V
  n <- sum(confusion_matrix)
  cramers_v_value <- sqrt(chi2 / (n * (min(dim(confusion_matrix)) - 1)))
  
  return(as.numeric(cramers_v_value))
}

# Function to calculate the correlation ratio between categorical and quantitative variables
correlation_ratio <- function(cat_var, num_var) {
  groups <- unique(cat_var)
  return(mean(sapply(groups, function(g) var(num_var[cat_var == g], na.rm = TRUE)), na.rm = TRUE))
}


# Function to create a correlation matrix that will identify the nature between variables and apply the appropriate method
create_correlation_matrix <- function(df) {
  variables <- colnames(df)
  corr_matrix <- matrix(NA, nrow = length(variables), ncol = length(variables))
  rownames(corr_matrix) <- colnames(corr_matrix) <- variables
  
  for (i in seq_along(variables)) {
    for (j in seq_along(variables)) {
      if (i == j) {
        corr_matrix[i, j] <- 1  # Perfect correlation
      } else if (is.numeric(df[[variables[i]]]) && is.numeric(df[[variables[j]]])) {
        # Pearson correlation for numerical vs numerical
        corr_matrix[i, j] <- cor(df[[variables[i]]], df[[variables[j]]], use = "complete.obs", method = "pearson")
      } else if (is.factor(df[[variables[i]]]) && is.factor(df[[variables[j]]])) {
        # Cramér's V for categorical vs categorical
        corr_matrix[i, j] <- cramers_v(df[[variables[i]]], df[[variables[j]]])
      } else if (is.numeric(df[[variables[i]]]) && is.factor(df[[variables[j]]])) {
        # Correlation ratio for numerical vs categorical
        corr_matrix[i, j] <- correlation_ratio(df[[variables[j]]], df[[variables[i]]])
      } else if (is.factor(df[[variables[i]]]) && is.numeric(df[[variables[j]]])) {
        # Correlation ratio for categorical vs numerical
        corr_matrix[i, j] <- correlation_ratio(df[[variables[i]]], df[[variables[j]]])
      }
    }
  }
  
  return(corr_matrix)
}

library(ggplot2)
library(reshape2)
library(farver)
# Creating correlation matrix
corr_matrix <- create_correlation_matrix(CData)

# Converting  correlation matrix to a long format
corr_matrix_melted <- melt(corr_matrix, na.rm = FALSE)  # Set na.rm = FALSE to keep NA values

# Plotting the correlation matrix heatmap
ggplot(corr_matrix_melted, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "black", mid = "blue", high = "yellow", na.value = "grey0", 
                       midpoint = 0, limit = c(-1, 1), guide = "colorbar") +  # Customize colors and NA value color
theme_minimal() +
labs(title = "Correlation Matrix Heatmap", x = "Variables", y = "Variables") +
theme(axis.text.x = element_text(angle = 45, hjust = 1))



#Let's split the data into training and testing test
set.seed(123) # Ensure reproducibility
train_index <-  sample(1:nrow(CData), 0.8*nrow(CData))  # 80% training data set
train_data <-  CData[train_index,] 
test_data <- CData[- train_index,]



##Let's check the modality distribution of both target variables "readmitted" and "diabetemed" in the training dataset
Dist_readm <- table(train_data$readmitted)%>%
  as.data.frame()%>%
  mutate(Percentage=(Freq/sum(Freq))*100)%>%
  arrange(desc(Percentage)) #Distributed in 54.03% for less than 30, 34.78% for sup 30 and 11.19% for No. So very imbalanced

Dist_diabmed <- table(train_data$diabetesMed)%>%
  as.data.frame()%>%
  mutate(Percentage=(Freq/sum(Freq))*100)%>%
  arrange(desc(Percentage)) #Distributed in 76.96% for Yes and 23.03% for No. Very imbalanced.




##Let's balance the training data set using the upsample  method
#Balancing the training data to predict readmission
library(caret) #Loading package to perform tup sample

set.seed(123)
smote_data_train1 <- upSample(train_data[, -which(names(train_data) == "readmitted")], train_data$readmitted)
summary(smote_data_train1)


#Let's check the new distribution of "Readmitted"
D1<- table(smote_data_train1$Class) %>%
as.data.frame() %>%
mutate(Percentage=(Freq/sum(Freq))*100) %>%
arrange(desc(Percentage)) #Now the Data is more balanced. 33.14% for less than 30, 28.42% for sup 30 and 27.43.% for No. 


#Balancing the training data to predict Diabetes medication 

set.seed(123)
smote_data_train2 <- upSample(train_data[, -which(names(train_data) == "diabetesMed")], train_data$diabetesMed)
summary(smote_data_train2)

#Let's check the new distribution of "DiabMed"
D2<- table(smote_data_train2$Class) %>%
  as.data.frame()%>%
  mutate(Percentage=(Freq/sum(Freq))*100)%>%
  arrange(desc(Percentage)) # Now the Data is more balanced. 52.68% for Yes, 47.31% for No. 


##DATA MINING 
##RANDOM FOREST
##Performing random Forest  to predict readmission
library(randomForest)
readm_model <- randomForest(as.factor(smote_data_train1$Class) ~.,
                          data=smote_data_train1, ntree=100)

#Making readmission prediction
readm_RF_predicted <-predict(readm_model, test_data)
summary(readm_RF_predicted)

#Let's check the random forest model accuracy to predict readmission
performance_metrics <- function(actual, predicted) {
  # Create confusion matrix
  cm <- table(Predicted = predicted, Actual = actual)
  
  # Extract values from confusion matrix
  TP <- cm[2, 2]  # True Positives
  TN <- cm[1, 1]  # True Negatives
  FP <- cm[2, 1]  # False Positives
  FN <- cm[1, 2]  # False Negatives
  
  # Calculate metrics
  accuracy <- (TP + TN) / sum(cm) * 100
  precision <- TP / (TP + FP) * 100
  recall <- TP / (TP + FN) * 100
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  # Return results
  return(list(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1_Score = f1_score
  ))
}



# Calculate performance metrics
performance1_1 <- performance_metrics(test_data$readmitted, readm_RF_predicted) #accuracy 17.25 ; precision 17.25

##Performing Random Forest to predict diabetes Medication
diabmed_model <- randomForest(as.factor(smote_data_train2$Class) ~ . ,
                            data=smote_data_train2, ntree=200)

#Making diabetesMed prediction
diabmed_RF_predicted <-predict(diabmed_model, test_data)
summary(diabmed_RF_predicted)

#Let's check the random forest model accuracy to predict diabetes med
performance2_1 <- performance_metrics(test_data$diabetesMed, diabmed_RF_predicted) #accuracy 99.84 ; precision 100



##GRADIENT BOOSTING
#Loading the required packages
library(xgboost)

#Readmissions

set.seed(123)  # For reproducibility
GB_read_model <- xgboost(data = as.matrix(data.frame(lapply(smote_data_train1[, -which(names(smote_data_train1) == "Class")], as.numeric))), 
                         label = as.numeric(smote_data_train1$Class) - 1,  # Because Classes should be 0, 1, 2 for xgboost
                         nrounds = 100, 
                         objective = "multi:softmax", 
                         num_class = 3)  # Specify number of classes

GB_predict_1 <- predict(GB_read_model, as.matrix(data.frame(lapply(test_data[, -which(names(test_data) == "readmitted")], as.numeric))))

performanceGB_1 <- performance_metrics(test_data$readmitted, GB_predict_1)# accuracy=19.19

#DiabetesMed

set.seed(123)  # For reproducibility
GB_diab_model <- xgboost(data = as.matrix(data.frame(lapply(smote_data_train2[, -which(names(smote_data_train2) == "Class")], as.numeric))), 
                         label = as.numeric(smote_data_train2$Class) - 1,  # Because Classes should be 0, 1, 2 for xgboost
                         nrounds = 100, 
                         objective = "multi:softmax", 
                         num_class = 2)  # Specify number of classes

GB_predict_2 <- predict(GB_diab_model, as.matrix(data.frame(lapply(test_data[, -which(names(test_data) == "diabetesMed")], as.numeric))))

performanceGB_2 <- performance_metrics(test_data$diabetesMed, GB_predict_2)# #accuracy 90.84 ; precision 100


##SUPPORT VENDOR MACHINE

#library(e1071)
#Readmissions

#svm_model1 <- svm(Class ~ ., data = smote_data_train1, kernel = "linear", cost = 1)

#svm_prediction1 <- predict(svm_model1, test_data)

#performance_SVM_1 <- performance_metrics(test_data$readmission, svm_prediction1)# accuracy=99.84

#DiabetesMed

#svm_model2 <- svm(Class ~ ., data = smote_data_train2, kernel = "linear", cost = 1)

#svm_prediction2 <- predict(svm_model2, test_data)

#performance_SVM_2 <- performance_metrics(test_data$diabetesMed, svm_prediction2)# accuracy=



##DECISION TREE
library(rpart)
#readmitted
tree_model1 <- rpart(Class ~ ., data = smote_data_train1, method = "class")

tree_prediction_1 <- predict(tree_model1, test_data, type = "class")

performance_DT_1 <- performance_metrics(test_data$readmitted, tree_prediction_1)# #accuracy 7.20 ; 

#DiabetesMed
tree_model2 <- rpart(Class ~ ., data = smote_data_train2, method = "class")

tree_prediction_2 <- predict(tree_model2, test_data, type = "class")

performance_DT_2 <- performance_metrics(test_data$diabetesMed, tree_prediction_2)#accuracy 99.15 ; precision 100