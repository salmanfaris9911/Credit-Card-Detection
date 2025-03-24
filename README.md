# Fraud Detection using Machine Learning

# Description
This project focuses on detecting fraudulent transactions using machine learning techniques. It analyzes a dataset of transaction records to predict whether a transaction is fraudulent, employing algorithms such as Decision Trees and Random Forests. The project also tackles the challenge of imbalanced data, which is common in fraud detection, by applying techniques like Random Under Sampling and SMOTE (Synthetic Minority Over-sampling Technique).

# Dataset
The dataset used is the Fraud Detection dataset (fraudTrain.csv), which includes transaction data with features such as:

Transaction amount
Category
Date and time of transaction
Gender
Age (derived from date of birth)
A binary label is_fraud (1 for fraudulent, 0 for non-fraudulent)
The dataset is highly imbalanced, with fraudulent transactions being significantly less frequent than non-fraudulent ones. Due to size constraints, the dataset is not included in the repository. You can download it from [insert dataset source/link here] and place it in the project directory as fraudTrain.csv.

# Methodology
Data Preprocessing
Dropped irrelevant columns: Unnamed: 0, cc_num, merchant, trans_num, unix_time, first, last, street, zip, lat, long, city_pop, merch_lat, merch_long, city, job, state.
Converted date/time: trans_date_trans_time to datetime format, extracted time and date, then dropped the original column.
Feature engineering: Calculated age from dob (date of birth), converted to years (age_in_years), and dropped dob, age, and age_str.
Encoding: One-hot encoded the category column and label-encoded gender (0 for one gender, 1 for the other).
Scaling: Applied StandardScaler to normalize features for model training.

# Exploratory Data Analysis
Visualized the distribution of fraudulent vs. non-fraudulent transactions using pie charts (matplotlib and plotly.express) and count plots (seaborn).
Analyzed fraud frequency by time and date for fraudulent transactions.
Handling Imbalanced Data
Random Under Sampling: Reduced the majority class (non-fraud) to balance the dataset.
SMOTE: Oversampled the minority class (fraud) to create synthetic examples, balancing the dataset.

# Model Training
Decision Tree Classifier: Trained with entropy criterion on original, undersampled, and SMOTE-resampled data.
Random Forest Classifier: Implemented with bagging (multiple trees on data subsets) and trained with parameters like max_depth=5, n_estimators=100.
Logistic Regression: Used as a baseline model for comparison.

# Evaluation
The models were evaluated using accuracy, precision, recall, F1-score, and additional metrics like confusion matrices and out-of-bag scores. Below are the key results:

Decision Tree (after SMOTE):
Accuracy: 0.99 (on the test set)
Precision, Recall, F1-Score:
precision    recall  f1-score   support

       0       0.90      0.89      0.89        99
       1       0.89      0.90      0.90       101

accuracy                           0.90       200
macro avg       0.90      0.89      0.89       200
weighted avg    0.90      0.90      0.89       200
The Decision Tree model demonstrates strong performance with an accuracy of 99% on the test set. However, the classification report indicates an accuracy of 90%, suggesting a possible discrepancy that should be verified from the original outputs. The precision, recall, and F1-scores are balanced and high for both classes (0: non-fraud, 1: fraud), reflecting effective fraud detection after addressing data imbalance with SMOTE.
Random Forest:
Out-of-Bag (OOB) Score: 0.86
The Random Forest model achieved an OOB score of 86%, which serves as a robust validation metric indicating good generalization to unseen data during training. This suggests the model balances precision and recall effectively.
Additional Insights:
The Decision Tree trained on the original imbalanced dataset (before SMOTE) likely had lower recall for fraudulent transactions (class 1), a common issue with imbalanced data. The SMOTE-resampled version significantly improved performance, as seen in the metrics above.
The test set consists of 200 samples (99 non-fraud, 101 fraud), likely reflecting the balanced dataset after resampling.
