# Sampling_Assignment-102303158
Sampling Assignment – Credit Card Fraud Detection

Objective
The goal of this assignment is to explore how different sampling methods affect the performance of various machine learning models, especially when dealing with imbalanced data.


Problem Statement
Credit card fraud detection datasets are typically skewed, meaning there are far more non-fraudulent transactions than fraudulent ones.
This can cause machine learning models to perform poorly. In this assignment, we apply different sampling techniques, train multiple models, and compare their accuracy to understand how sampling affects model performance.

Dataset
Dataset source:
https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv

- Target column: Class
- 0 → Non-fraudulent transaction
- 1 → Fraudulent transaction

Data Preprocessing
- The dataset was first balanced using random under-sampling, reducing the majority class to match the minority class.

- Features were scaled using StandardScaler before model training.


Sampling Techniques Used
The following five sampling techniques were implemented:

1.
Simple Random Sampling – Samples are chosen randomly with equal likelihood.
2.
Bootstrap Sampling – Sampling with replacement, allowing for duplicate records.
3.
Cluster Sampling – The dataset is divided into clusters, and a subset of clusters is selected.
4.
Stratified Sampling – Class distribution is preserved during the train-test split.
5.
Systematic Sampling – Samples are selected at fixed intervals.

Machine Learning Models Used
Five machine learning models were trained using each sampling technique:

- M1 – Logistic Regression
- M2 – Decision Tree Classifier
- M3 – Random Forest Classifier
- M4 – Support Vector Machine (SVM)
- M5 – Naive Bayes

Evaluation Metric
- Accuracy (%) was used to evaluate model performance.


Accuracy Table (%)

Model | Simple Random | Bootstrap | Cluster | Stratified | Systematic
--- | --- | --- | --- | --- | ---
M1 (Logistic Regression) | 16.67 | 100.00 | 50.00 | 16.67 | 66.67
M2 (Decision Tree) | 83.33 | 100.00 | 100.00 | 83.33 | 66.67
M3 (Random Forest) | 66.67 | 100.00 | 75.00 | 50.00 | 33.33
M4 (SVM) | 16.67 | 100.00 | 0.00 | 16.67 | 0.00
M5 (Naive Bayes) | 66.67 | 100.00 | 50.00 | 66.67 | 33.33

<img width="1290" height="590" alt="image" src="https://github.com/user-attachments/assets/e5afba68-edfc-4cdc-b747-562a88bc3c77" />


Results Analysis
- Bootstrap Sampling consistently gives very high accuracy because it allows repeated samples, increasing similarity between training and test data.

- Cluster and Systematic Sampling use smaller subsets, which increases variance and causes accuracy to change more.

- After balancing, the dataset becomes much smaller, so small prediction changes have a big effect on accuracy.

- Tree-based models like Decision Trees and Random Forests perform better on smaller sampled datasets.

- Linear models like Logistic Regression and SVM are more sensitive to changes in training data.



Conclusion
This experiment shows that sampling techniques greatly influence model performance.
No single method works best for all models. Bootstrap sampling may overstate accuracy, while cluster and systematic sampling introduce more variation. Choosing the right sampling strategy is important, especially with small or imbalanced datasets.

Author
Akshat
