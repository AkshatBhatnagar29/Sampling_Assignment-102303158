# Sampling_Assignment-102303158

Sampling Assignment – Credit Card Fraud Detection

## Objective
The goal of this assignment is to learn about the importance of **sampling techniques** when working with **imbalanced datasets** and to explore how different sampling methods impact the performance and accuracy of various machine learning models.


---

## Problem Statement
The credit card dataset provided is heavily imbalanced, which can greatly affect how well a model performs in real situations.
The task is to balance the dataset using different sampling methods, train several machine learning models on those balanced datasets, and examine how sampling affects model accuracy.

---

## Dataset
The dataset is available from this GitHub link:

https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv

- The target column is **Class**
- `0` → Normal transaction
- `1` → Fraudulent transaction

---

## Sampling Techniques Used
Five sampling techniques were applied:

1.
**Random Over Sampling**
2.
**Random Under Sampling**
3.
**SMOTE (Synthetic Minority Over-sampling Technique)**
4.
**SMOTEENN (SMOTE + Edited Nearest Neighbors)**
5.
**No Sampling (Original Dataset)**

---

## Machine Learning Models Used
Five machine learning models were trained on each sampled dataset:

- **M1** – Logistic Regression
- **M2** – Decision Tree Classifier
- **M3** – Random Forest Classifier
- **M4** – Support Vector Machine (SVM)
- **M5** – Naive Bayes

---

## Evaluation Metric
- **Accuracy (%)** was used to compare how well different models and sampling techniques performed.


---

## Accuracy Table (%)

| Model | RandomOver | RandomUnder | SMOTE | SMOTEENN | NoSampling |
|------|------------|------------|-------|----------|------------|
| M1 (Logistic Regression) | 93.14 | 25.00 | 93.14 | 93.52 | 99.35 |
| M2 (Decision Tree) | 99.02 | 75.00 | 97.71 | 97.95 | 96.77 |
| M3 (Random Forest) | 100.00 | 0.00 | 99.35 | 99.32 | 99.35 |
| M4 (SVM) | 96.08 | 0.00 | 96.73 | 99.66 | 99.35 |
| M5 (Naive Bayes) | 75.82 | 25.00 | 72.55 | 73.72 | 98.06 |

The accuracy results are generated automatically and saved as a CSV file called `results/accuracy_table.
csv`.

---

## Visual Analysis

### Model-wise Accuracy Comparison
This grouped bar chart compares the accuracy of each machine learning model across all sampling techniques.


<img width="1289" height="590" alt="image" src="https://github.com/user-attachments/assets/c0faa710-be6b-4ceb-b377-c5b328fb8b78" />

- This chart helps identify the best model–sampling combinations.


---

## Key Observations
- **Random Forest and SVM** perform well with oversampling methods.

- **Random Under Sampling** leads to very poor results because it removes important information from the majority class.

- **SMOTE and SMOTEENN** offer a good balance between accuracy and reliability.

- Strong models like Random Forest also show good results even without any sampling.

## How to Run the Code

### Install required libraries
```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn
