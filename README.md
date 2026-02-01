# Sampling_Assignment-102303158
Sampling Assignment – Credit Card Fraud Detection

The goal of this assignment is to examine how different sampling methods affect the performance of various machine learning models when working with an imbalanced dataset.


The credit card dataset provided is highly imbalanced, which can lead to poor model performance.
The task involves applying different sampling techniques, training multiple machine learning models, and comparing their accuracy to find out which sampling strategy works best for each model.

Dataset Source:

https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv

- Target Column: **Class**
- `0` → Normal transaction
- `1` → Fraudulent transaction

Sampling Techniques Used

Five classical sampling techniques were implemented:

1.
**Simple Random Sampling** – Random selection where every data point has an equal chance of being selected.
2.
**Bootstrap Sampling** – Sampling with replacement.
3.
**Cluster Sampling** – Sampling entire groups (clusters) of data.
4.
**Stratified Sampling** – Preserves the class distribution during sampling.
5.
**Systematic Sampling** – Selecting samples at regular intervals.

Machine Learning Models Used

Five machine learning models were trained on each sampled dataset:

- **M1** – Logistic Regression
- **M2** – Decision Tree Classifier
- **M3** – Random Forest Classifier
- **M4** – Support Vector Machine (SVM)
- **M5** – Naive Bayes

Evaluation Metric

- **Accuracy (%)** was used as the evaluation metric to compare model performance.


Accuracy Table (%)

| Model | Simple Random | Bootstrap | Cluster | Stratified | Systematic |
|------------------------------|---------------|-----------|---------|------------|------------|
| M1 (Logistic Regression) | 100.00 | 98.06 | 100.00 | 98.39 | 100.00 |
| M2 (Decision Tree) | 98.71 | 98.71 | 100.00 | 96.77 | 100.00 |
| M3 (Random Forest) | 100.00 | 98.71 | 100.00 | 98.39 | 100.00 |
| M4 (SVM) | 100.00 | 98.06 | 100.00 | 98.39 | 100.00 |
| M5 (Naive Bayes) | 97.42 | 89.03 | 100.00 | 97.58 | 100.00 |

The table above is generated automatically and saved as `results/accuracy_table.
csv`.

Best Sampling Technique for Each Model

| Model | Best Sampling Technique | Best Accuracy (%) |
|------------------------------|-------------------------|-------------------|
| M1 – Logistic Regression | Simple Random | 100.00 |
| M2 – Decision Tree | Cluster | 100.00 |
| M3 – Random Forest | Simple Random | 100.00 |
| M4 – SVM | Simple Random | 100.00 |
| M5 – Naive Bayes | Cluster | 100.00 |

Visual Analysis

Graphs are created to visually compare performance and are saved in the `results/` folder.


- <img width="1290" height="590" alt="image" src="https://github.com/user-attachments/assets/281fbcc8-427d-4edb-a9be-8d8475d5f411" />


-<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/8e3009e4-ca59-4436-a56d-a25887ed2d52" />
 – Shows average accuracy per sampling method.

- <img width="933" height="590" alt="image" src="https://github.com/user-attachments/assets/c4418a73-a49a-43fa-ab2b-80c9720af761" />



Key Observations

- **Simple Random Sampling** performs well for Logistic Regression, Random Forest, and SVM.

- **Cluster Sampling** provides the best performance for Decision Tree and Naive Bayes models.

- **Bootstrap Sampling** generally performs slightly worse due to repeated samples.

- Several models achieved **100% accuracy**, which can happen when the dataset becomes highly separable after sampling.

- Accuracy alone may not be enough in real-world scenarios; additional metrics can offer better insight.


How to Run the Code

Install required libraries

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```ll numpy pandas scikit-learn imbalanced-learn matplotlib seaborn
