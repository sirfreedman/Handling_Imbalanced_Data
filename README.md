# 🏦 Bank Customer Churn Prediction: Imbalanced Data Strategies

This notebook demonstrates and evaluates multiple strategies for handling class imbalance using a deep artificial neural network (ANN) built with Keras. We address the minority churn prediction class (`churn = 1`) through various sampling and ensemble approaches.

## 📋 Project Overview
Customer churn is heavily imbalanced in this dataset. To improve the classification metrics (particularly minority class recall), we train and compare deep learning models using the following techniques:

1. **Baseline with Imbalanced Data**: Simple preprocessing with unbalanced data.
2. **Random Under-Sampling**: Reducing majority class samples to match the minority size.
3. **Random Over-Sampling**: Replicating minority class samples to match the majority size.
4. **SMOTE (Synthetic Minority Over-sampling Technique)**: Generating synthetic samples for the minority class.
5. **Ensemble Method**: Under-sampling the majority class into multiple balanced batches, training individual classifiers, and combining predictions via a majority voting scheme.

## 📊 Performance Summary
Below is a comparative breakdown of how the different strategies performed against the holdout test set:

| Technique | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Overall Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Under-Sampling** | 0.71 | 0.67 | 0.69 | 70% |            
| **Over-Sampling** | 0.78 | 0.71 | 0.74 | 76% |  
| **SMOTE** | **0.75** | **0.80** | **0.77** | **76%** |  
| **Ensemble (Voting)** | 0.75 | 0.57 | 0.65 | 69% | 

### 💡 Key Takeaway
**SMOTE** yielded the highest performance across minority-class Recall (**80%**) and F1-score (**0.77**), proving to be the most optimal strategy for this dataset.
