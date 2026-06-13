# Personal Health Mention (PHM) Tweet Classification — LSTM vs. Bi-LSTM

A deep learning project that classifies tweets as **Personal Health Mentions (PHM)** or **non-PHM**, comparing the performance of an **LSTM** and a **Bidirectional LSTM (Bi-LSTM)** model.


## Overview

Social media posts often contain mentions of personal health experiences (symptoms, medication, treatments) that are valuable for public health monitoring. This project builds and evaluates two recurrent neural network architectures to automatically detect whether a tweet describes a **personal health experience (PHM = 1)** or is **unrelated (Non-PHM = 0)**.

## Dataset

- **Training set:** 9,991 tweets (7,091 Non-PHM / 2,900 PHM)
- **Test set:** 3,331 tweets (2,364 Non-PHM / 967 PHM)
- Class imbalance ratio (Non-PHM : PHM) ≈ **2.45×**, addressed using computed class weights during training.

## Methodology

1. **Text Preprocessing**
   - Lowercasing, URL and `@mention` removal, hashtag normalisation
   - Custom stopword list preserving pronouns and negations (important for health-context sentiment)
   - Removal of non-alphabetic characters and whitespace normalisation

2. **Text Length Analysis**
   - Token length distribution analysis to determine optimal sequence length (95th percentile ≈ 18 tokens)

3. **Tokenisation & Padding**
   - Vocabulary size: 10,000
   - Max sequence length: 30
   - Embedding dimension: 64

4. **Class Imbalance Handling**
   - Balanced class weights: Non-PHM = 0.704, PHM = 1.723

5. **Model Architectures**
   - **LSTM:** Embedding → SpatialDropout1D → LSTM(64) → Dense(32, ReLU, L2) → Dropout → Dense(1, Sigmoid)
   - **Bi-LSTM:** Same architecture with a Bidirectional LSTM layer

6. **Training**
   - Optimizer: Adam (lr = 1e-3)
   - Loss: Binary Cross-Entropy
   - Callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`
   - Up to 20 epochs, batch size 64

7. **Optimal Threshold Selection**
   - Decision thresholds tuned on the validation set by sweeping 0.20–0.80 and maximising macro-F1

8. **Evaluation**
   - Accuracy, Precision, Recall, Weighted/Macro F1, ROC-AUC
   - Confusion matrices, ROC and Precision-Recall curves
   - 5-fold stratified cross-validation for robust performance estimates

## Results

### Test Set Performance (optimised thresholds)

| Metric      | LSTM (t=0.45) | Bi-LSTM (t=0.49) |
|-------------|:-------------:|:----------------:|
| Accuracy    | 0.8343        | 0.8235           |
| Precision   | 0.7155        | 0.6773           |
| Recall      | 0.7125        | 0.7487           |
| Weighted F1 | 0.8342        | 0.8260           |
| Macro F1    | 0.7987        | 0.7920           |
| ROC-AUC     | 0.8917        | 0.8953           |

### 5-Fold Cross-Validation (mean ± std)

| Metric      | LSTM            | Bi-LSTM          |
|-------------|:----------------:|:----------------:|
| Accuracy    | 0.8077 ± 0.0093 | 0.8102 ± 0.0081 |
| Macro F1    | 0.7773 ± 0.0072 | 0.7828 ± 0.0059 |
| Weighted F1 | 0.8117 ± 0.0073 | 0.8151 ± 0.0067 |
| ROC-AUC     | 0.8748 ± 0.0069 | 0.8815 ± 0.0048 |

### Key Findings

- The **LSTM** achieved slightly higher accuracy and weighted F1 on the test set, with stronger precision.
- The **Bi-LSTM** achieved higher recall and a better ROC-AUC, both on the test set and in cross-validation, indicating it generalises slightly better at distinguishing PHM from Non-PHM tweets overall.
- Cross-validation confirms that both models perform consistently across folds, with the Bi-LSTM having a small but consistent edge in macro-F1 and AUC.

## Project Structure

```
.
├── Personal Health Mention Tweet Classification.ipynb   # Main notebook (EDA, preprocessing, training, evaluation)
├── Personal Health Mention Tweet Classification.pdf      # Exported report / notebook output
└── README.md
```

## Tech Stack

- **Python** — NumPy, Pandas, Matplotlib, Seaborn
- **TensorFlow / Keras** — Sequential, LSTM, Bidirectional LSTM
- **scikit-learn** — metrics, class weighting, stratified K-fold cross-validation
- **NLTK** — stopword handling
- **Google Colab** — execution environment

## How to Run

1. Open `Personal Health Mention Tweet Classification.ipynb` in Google Colab (or Jupyter with the required packages installed).
2. Install dependencies:
   ```
   pip install nltk tensorflow scikit-learn matplotlib seaborn scipy
   ```
3. Provide the training and test CSV files (`phm_train.csv`, `phm_test.csv`) with `tweet` and `label` columns.
4. Run all cells sequentially to reproduce preprocessing, training, evaluation, and visualisations.
