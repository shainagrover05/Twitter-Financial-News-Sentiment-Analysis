# 📈 Financial Sentiment Classification from Twitter

End-to-end NLP pipeline for classifying financial tweets as bearish, bullish, or neutral using the HuggingFace *Twitter Financial News Sentiment* dataset.

## 🚀 Project Highlights

- Designed and implemented a full ML workflow from data ingestion → feature engineering → modeling → evaluation.
- Compared multiple modeling approaches including Logistic Regression, Random Forest, and a class-weighted PyTorch Neural Network.
- Applied TF-IDF vectorization (5,000 features) and engineered domain-relevant features (tickers, numeric signals, text length).
- Performed ROC-AUC analysis, cross-validation, and threshold optimization (Youden’s J) to improve decision boundaries.
- Explored representation learning via TruncatedSVD (PCA for sparse text) and evaluated clustering separability using K-Means + Hungarian alignment.

## 📊 Results

- Achieved strong ROC-AUC performance in binary sentiment classification.
- Neural network reached ~73% accuracy on 3-class sentiment prediction.
- Demonstrated that financial sentiment signals are partially separable but not linearly trivial.

## 🧠 Technical Focus

- Text representation learning
- Model comparison & evaluation strategy
- Class imbalance handling
- Supervised vs. unsupervised signal structure

## 🛠 Tech Stack

Python · scikit-learn · PyTorch · HuggingFace Datasets · NumPy · Pandas · Matplotlib
