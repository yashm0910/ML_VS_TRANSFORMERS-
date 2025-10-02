# Multi-Class News Classification: Traditional ML vs. Transformer Models

---

## Overview

News article classification is essential for organizing vast amounts of textual data, enabling efficient information retrieval and recommendation systems. This project benchmarks **six traditional machine learning models** against a state-of-the-art **transformer-based model** (DistilRoBERTa-base) for multi-class text classification across four categories: **World**, **Sports**, **Business**, and **Sci/Tech**. 

Traditional models leverage TF-IDF vectorization for feature extraction, while the transformer employs contextual embeddings for superior semantic understanding. The evaluation focuses on accuracy, F1-score, and computational efficiency, highlighting trade-offs between simplicity, speed, and performance in real-world NLP pipelines.

---

## Dataset

- **Source**: AG News Dataset (subset of 7,600 test samples; balanced across classes)
- **Classes**: `World`, `Sports`, `Business`, `Sci/Tech` (1900 samples each)
- **Preprocessing**:
  - Text cleaning: Lowercasing, punctuation removal, stopword filtering (for ML models)
  - Tokenization and padding (for transformer)
  - Train/validation/test splits: 70/15/15 (stratified to preserve class balance)

---

## Methodology

### 1. Traditional ML Models
Six baseline models were trained using **TF-IDF vectorization** (max features: 10,000) on preprocessed text:
- **Logistic Regression**: Linear classifier with L2 regularization
- **Multinomial Naive Bayes**: Probabilistic model assuming feature independence
- **Support Vector Machine (SVM)**: Kernel-based classifier (linear kernel, C=1.0)
- **Random Forest**: Ensemble of decision trees (n_estimators=100)
- **Gradient Boosting**: XGBoost implementation (n_estimators=100, learning_rate=0.1)
- **K-Nearest Neighbors (KNN)**: Instance-based learning (k=5)

Hyperparameters tuned via GridSearchCV on validation set.

### 2. Transformer Model
- **Architecture**: DistilRoBERTa-base (distilled version of RoBERTa for efficiency; 82M parameters)
- **Fine-Tuning**: Added classification head (hidden_size=768 → Dropout(0.1) → Linear(4))
- **Training**: AdamW optimizer (lr=2e-5), CrossEntropyLoss, batch_size=16, epochs=3
- **Inference**: Hugging Face Transformers pipeline for tokenization and prediction

### 3. Evaluation Strategy
- **Metrics**: Accuracy, Macro F1-Score (balanced multi-class measure)
- **Efficiency**: Training and prediction times recorded on standard hardware (CPU: Intel i7)
- **Cross-Validation**: 5-fold for ML models; held-out test set for all

---

## Results

### Transformer Model (DistilRoBERTa-base)
| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| **World**  | 0.94     | 0.92   | 0.93    | 1900   |
| **Sports** | 0.97     | 0.99   | 0.98    | 1900   |
| **Business**| 0.90    | 0.88   | 0.89    | 1900   |
| **Sci/Tech**| 0.89    | 0.91   | 0.90    | 1900   |
| **Accuracy** | -       | -      | 0.92    | 7600   |
| **Macro Avg** | 0.92  | 0.92   | 0.92    | 7600   |
| **Weighted Avg** | 0.92 | 0.92 | 0.92 | 7600 |

### Traditional ML Models
| Model                  | Accuracy | F1-Score | Train Time (s) | Predict Time (s) |
|------------------------|----------|----------|----------------|------------------|
| **Logistic Regression** | 0.8870  | 0.8864  | 0.54          | 0.00            |
| **Multinomial NB**     | 0.8910  | 0.8905  | 0.01          | 0.00            |
| **SVM**                | 0.8930  | 0.8925  | 328.06        | 3.11            |
| **Random Forest**      | 0.7990  | 0.7953  | 4.14          | 0.07            |
| **Gradient Boosting**  | 0.8500  | 0.8494  | 199.47        | 0.01            |
| **KNN**                | 0.8580  | 0.8575  | 0.00          | 0.85            |

*Note*: Times measured on test set (7,600 samples). Transformer training: ~15 min on GPU; inference: ~2s (batch=1).

**Key Comparison**: DistilRoBERTa outperforms all ML models (92% vs. max 89.3% accuracy), with balanced per-class performance. ML models excel in speed for low-resource settings, with Naive Bayes offering the best train/predict efficiency.

---

## Key Highlights

- **Performance Edge**: Transformers capture nuanced semantics, yielding ~3% accuracy gain over top ML baseline (SVM)
- **Efficiency Trade-Off**: ML models (e.g., Naive Bayes) train in seconds on CPU, ideal for rapid prototyping; transformers require GPU for fine-tuning but enable scalable deployment
- **Interpretability**: ML models support feature importance analysis; transformers integrable with attention visualization for explainable predictions

---

## Future Enhancements

- Integrate **ensemble methods** (e.g., stacking ML + transformer outputs) for hybrid gains
- Scale to **larger datasets** (e.g., full AG News) or **few-shot learning** with prompt tuning
- Deploy via **FastAPI** for real-time news categorization API
- Explore **lightweight transformers** (e.g., DistilBERT variants) for edge devices

---

## Tech Stack

- **Language**: Python 3.10+
- **ML Frameworks**: Scikit-learn (traditional models), XGBoost (boosting)
- **Transformers**: Hugging Face Transformers, PyTorch
- **Data Handling**: Pandas, NumPy
- **Evaluation/Visualization**: Scikit-learn metrics, Matplotlib/Seaborn
- **Environment**: Jupyter Notebook / Google Colab (GPU-enabled for transformers)

---

## Impact

This benchmarking study underscores the evolution of NLP paradigms—from classical ML's efficiency to transformers' contextual prowess—guiding practitioners in selecting models for production. By open-sourcing code and results, it fosters reproducible research in text classification, with applications in journalism, content moderation, and personalized feeds. Full implementation available in the repository for easy replication and extension.
