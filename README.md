# AI-Tranformer-Transfer-Learning

# 🧠 Cyberbullying Detection with BERT and Transfer Learning

---

## 📄 Project Overview

This project aims to detect cyberbullying in tweets using NLP techniques, ranging from traditional models with word embeddings to state-of-the-art transfer learning with BERT. The final goal is to implement a robust, real-world-ready binary classifier.

---

## 📊 Dataset Overview

**Source**: [Kaggle - Cyberbullying Tweets](https://www.kaggle.com/datasets/soorajtomar/cyberbullying-tweets/data)  
**Collected By**: Sooraj Tomar  
**Size**: 11,100 tweets  
**Classes**:  
- `1` - Cyberbullying  
- `0` - Not cyberbullying  

**Key Characteristics**:
- Perfectly balanced dataset (50/50)
- Merged from multiple Kaggle datasets
- Labels: `CB_Label` (binary)
- Feature: Raw tweet text (single input feature)
- Uni-modal (text only)

---

## 🧪 Problem Statement

- **Task Type**: Binary text classification  
- **Not Multi-task**: Only one label per tweet  
- **Evaluation Metric**: **F1 Score**  
  - Balances precision and recall
  - Minimizes false positives and false negatives
  - Crucial for real-world use in school settings

---

## ⚙️ Preprocessing & Data Split

- Tokenized tweets using BERT tokenizer
- Sequence length: Padded/truncated to 35 tokens
- Dropped tweets with >50 tokens
- Random train/test split with stratified labels
- Final Dataset:  
  - `X` Shape: (10757, 35)  
  - `y` Shape: (10757,)  
  - Class Distribution: [5537, 5220]

---

## 🧠 Models Used

### Model 1: **Baseline (No Transfer Learning)**
- Traditional model trained from scratch
- Word embeddings from [GloVe](https://nlp.stanford.edu/projects/glove/)
- Simple dense architecture
- Used for performance benchmarking

### Model 2: **Transfer Learning with BERT (Bottleneck Features)**
- Utilized pretrained BERT model for embeddings
- Added classification head
- BERT layers frozen (no fine-tuning)

### Model 3: **Fine-Tuned BERT**
- Same as Model 2, but unfroze last BERT layer for fine-tuning
- Trained for 2 epochs (showed signs of convergence)
- Best performing model

---

## 📦 Packages & Technologies

- Python 3.x
- TensorFlow / Keras
- Hugging Face Transformers
- scikit-learn
- Pandas / NumPy
- Matplotlib
- BERT (via `bert-base-uncased`)
- GloVe embeddings

---

## 📈 Results

### **F1 Score Comparison**
| Model                 | F1 Score |
|----------------------|----------|
| Model 1 (GloVe)       | 0.6945   |
| Model 2 (BERT static) | 0.6594   |
| Model 3 (BERT fine-tuned) | **0.7720** |

### **Accuracy Comparison**
| Model                 | Accuracy |
|----------------------|----------|
| Model 1              | 0.7412   |
| Model 2              | 0.7640   |
| Model 3              | **0.8081** |

### **Validation Loss**
| Model                 | Val Loss |
|----------------------|----------|
| Model 1              | 0.5496   |
| Model 2              | 0.4553   |
| Model 3              | **0.4070** |

---

## ⏱️ Performance vs. Cost Tradeoff

While Model 3 delivers the best performance, it comes at the cost of high training time (1000x vs Model 1). For a yearly retraining schedule in schools, the tradeoff is justified due to:
- Higher safety and trust in classification
- Greater adaptability to evolving language
- Potential centralized training and decentralized usage

---

## 📌 Conclusion

Fine-tuning BERT significantly enhances the model’s ability to identify cyberbullying content in tweets. Transfer learning, especially with partial layer retraining, provides the best balance between performance and generalization. This approach is well-suited for deployment in sensitive real-world scenarios like schools.

---
