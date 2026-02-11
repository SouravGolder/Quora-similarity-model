# Quora Similarity Model (Optimized Siamese GRU)

A high-performance **Siamese GRU model** for detecting duplicate Quora question pairs. Optimized for **fast training** and **robust generalization**, achieving strong accuracy and F1 metrics while preventing overfitting.

---

## 🛠 Features

- **Fast Training (seconds/minutes per epoch)**  
  - Optimized GRU with CuDNN acceleration  
  - Removed recurrent dropout  
  - Reduced sequence length  
  - Frozen embeddings  
  - Larger batch size  

- **Good Generalization**  
  - Dropout (0.3–0.4)  
  - L2 regularization  
  - Batch normalization  
  - Early stopping  
  - ReduceLROnPlateau  
  - Balanced class weights  

- **Performance**  
  - Accuracy: ~0.73–0.74  
  - F1 Score: ~0.68  
  - ROC AUC: ~0.83  
  - Stable validation curves, no overfitting or underfitting

---

## 📁 Project Structure
Quora-similarity-model  
├── README.md  
├── train.csv  
├── model  
│   ├── siamese_fast_gru.h5  
│   └── tokenizer.pkl  
└── notebook.ipynb  


---

## 🧠 Approach Summary

### 1️⃣ Data Preprocessing
- Lowercasing  
- Removing punctuation  
- Stopword removal  
- Lemmatization  
- Tokenization  
- Padding sequences to fixed length  

### 2️⃣ Siamese GRU Architecture (Fast Version)
Both questions are passed through a **shared GRU encoder**, producing vectors:  
- Q1 ─► [Shared Encoder] ─► Vector 1  
- Q2 ─► [Shared Encoder] ─► Vector 2

Similarity is computed using:
- Absolute difference  
- Elementwise multiplication  

Final prediction is generated through **dense layers** with dropout & L2 regularization.

### Key Design Decisions
- GRU instead of LSTM → fewer parameters, faster training  
- No recurrent dropout → enables CuDNN acceleration  
- Shorter sequence length (25 tokens)  
- Embedding dimension = 50, frozen layer  
- Increased batch size (up to 512)  

**Training time:** reduced from 4 hours → <1 minute per epoch (GPU)

### Overfitting Prevention
- Lower model capacity  
- Dropout (0.3–0.4)  
- L2 regularization  
- Early stopping  
- Reduce learning rate on plateau  

---

## 📈 Interpreting Training Curves
- Non-monotonic learning is normal  
- Validation loss should stabilize or decrease  
- Validation AUC should trend upward  

---

## 📦 Future Improvements
- Pre-trained embeddings (GloVe / fastText)  
- BERT or Sentence-BERT Siamese architecture  
- Hard negative mining  
- Balanced classes  

---

