✅ Fast Training (seconds/minutes per epoch)

Uses optimized GRU with CuDNN acceleration

Removed recurrent dropout

Reduced sequence length

Frozen embeddings

Larger batch size

🎯 Good Generalization

Dropout

L2 regularization

Batch normalization

Early stopping

ReduceLROnPlateau

Balanced class weights

📈 Performance

The optimized model typically reaches:

Metric	Score
Accuracy	~0.73–0.74
F1 Score	~0.68
ROC AUC	~0.83

No overfitting or underfitting, stable validation curves.

📄 Project Structure
├── README.md
├── train.csv
├── model/
│   ├── siamese_fast_gru.h5
│   └── tokenizer.pkl
└── notebook.ipynb

🧠 Approach Summary
1️⃣ Data Preprocessing

Lowercasing

Removing punctuation

Stopword removal

Lemmatization

Tokenization

Padding sequences to fixed length

2️⃣ Siamese GRU Architecture (Fast Version)

A Siamese network processes both questions through the same shared encoder:

Q1 ─► [Shared Encoder] ─► Vector 1
Q2 ─► [Shared Encoder] ─► Vector 2


We compute similarity using:

Absolute difference

Elementwise multiplication

Final prediction is generated through dense layers with dropout & L2 regularization.

* Key Design Decisions
* Why the model is much faster

GRU instead of LSTM → fewer parameters

No recurrent dropout → enables CuDNN acceleration

Smaller sequence length (25 tokens)

Embedding dimension = 50

Frozen embedding layer

Batch size increased (up to 512)

Training time dropped from 4 hours to <1 minute per epoch (GPU).

🔍 Why Overfitting is Solved

Lower model capacity

Dropout (0.3–0.4)

L2 regularization

Early stopping

Reduce learning rate on plateau

📊 Interpreting Training Curves

If validation accuracy increases, then drops, then increases again →
this is called non-monotonic learning and is normal.

The model is healthy as long as:

Validation loss stabilizes or decreases

Validation AUC trends upward

📦 Future Improvements

You can extend the project by adding:

❇️ Pre-trained embeddings (GloVe/fastText)

❇️ BERT or Sentence-BERT Siamese architecture

❇️ Hard negative mining

Balanced classes

Results show validation and training curves closely aligned.
