# Arabic POS Tagging using BiLSTM

This project focuses on building an Arabic Part-of-Speech (POS) Tagging system using Deep Learning with a BiLSTM (Bidirectional Long Short-Term Memory) model.

## Project Overview

The goal of this project is to automatically assign grammatical tags such as:

- NOUN
- VERB
- ADJ
- ADP
- PRON
- DET
- PUNCT
- and more...

to each word in an Arabic sentence.

Example:

ترفض → VERB  
شركة → NOUN  
أمريكية → ADJ

POS Tagging is one of the fundamental tasks in Natural Language Processing (NLP).

---

## Dataset

The dataset is stored in `.conllu` format and contains:

- Tokens (words)
- Lemmas
- Universal POS Tags (UPOS)

Example:

| Token | POS Tag |
|---|---|
| ترفض | VERB |
| شركة | NOUN |
| أمريكية | ADJ |

---

## Preprocessing

The preprocessing steps include:

- Reading `.conllu` files
- Extracting tokens and POS tags
- Cleaning tokens
- Replacing missing tags (`_`) with `O`
- Tokenization
- Padding sequences
- Label encoding

---

## Model Architecture

The model consists of:

- Embedding Layer
- Bidirectional LSTM (BiLSTM)
- TimeDistributed Dense Layer
- Softmax Output Layer

Architecture Flow:

Input → Embedding → BiLSTM → Dense → POS Tag Prediction

---

## Training Setup

Used:

- Adam Optimizer
- Sparse Categorical Crossentropy
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint

---

## Results

### Final Performance

- Accuracy: **95%**
- Weighted F1-score: **95%**

### Strong Classes

- ADP
- PRON
- CCONJ
- PUNCT

### Challenging Classes

- PROPN
- X
- PART

due to class imbalance and linguistic ambiguity.

---

## Evaluation

Evaluation includes:

- Classification Report
- Confusion Matrix
- Normalized Confusion Matrix

Example:

(Attach confusion_matrix.png here)

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## Future Improvements

Possible improvements include:

- CRF Layer
- AraBERT / Transformers
- FastText Embeddings
- Attention Mechanism

---

## Author

Moamen Hamed  
AI Engineer Track | Deep Learning | NLP | Machine Learning
