# SMS Spam Detection with DistilBERT (Hugging Face, NLP)

**Text Classification Project | NLP | Transformers | PyTorch | Hugging Face**

---

## Project Overview

This project demonstrates **SMS spam detection** using **DistilBERT**, a lightweight Transformer model from Hugging Face. The goal is to classify text messages as **spam** or **ham (not spam)** using modern NLP techniques.

- **Dataset:** Public **SMS Spam Collection dataset** (downloadable from GitHub)  
- **Training:** GPU-ready training with **cosine learning rate scheduler** and **warmup steps**  
- **Task:** Binary text classification  

---

## Process

1. Split dataset into **training** and **validation** sets  
2. **Tokenize** the text messages using Hugging Face tokenizer  
3. **Fine-tune DistilBERT** on the SMS messages  
4. Evaluate model using **accuracy, precision, recall, and F1-score**  
5. Make **predictions on new SMS messages**

---

## Installation

```bash
# Clone the repository
git clone https://github.com/kimiarezaei/sms-spam-distilbert.git
cd sms-spam-distilbert

# Install dependencies
pip install -r requirements.txt
