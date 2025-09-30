import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import evaluate

from datasetbuilder import TextDataset

# read dataset from the huggingface hub
dataset = load_dataset("sms_spam")

# split into train & validation sets
train_texts, validation_texts, train_labels, validation_labels = train_test_split(
    dataset['train']['sms'], dataset['train']['label'], test_size=0.2, random_state=42
)

# load the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# tokenize the texts
train_encodings = tokenizer(
    train_texts, 
    truncation=True,
    padding=True,
    max_length=16
)

validation_encodings = tokenizer(
    validation_texts, 
    truncation=True,
    padding=True,
    max_length=16
)

# Create train & validation customized datasets
train_dataset = TextDataset(train_encodings, train_labels)
validation_dataset = TextDataset(validation_encodings, validation_labels)

# load the model from pretrained DistilBERT in huggingface
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2   # spam vs ham
)

# set device (GPU or CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

# evaluation metrics
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return clf_metrics.compute(predictions=predictions, references=labels)

# define training arguments
training_args = TrainingArguments(
    output_dir="spam detection",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    save_strategy="epoch",    
    lr_scheduler_type="cosine",
    learning_rate=2e-5,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    num_train_epochs=3,
    warmup_steps=50,
    logging_dir="logs",
    logging_steps=10,
    metric_for_best_model= "eval_loss",

)
# create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# train and evaluate the model
trainer.train()
metrics = trainer.evaluate()

# test the model on new data
test_data = ['Let\'s go to gym tomorrow morning.', 'Win lottery by clicking on the link.']
test_encodings = tokenizer(
    test_data,
    truncation=True,
    padding=True,
    max_length=16,
    return_tensors="pt"
).to(device)

test_dataset = TextDataset(test_encodings, labels=[0]*len(test_data))  # dummy labels

predictions = trainer.predict(test_dataset)
logits = predictions.predictions
preds = np.argmax(logits, axis=-1)
print('predictions:', preds)  # 0: ham, 1: spam