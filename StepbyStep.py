from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import evaluate
import numpy as np


def compute_metrics(eval_preds):
    # This function is needed to eval during training process @epocs 
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#Tokenize entier dataset, this is very expensive since
#padding is applied for entier dataset and need huge RAM
# instead we need to 
#device a mentod where we apply padding only just before using 
#the data for training and we need to explore to batch the date as well

"""We can use whereas the datasets from the 🤗 Datasets 
library are Apache Arrow files stored on the disk, so you only
 keep the samples you ask for loaded in memory"""
def tokenize_function(example):
    #input_ids, attention_mask, and token_type_ids
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments("test-trainer")



model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
)
trainer.train()
#output of predict fuction will be : predictions, label_ids, and metrics. 
predictions = trainer.predict(tokenized_datasets["validation"])

# Get max index of logits of dimention two 
preds = np.argmax(predictions.predictions, axis=-1)

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)


""" if want to analyze perforance during the training process we ned to configure 
Trainier to evel at each epocs by passing compure_eval function """

training_args = TrainingArguments("test-trainer", eval_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics, #here 
)

""" to optimize training process we can use use below mentionded attributes of trainer
object 

Advanced Training Features
fp16=True,  # Enable mixed precision
per_device_train_batch_size=4,
gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
eval_strategy="epoch",
learning_rate=2e-5,
lr_scheduler_type="cosine",  # Try different schedulers
"""

