from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
def compute_metrics(eval_pred):
 """Calculate F1 score"""
 logits, labels = eval_pred
 predictions = np.argmax(logits, axis=-1)
 load_f1 = evaluate.load("f1")
 f1 = load_f1.compute(predictions=predictions,
references=labels)["f1"]
 return {"f1": f1}


def preprocess_function(examples):
 """Tokenize input data"""
 return tokenizer(examples["text"], truncation=True)

# Prepare data and splits


tomatoes = load_dataset("rotten_tomatoes")
train_data, test_data = tomatoes["train"], tomatoes["test"]

# Load model and tokenizer
model_id = "bert-base-cased"
model = AutoModelForSequenceClassification.from_pretrained(
 model_id, num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Tokenize train/test data
tokenized_train = train_data.map(preprocess_function,
batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments(
 "model",
 learning_rate=2e-5,
 per_device_train_batch_size=16,
 per_device_eval_batch_size=16,
 num_train_epochs=1,
 weight_decay=0.01,
 save_strategy="epoch",
 report_to="none"
)
# Trainer which executes the training process
trainer = Trainer(
 model=model,
 args=training_args,
 train_dataset=tokenized_train,
 eval_dataset=tokenized_test,
 tokenizer=tokenizer,
 data_collator=data_collator,
 compute_metrics=compute_metrics,

)

trainer.evaluate()


# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
 model_id, num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
for name, param in model.named_parameters():
 print(name)

for name, param in model.named_parameters():
  # Trainable classification head
  if name.startswith("classifier"):
    param.requires_grad = True
  # Freeze everything else
  else:
    param.requires_grad = False

trainer = Trainer(
 model=model,
 args=training_args,
 train_dataset=tokenized_train,
 eval_dataset=tokenized_test,
 tokenizer=tokenizer,
 data_collator=data_collator,
 compute_metrics=compute_metrics,
)
trainer.train()


trainer.evaluate()
# Load model
model_id = "bert-base-cased"
model = AutoModelForSequenceClassification.from_pretrained(
 model_id, num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Encoder block 11 starts at index 165 and
# we freeze everything before that block
for index, (name, param) in enumerate(model.named_parameters()):
 if index < 165:
  param.requires_grad
# Trainer which executes the training process
trainer = Trainer(
 model=model,
 args=training_args,
 train_dataset=tokenized_train,
 eval_dataset=tokenized_test,
 tokenizer=tokenizer,
 data_collator=data_collator,
 compute_metrics=compute_metrics,
)
trainer.train()

