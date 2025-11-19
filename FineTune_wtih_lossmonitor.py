from transformers import AutoModel
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
import numpy as np
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments
from transformers import Trainer
from torch.nn.functional import cross_entropy
"""model has a
classification head on top of the pretrained model outputs, which can be easily trained with the
base model"""
from transformers import AutoModelForSequenceClassification


def forward_pass_with_label(batch):
 # Place all input tensors on the same device as the model
 inputs = {k:v.to(device) for k,v in batch.items()
 if k in tokenizer.model_input_names}
 with torch.no_grad():
  output = model(**inputs)
  pred_label = torch.argmax(output.logits, axis=-1)
  loss = cross_entropy(output.logits, batch["label"].to(device),
  reduction="none")
 # Place outputs on CPU for compatibility with other dataset columns
 return {"loss": loss.cpu().numpy(),
 "predicted_label": pred_label.cpu().numpy()}


def compute_metrics(pred):
 labels = pred.label_ids
 preds = pred.predictions.argmax(-1)
 f1 = f1_score(labels, preds, average="weighted")
 acc = accuracy_score(labels, preds)
 return {"accuracy": acc, "f1": f1}

def tokenize(batch):
 return tokenizer(batch["text"], padding=True, truncation=True)
 
model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""The AutoModel class converts the token encodings to embeddings, and then feeds them
through the encoder stack to return the hidden states"""

model = AutoModel.from_pretrained(model_ckpt).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


emotions = load_dataset("emotion")
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)


num_labels = 6
model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device))

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"

training_args = TrainingArguments(output_dir=model_name,
 num_train_epochs=2,
 learning_rate=2e-5,
 per_device_train_batch_size=batch_size,
 per_device_eval_batch_size=batch_size,
 weight_decay=0.01,
 disable_tqdm=False,
 logging_steps=logging_steps,
 push_to_hub=True,
 log_level="error")

trainer = Trainer(model=model, args=training_args,
 compute_metrics=compute_metrics,
 train_dataset=emotions_encoded["train"],
 eval_dataset=emotions_encoded["validation"],
 tokenizer=tokenizer)
trainer.train();

# Convert our dataset back to PyTorch tensors
emotions_encoded.set_format("torch",
 columns=["input_ids", "attention_mask", "label"])
# Compute loss values
emotions_encoded["validation"] = emotions_encoded["validation"].map(
 forward_pass_with_label, batched=True, batch_size=16)


emotions_encoded.set_format("pandas")
cols = ["text", "label", "predicted_label", "loss"]
df_test = emotions_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = (df_test["predicted_label"]
 .apply(label_int2str))







