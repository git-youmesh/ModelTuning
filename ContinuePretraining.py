from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from transformers import pipeline
# Load model for masked language modeling (MLM)
def preprocess_function(examples):
 return tokenizer(examples["text"], truncation=True)

model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tomatoes = load_dataset("rotten_tomatoes")
train_data, test_data = tomatoes["train"], tomatoes["test"]

tokenized_train = train_data.map(preprocess_function,
batched=True)
tokenized_train = tokenized_train.remove_columns("label")
tokenized_test = test_data.map(preprocess_function, batched=True)
tokenized_test = tokenized_test.remove_columns("label")

data_collator = DataCollatorForLanguageModeling(
 tokenizer=tokenizer,
 mlm=True,
 mlm_probability=0.15
)

training_args = TrainingArguments(
 "model",
 learning_rate=2e-5,
 per_device_train_batch_size=16,
 per_device_eval_batch_size=16,
 num_train_epochs=10,
 weight_decay=0.01,
 save_strategy="epoch",
 report_to="none"
)

# Initialize Trainer
trainer = Trainer(
 model=model,
 args=training_args,
 train_dataset=tokenized_train,
 eval_dataset=tokenized_test,
 tokenizer=tokenizer,
 data_collator=data_collator
)

# Save pre-trained tokenizer
tokenizer.save_pretrained("mlm")
# Train model
trainer.train()
# Save updated model
model.save_pretrained("mlm")

mask_filler = pipeline("fill-mask", model="bert-base-cased")
preds = mask_filler("What a horrible [MASK]!")

for pred in preds:
 print(f">>> {pred["sequence"]}")


mask_filler = pipeline("fill-mask", model="mlm")
preds = mask_filler("What a horrible [MASK]!")
# Print results
for pred in preds:
 print(f">>> {pred["sequence"]}")

 
