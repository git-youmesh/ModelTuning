from tqdm import tqdm
from datasets import load_dataset, Dataset
from sentence_transformers import InputExample
from sentence_transformers.datasets import NoDuplicatesDataLoader
import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder
# Prepare a small set of 10000 documents for the cross-encoder
dataset = load_dataset("glue", "mnli",
split="train").select(range(10_000))
mapping = {2: 0, 1: 0, 0:1}
# Data loader
gold_examples = [
 InputExample(texts=[row["premise"], row["hypothesis"]],
label=mapping[row["label"]])
 for row in tqdm(dataset)
]
gold_dataloader = NoDuplicatesDataLoader(gold_examples,
batch_size=32)
# Pandas DataFrame for easier data handling
gold = pd.DataFrame(
 {
 "sentence1": dataset["premise"],
 "sentence2": dataset["hypothesis"],
 "label": [mapping[label] for label in dataset["label"]]
 }
)
print(gold.head(2))



# Train a cross-encoder on the gold dataset
cross_encoder = CrossEncoder("bert-base-uncased", num_labels=2)
cross_encoder.fit(
 train_dataloader=gold_dataloader,
 epochs=1,
 show_progress_bar=True,
 warmup_steps=100,
 use_amp=False
)

# Prepare the silver dataset by predicting labels with the crossencoder
silver = load_dataset(
 "glue", "mnli", split="train"
).select(range(10_000, 50_000))
pairs = list(zip(silver["premise"], silver["hypothesis"]))

# Label the sentence pairs using our fine-tuned cross-encoder
output = cross_encoder.predict(
 pairs, apply_softmax=True,
show_progress_bar=True
)


silver = pd.DataFrame(
 {
 "sentence1": silver["premise"],
 "sentence2": silver["hypothesis"],
 "label": np.argmax(output, axis=1)
 }
)

data = pd.concat([gold, silver], ignore_index=True, axis=0)
data = data.drop_duplicates(subset=["sentence1", "sentence2"],keep="first")
train_dataset = Dataset.from_pandas(data, preserve_index=False)

embedding_model = SentenceTransformer("bert-base-uncased")
# Loss function
train_loss = losses.CosineSimilarityLoss(model=embedding_model)
# Define the training arguments
args = SentenceTransformerTrainingArguments(
 output_dir="augmented_embedding_model",
 num_train_epochs=1,
 per_device_train_batch_size=32,
 per_device_eval_batch_size=32,
 warmup_steps=100,
 fp16=True,
 eval_steps=100,
 logging_steps=100,
)
# Train model
trainer = SentenceTransformerTrainer(
 model=embedding_model,
 args=args,
 train_dataset=train_dataset,
 loss=train_loss,
 evaluator=evaluator
)
trainer.train()

evaluator(embedding_model)





