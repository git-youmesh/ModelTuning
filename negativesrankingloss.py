import random
from tqdm import tqdm
from datasets import Dataset, load_dataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from sentence_transformers import losses, SentenceTransformer
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


# # Load MNLI dataset from GLUE
mnli = load_dataset("glue", "mnli",
split="train").select(range(50_000))
mnli = mnli.remove_columns("idx")
mnli = mnli.filter(lambda x: True if x["label"] == 0 else False)
print(mnli[2])
# Prepare data and add a soft negative
train_dataset = {"anchor": [], "positive": [], "negative": []}
soft_negatives = mnli["hypothesis"]
random.shuffle(soft_negatives)
for row, soft_negative in tqdm(zip(mnli, soft_negatives)):
 train_dataset["anchor"].append(row["premise"])
 train_dataset["positive"].append(row["hypothesis"])
 train_dataset["negative"].append(soft_negative)
train_dataset = Dataset.from_dict(train_dataset)

print(train_dataset[3])


# Create an embedding similarity evaluator for stsb
val_sts = load_dataset("glue", "stsb", split="validation")
evaluator = EmbeddingSimilarityEvaluator(
 sentences1=val_sts["sentence1"],
 sentences2=val_sts["sentence2"],
 scores=[score/5 for score in val_sts["label"]],
 main_similarity="cosine"
)

embedding_model = SentenceTransformer('bert-base-uncased')
# Loss function
train_loss =
losses.MultipleNegativesRankingLoss(model=embedding_model)
# Define the training arguments
args = SentenceTransformerTrainingArguments(
 output_dir="mnrloss_embedding_model",
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
