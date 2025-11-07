from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers import losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer

# Load MNLI dataset from GLUE
# 0 = entailment, 1 = neutral, 2 = contradiction
train_dataset = load_dataset(
 "glue", "mnli", split="train"
).select(range(50_000))
train_dataset = train_dataset.remove_columns("idx")
print(train_dataset[2])

embedding_model = SentenceTransformer('bert-base-uncased')
train_loss = losses.SoftmaxLoss(
model=embedding_model,
sentence_embedding_dimension=embedding_model.get_sentence_embedding_dimension(),
num_labels=3
)

val_sts = load_dataset("glue", "stsb", split="validation")
evaluator = EmbeddingSimilarityEvaluator(
 sentences1=val_sts["sentence1"],
 sentences2=val_sts["sentence2"],
 scores=[score/5 for score in val_sts["label"]],
 main_similarity="cosine",
)

args = SentenceTransformerTrainingArguments(
 output_dir="base_embedding_model",
 num_train_epochs=1,
 per_device_train_batch_size=32,
 per_device_eval_batch_size=32,
 warmup_steps=100,
 fp16=True,
 eval_steps=100,
 logging_steps=100,
)

trainer = SentenceTransformerTrainer(
 model=embedding_model,
 args=args,
 train_dataset=train_dataset,
 loss=train_loss,
 evaluator=evaluator
)
trainer.train()

elevaluator(embedding_model)

