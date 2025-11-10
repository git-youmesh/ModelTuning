from datasets import load_dataset,Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers import losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
# Load MNLI dataset from GLUE
# 0 = entailment, 1 = neutral, 2 = contradiction
train_dataset = load_dataset(
 "glue", "mnli", split="train"
).select(range(50_000))
train_dataset = train_dataset.remove_columns("idx")


mapping = {2: 0, 1: 0, 0:1}
train_dataset = Dataset.from_dict({
 "sentence1": train_dataset["premise"],
 "sentence2": train_dataset["hypothesis"],
 "label": [float(mapping[label]) for label in
train_dataset["label"]]
})
"""{'premise': 'One of our number will carry out your instructions minutely.',
'hypothesis': 'A member of my team will execute your orders with immense precision.',
'label': 0}"""
print(train_dataset[2])

#Semantic Textual Similarity Benchmark (STSB), Moreover, we process the STSB data to make sure all values
#are between 0 and 1


val_sts = load_dataset("glue", "stsb", split="validation")
evaluator = EmbeddingSimilarityEvaluator(
 sentences1=val_sts["sentence1"],
 sentences2=val_sts["sentence2"],
 scores=[score/5 for score in val_sts["label"]],
 main_similarity="cosine"
)

train_loss = losses.SoftmaxLoss(
model=embedding_model,
sentence_embedding_dimension=embedding_model.get_sentence_embedding_dimension(),
num_labels=3
)
 

args = SentenceTransformerTrainingArguments( 
 output_dir="cosineloss_embedding_model", 
 num_train_epochs=1,
 per_device_train_batch_size=32,
 per_device_eval_batch_size=32,
 warmup_steps=100,
 fp16=True,
 eval_steps=100,
 logging_step=100
)
embedding_model = SentenceTransformer("bert-base-uncased")
trainer = SentenceTransformerTrainer(
 model=embedding_model,
 args=args,
 train_dataset=train_dataset, # this contain two senetence ( Sentence 1, and Sentence 2) and a lable ( 0 ,1,2) 
 loss=train_loss, # use cosine loss 
 evaluator=evaluator 
)
trainer.train()

evaluator(embedding_model)






