"""The underlying idea of TSDAE is that we add noise to the input sentence by
removing a certain percentage of words from it. This “damaged” sentence is
put through an encoder, with a pooling layer on top of it, to map it to a
sentence embedding. From this sentence embedding, a decoder tries to
reconstruct the original sentence from the “damaged” sentence but without
the artificial noise. The main concept here is that the more accurate the
sentence embedding is, the more accurate the reconstructed sentence will
be"""


import nltk
nltk.download("punkt")
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from tqdm import tqdm
from datasets import Dataset, load_dataset
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from sentence_transformers import losses
from sentence_transformers import models, SentenceTransformer


# Create a flat list of sentences
mnli = load_dataset("glue", "mnli",
split="train").select(range(25_000))
flat_sentences = mnli["premise"] + mnli["hypothesis"]
# Add noise to our input data
damaged_data = DenoisingAutoEncoderDataset(list(set(flat_sentences)))
# Create dataset
train_dataset = {"damaged_sentence": [], "original_sentence": []}
for data in tqdm(damaged_data):
 train_dataset["damaged_sentence"].append(data.texts[0])
 train_dataset["original_sentence"].append(data.texts[1])
train_dataset = Dataset.from_dict(train_dataset)
print(train_dataset[2])

# Create an embedding similarity evaluator for stsb
val_sts = load_dataset("glue", "stsb", split="validation")
evaluator = EmbeddingSimilarityEvaluator(
 sentences1=val_sts["sentence1"],
 sentences2=val_sts["sentence2"],
 scores=[score/5 for score in val_sts["label"]],
 main_similarity="cosine"
)

word_embedding_model = models.Transformer("bert-base-uncased")
pooling_model =models.Pooling(word_embedding_model.get_word_embedding_dimension(
), "cls")
embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_loss = losses.DenoisingAutoEncoderLoss(
 embedding_model, tie_encoder_decoder=True
)
train_loss.decoder = train_loss.decoder.to("cuda")
args = SentenceTransformerTrainingArguments(
 output_dir="tsdae_embedding_model",
 num_train_epochs=1,
 per_device_train_batch_size=16,
 per_device_eval_batch_size=16,
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

evaluator(embedding_model)





