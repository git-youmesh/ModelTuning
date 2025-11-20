from datasets import get_dataset_config_names,load_dataset
import pandas as pd
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForQuestionAnswering


question = "What should financial do for correct implementation of security policies?"
context = """To ensure the correct implementation over time of ICT security policies, procedures,
protocols, and tools referred to in Title II, Chapter I of this Regulation, it is important
that financial entities correctly assign and maintain any roles and responsibilities
relating to ICT security, and that they lay down the consequences of non-compliance
with ICT security policies or procedures.
In a dynamic environment where ICT risks constantly evolve, it is important that
financial entities develop their set of ICT security policies on the basis of leading
practices, and where applicable, of standards as defined in Article 2, point (1), of
Regulation (EU) No 1025/2012 of the European Parliament and of the Council3
.This
should enable financial entities referred to in Title II of this Regulation to remain
informed and prepared in a changing landscape

"""



subjqa = load_dataset("megagonlabs/subjqa", name="electronics")
dfs = {split: dset.to_pandas() for split, dset in subjqa.flatten().items()}
qa_cols = ["title", "question", "answers.text",
 "answers.answer_start", "context"]
sample_df = dfs["train"][qa_cols].sample(2, random_state=7)


start_idx = sample_df["answers.answer_start"].iloc[0][0]
end_idx = start_idx + len(sample_df["answers.text"].iloc[0][0])
sample_df["context"].iloc[0][start_idx:end_idx]


model_ckpt = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


""", we can set return_overflowing_tokens=True in the tokenizer to enable the sliding
window"""

example = dfs["train"].iloc[0][["question", "context"]]
tokenized_example = tokenizer(example["question"], example["context"],
 return_overflowing_tokens=True, max_length=100,
 stride=2)


for idx, window in enumerate(tokenized_example["input_ids"]):
 print(f"Window #{idx} has {len(window)} tokens")

"""we can see the familiar input_ids and attention_mask tensors, while the token_type_ids tensor
indicates which part of the inputs corresponds to the question and context (a 0 indicates a question token, a 1
indicates a context token)"""
inputs = tokenizer(question, context, return_tensors="pt")

print(tokenizer.decode(inputs["input_ids"][0]))
"""[CLS] how much music can this hold? [SEP] an mp3 is about 1 mb / minute, so
about 6000 hours depending on file size. [SEP]
We see that for each QA example, the inputs take the format:
[CLS] question tokens [SEP] context tokens [SEP]

"""

model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)
with torch.no_grad():
 outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits

start_idx = torch.argmax(start_logits)
end_idx = torch.argmax(end_logits) + 1
answer_span = inputs["input_ids"][0][start_idx:end_idx]
answer = tokenizer.decode(answer_span)
print(f"Question: {question}")
print(f"Answer: {answer}")



from transformers import pipeline


pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
print(pipe(question=question, context=context, topk=3))




