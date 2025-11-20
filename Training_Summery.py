from datasets import load_dataset
from transformers import pipeline
from huggingface_hub import login
import torch 
import tqdm
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
model_ckpt = "google/pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

device = "cuda" if torch.cuda.is_available() else "cpu"
def chunks(list_of_elements, batch_size):
 """Yield successive batch-sized chunks from list_of_elements."""
 for i in range(0, len(list_of_elements), batch_size):
   yield list_of_elements[i : i + batch_size]


def evaluate_summaries_pegasus(dataset, metric, model, tokenizer,
 batch_size=16, device=device,
 column_text="article",
 column_summary="highlights"):



 article_batches = list(chunks(dataset[column_text], batch_size))
 target_batches = list(chunks(dataset[column_summary],batch_size))
 for article_batch, target_batch in tqdm(zip(article_batches, target_batches),total=len(article_batches)):
  inputs = tokenizer(article_batch, max_length=1024,truncation=True,padding="max_length", return_tensors="pt")
  summaries = model.generate(input_ids=inputs["input_ids"].to(device),attention_mask=inputs["attention_mask"].to(device),length_penalty=0.8, num_beams=8,max_length=128)
  decoded_summaries = [tokenizer.decode(s,skip_special_tokens=True,clean_up_tokenization_spaces=True) for s in summaries]
  decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries]
  metric.add_batch(predictions=decoded_summaries, references=target_batch)
  score = metric.compute()
 return score



 
dataset_samsum = load_dataset("knkarthick/samsum")
split_lengths = [len(dataset_samsum[split])for split in
dataset_samsum]




pipe = pipeline("summarization", model="google/pegasuscnn_dailymail")
pipe_out = pipe(dataset_samsum["test"][0]["dialogue"])


def convert_examples_to_features(example_batch):
 input_encodings = tokenizer(example_batch["dialogue"],max_length=1024,truncation=True)
 with tokenizer.as_target_tokenizer():
  target_encodings = tokenizer(example_batch["summary"],max_length=128, truncation=True)
 return {"input_ids": input_encodings["input_ids"],
 "attention_mask": input_encodings["attention_mask"],
 "labels": target_encodings["input_ids"]}

dataset_samsum_pt = dataset_samsum.map(convert_examples_to_features,batched=True)
columns = ["input_ids", "labels", "attention_mask"]
dataset_samsum_pt.set_format(type="torch", columns=columns)




seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer,model=model)

training_args = TrainingArguments(
 output_dir='pegasus-samsum', num_train_epochs=1,
warmup_steps=500,
 per_device_train_batch_size=1, per_device_eval_batch_size=1,
 weight_decay=0.01, logging_steps=10, push_to_hub=True,
 evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
 gradient_accumulation_steps=16)

trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer,
data_collator=seq2seq_data_collator,
 train_dataset=dataset_samsum_pt["train"],
 eval_dataset=dataset_samsum_pt["validation"])

 trainer.train()
score = evaluate_summaries_pegasus(
 dataset_samsum["test"], rouge_metric, trainer.model, tokenizer,
 batch_size=2, column_text="dialogue", column_summary="summary")
rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in
rouge_names)
pd.DataFrame(rouge_dict, index=[f"pegasus"])
gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length":
128}
sample_text = dataset_samsum["test"][0]["dialogue"]
reference = dataset_samsum["test"][0]["summary"]
pipe = pipeline("summarization", model="transformersbook/pegasussamsum")
print("Dialogue:")
print(sample_text)
print("\nReference Summary:")
print(reference)
print("\nModel Summary:")
print(pipe(sample_text, **gen_kwargs)[0]["summary_text"])

custom_dialogue = """\
Thom: Hi guys, have you heard of transformers?
Lewis: Yes, I used them recently!
Leandro: Indeed, there is a great library by Hugging Face.
Thom: I know, I helped build it ;)
Lewis: Cool, maybe we should write a book about it. What do you
think?
Leandro: Great idea, how hard can it be?!
Thom: I am in!
Lewis: Awesome, let's do it together!
"""
print(pipe(custom_dialogue, **gen_kwargs)[0]["summary_text"])




