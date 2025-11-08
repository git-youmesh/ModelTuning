from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from peft import LoraConfig, prepare_model_for_kbit_training,get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
from transformers import pipeline
from peft import AutoPeftModelForCausalLM
# Load a tokenizer to use its chat template
from transformers import TrainingArguments
template_tokenizer = AutoTokenizer.from_pretrained(
 "TinyLlama/TinyLlama-1.1BChat-v1.0"
)
def format_prompt(example):
  """Format the prompt to using the <|user|> template TinyLLama
  is using"""
  # Format answers
  chat = example["messages"]
  prompt = template_tokenizer.apply_chat_template(chat,
  tokenize=False)
  return {"text": prompt}
  # Load and format the data using the template TinyLLama is using
dataset = (
  load_dataset("HuggingFaceH4/ultrachat_200k",
  split="test_sft")
  .shuffle(seed=42)
  .select(range(3_000))
  )
dataset = dataset.map(format_prompt)
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k3T"
# 4-bit quantization configuration - Q in QLoRA
bnb_config = BitsAndBytesConfig(
 load_in_4bit=True, # Use 4-bit precision model loading
 bnb_4bit_quant_type="nf4", # Quantization type
 bnb_4bit_compute_dtype="float16", # Compute dtype
 bnb_4bit_use_double_quant=True, # Apply nested quantization
)
# Load the model to train on the GPU
model = AutoModelForCausalLM.from_pretrained(
 model_name,
 device_map="auto",
 # Leave this out for regular SFT
 quantization_config=bnb_config,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name,
trust_remote_code=True)
tokenizer.pad_token = "<PAD>"
tokenizer.padding_side = "left"


# Prepare LoRA Configuration
peft_config = LoraConfig(
 lora_alpha=32, # LoRA Scaling
 lora_dropout=0.1, # Dropout for LoRA Layers
 r=64, # Rank
 bias="none",
 task_type="CAUSAL_LM",
 target_modules= # Layers to target
 ["k_proj", "gate_proj", "v_proj", "up_proj", "q_proj",
"o_proj", "down_proj"])
# Prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
output_dir = "./results"
# Training arguments
training_arguments = TrainingArguments(
 output_dir=output_dir,
 per_device_train_batch_size=2,
 gradient_accumulation_steps=4,
 optim="paged_adamw_32bit",
 learning_rate=2e-4,
 lr_scheduler_type="cosine",
 num_train_epochs=1,
 logging_steps=10,
 fp16=True,
 gradient_checkpointing=True
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
 model=model,
 train_dataset=dataset,
 dataset_text_field="text",
 tokenizer=tokenizer,
 args=training_arguments,
 max_seq_length=512,
 # Leave this out for regular SFT
 peft_config=peft_config,
)
# Train model
trainer.train()
# Save QLoRA weights
trainer.model.save_pretrained("TinyLlama-1.1B-qlora")

#Merge the weigts 
model = AutoPeftModelForCausalLM.from_pretrained(
 "TinyLlama-1.1B-qlora",
 low_cpu_mem_usage=True,
 device_map="auto",
)

# Merge LoRA and base model
merged_model = model.merge_and_unload()


#Test with new weigths 

# Use our predefined prompt template
prompt = """<|user|>
Tell me something about Large Language Models.</s>
<|assistant|>
"""
# Run our instruction-tuned model
pipe = pipeline(task="text-generation", model=merged_model,
tokenizer=tokenizer)
print(pipe(prompt)[0]["generated_text"])
