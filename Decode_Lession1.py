import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained(
 "microsoft/Phi-3-mini-4k-instruct",
 device_map="cuda",
 torch_dtype="auto",
 trust_remote_code=False, #'DynamicCache' object has no attribute 'seen_tokens'
)
# Create a pipeline
generator = pipeline(
 "text-generation",
 model=model,
 tokenizer=tokenizer,
 return_full_text=False,
 max_new_tokens=50,
 do_sample=False,
)

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to("cuda")

generation_output = model.generate(
 input_ids=input_ids,
 max_new_tokens=100,
 use_cache=True
)

output = generator(prompt)
print(output[0]['generated_text'])
print(generation_output[0])

# Getting last layer prohb numbers 
prompt = "The capital of France is"
# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# Tokenize the input prompt
input_ids = input_ids.to("cuda")
# Get the output of the model before the lm_head
model_output = model.model(input_ids)
# Get the output of the lm_head
lm_head_output = model.lm_head(model_output[0])

token_id = lm_head_output[0,-1].argmax(-1)
tokenizer.decode(token_id)

 
# Tokenize the input prompt


