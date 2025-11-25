from transformers import AutoTokenizer
import torch
from transformers import AutoModelForQuestionAnswering
model_ckpt = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
question = "What is the requirements for cryptographic key management life cycle?"
quesions2 = "who should perform cryptographic key life cycle management?"

quesions3 = "who should perform automated vulnerability scanning?"

context = """Financial entities shall include in the cryptographic key management policy referred
to in Article 6(2), point (d), requirements for managing cryptographic keys through
their whole lifecycle, including generating, renewing, storing, backing up, archiving,
retrieving, transmitting, retiring, revoking, and destroying those cryptographic keys."""

context2 ="""For the purposes of point (b), financial entities shall perform the automated
vulnerability scanning and assessments on ICT assets for the ICT assets supporting
critical or important functions on at least a weekly basis.
For the purposes of point (c), financial entities shall request that ICT third-party
service providers investigate the relevant vulnerabilities, determine the root causes,
and implement appropriate mitigating action.
For the purposes of point (d), financial entities shall, where appropriate in
collaboration with the ICT third-party service provider, monitor the version and
possible updates of the third-party libraries. In case of ready to use (off-the-shelf)
ICT assets or components of ICT assets acquired and used in the operation of ICT
services not supporting critical or important functions, financial entities shall track
the usage to the extent possible of third-party libraries, including open-source
libraries.
For the purposes of point (f), financial entities shall consider the criticality of the
vulnerability, the classification established in accordance with Article 8(1) of
Regulation (EU) 2022/2554, and the risk profile of the ICT assets affected by the
identified vulnerabilities."""

inputs = tokenizer(question, context, return_tensors="pt")
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
ans = pipe(question=quesions2, context=context, topk=3)
print(ans[0]['answer'])
#Long passages 
tokenized_example = tokenizer(quesions3, context2, return_overflowing_tokens=True, max_length=100, stride=25)

pipe = pipeline("question-answering", model=model, tokenizer=tokenized_example,truncation=True)
ans = pipe(question=quesions3, context=context2, top_k=3)
print(ans[0]['answer 3'])

 
