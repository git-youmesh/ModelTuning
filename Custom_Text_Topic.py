import torch.nn as nn
from transformers import XLMRobertaConfig
from transformers import AutoTokenizer
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from collections import defaultdict
from datasets import DatasetDict
from datasets import load_dataset
import pandas as pd
import torch
from transformers import AutoConfig
load_dataset("xtreme", name="PAN-X.de")
import pandas as pd
text = "Jack Sparrow loves New York!"

bert_model_name = "bert-base-cased"
xlmr_model_name = "xlm-roberta-base"





bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)

bert_tokens = bert_tokenizer(text).tokens()
xlmr_tokens = xlmr_tokenizer(text).tokens()


def create_tag_names(batch):
 return {"ner_tags_str": [tags.int2str(idx) for idx in batch["ner_tags"]]}


langs = ["de", "fr", "it", "en"]
fracs = [0.629, 0.229, 0.084, 0.059]
panx_ch = defaultdict(DatasetDict)
for lang, frac in zip(langs, fracs):
 # Load monolingual corpus
  ds = load_dataset("xtreme", name=f"PAN-X.{lang}")
  # Shuffle and downsample each split according to spoken proportion
  for split in ds:
      panx_ch[lang][split] = (
      ds[split]
      .shuffle(seed=0)
      .select(range(int(frac * ds[split].num_rows))))

element = panx_ch["de"]["train"][0]
for key, value in element.items():
 print(f"{key}: {value}")
for key, value in panx_ch["de"]["train"].features.items():
 print(f"{key}: {value}")
tags = panx_ch["de"]["train"].features["ner_tags"].feature
panx_de = panx_ch["de"].map(create_tag_names)
class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
    config_class = XLMRobertaConfig
    def __init__(self, config):
      super().__init__(config)
      self.num_labels = config.num_labels
      # Load model body
      self.roberta = RobertaModel(config, add_pooling_layer=False)
      # Set up token classification head
      self.dropout = nn.Dropout(config.hidden_dropout_prob)
      self.classifier = nn.Linear(config.hidden_size, config.num_labels)
      # Load and initialize weights
      self.init_weights()
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
      labels=None, **kwargs):
      # Use model body to get encoder representations
      outputs = self.roberta(input_ids, attention_mask=attention_mask,
      token_type_ids=token_type_ids, **kwargs)
      # Apply classifier to encoder representation
      sequence_output = self.dropout(outputs[0])
      logits = self.classifier(sequence_output)
      # Calculate losses
      loss = None
      if labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      # Return model output object
      return TokenClassifierOutput(loss=loss, logits=logits,hidden_states=outputs.hidden_states,attentions=outputs.attention)
index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

xlmr_config = AutoConfig.from_pretrained(xlmr_model_name,
 num_labels=tags.num_classes,
 id2label=index2tag, label2id=tag2index)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xlmr_model = (XLMRobertaForTokenClassification
 .from_pretrained(xlmr_model_name, config=xlmr_config)
 .to(device))


input_ids = xlmr_tokenizer.encode(text, return_tensors="pt")
pd.DataFrame([xlmr_tokens, input_ids[0].numpy()], index=["Tokens", "Input IDs"])

print(pd)

words, labels = de_example["tokens"], de_example["ner_tags"]


tokenized_input = xlmr_tokenizer(de_example["tokens"], is_split_into_words=True)
tokens = xlmr_tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
pd.DataFrame([tokens], index=["Tokens"])








