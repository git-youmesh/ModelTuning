from transformers import pipeline
from bertopic.representation import TextGeneration
from datasets import load_dataset
from transformers import pipeline
import numpy as np
from tqdm import tqdm
from bertopic import BERTopic
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd
from copy import deepcopy
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance


prompt = """I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: '[KEYWORDS]'.
Based on the documents and keywords, what is this topic about?"""
# Update our topic representations using Flan-T5

#from datasets import load_dataset
dataset = load_dataset("maartengr/arxiv_nlp")["train"]
# Extract metadata
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]



embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts,
show_progress_bar=True)
umap_model = UMAP(
 n_components=5, min_dist=0.0, metric='cosine',
random_state=42
)
reduced_embeddings = umap_model.fit_transform(embeddings)


generator = pipeline("text2text-generation", model="google/flant5-small")
representation_model = TextGeneration(
 generator, prompt=prompt, doc_length=50,
tokenizer="whitespace"
)

topic_model = BERTopic(
 embedding_model=embedding_model,
 umap_model=umap_model,
 hdbscan_model=hdbscan_model,
 verbose=True
).fit(abstracts, embeddings)

topic_model.update_topics(abstracts,representation_model=representation_model)

topic_differences(topic_model, original_topics)


