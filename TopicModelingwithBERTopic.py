#r Text Clustering
"""1. Convert the input documents to embeddings with an embedding
model.
2. Reduce the dimensionality of embeddings with a dimensionality
reduction model.
3. Find groups of semantically similar documents with a cluster
model."""

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
# Train our model with our previously defined models
topic_model = BERTopic(
 embedding_model=embedding_model,
 umap_model=umap_model,
 hdbscan_model=hdbscan_model,
 verbose=True
).fit(abstracts, embeddings)

topic_model.get_topic_info()
topic_model.get_topic(0)
topic_model.find_topics("topic modeling")
topic_model.get_topic(22)
topic_model.topics_[titles.index("BERTopic: Neural topic modeling with a class-based TF-IDF procedure")]

fig = topic_model.visualize_documents(
 titles,
 reduced_embeddings=reduced_embeddings,
 width=1200,
 hide_annotations=True
)
# Update fonts of legend for easier visualization
fig.update_layout(font=dict(size=16))


