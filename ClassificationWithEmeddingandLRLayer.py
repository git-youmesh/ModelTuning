from datasets import load_dataset
from transformers import pipeline
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
# Load our data
# Path to our HF model
def evaluate_performance(y_true, y_pred):
 """Create and print the classification report"""
 performance = classification_report(
 y_true, y_pred,
 target_names=["Negative Review", "Positive Review"]
 )
 print(performance)

model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
data = load_dataset("rotten_tomatoes")
# Load model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
# Convert text to embeddings
train_embeddings = model.encode(data["train"]["text"],show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"],show_progress_bar=True)

# Train a logistic regression on our train embeddings
clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, data["train"]["label"])
# Predict previously unseen instances
y_pred = clf.predict(test_embeddings)
evaluate_performance(data["test"]["label"], y_pred)
