from datasets import load_dataset
from transformers import pipeline
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
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
print(data["train"]["label"])
# Load model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
# Convert text to embeddings
train_embeddings = model.encode(data["train"]["text"],show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"],show_progress_bar=True)
label_embeddings = model.encode(["A negative review", "Apositive review"])
sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
y_pred = np.argmax(sim_matrix, axis=1)
evaluate_performance(data["test"]["label"], y_pred)
