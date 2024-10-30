from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def average_embedding(desc):
    # Split description by tags (e.g., commas) and compute embeddings for each tag
    tags = [tag.strip() for tag in desc.split(',')]
    tag_embeddings = model.encode(tags)
    
    # Compute the average of all tag embeddings
    avg_embedding = np.mean(tag_embeddings, axis=0)
    
    return avg_embedding

def cosine_similarity_score(ground_truth_desc, predicted_desc):
    ground_truth_embedding = average_embedding(ground_truth_desc)
    predicted_embedding = average_embedding(predicted_desc)
    
    # Calculate cosine similarity between the averaged embeddings
    cosine_sim = cosine_similarity([ground_truth_embedding], [predicted_embedding])[0][0]

    return cosine_sim

if __name__ == '__main__':
    ground_truth = "red hat"
    predicted = "cap, crimson"

    score = cosine_similarity_score(ground_truth, predicted)
    print(f"Cosine Similarity Score: {score:.2f}")
