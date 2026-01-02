from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_similarity(resume_text: str, job_text: str) -> float:
    """
    Compute semantic similarity between resume and job description.
    """
    embeddings = model.encode([resume_text, job_text])
    resume_embedding = embeddings[0]
    job_embedding = embeddings[1]

    # Cosine similarity
    similarity = np.dot(resume_embedding, job_embedding) / (
        np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
    )

    return float(similarity)
