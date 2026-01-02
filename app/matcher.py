from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Dict, Tuple
import pickle
import os


class ResumeJobMatcher:
    """
    Matcher class for computing semantic similarity between resumes and jobs
    using embeddings and FAISS for efficient similarity search.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the matcher with an embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.job_descriptions = []
        self.job_metadata = []
    
    def compute_similarity(self, resume_text: str, job_text: str) -> float:
        """
        Compute semantic similarity between resume and job description.
        
        Args:
            resume_text: Resume text
            job_text: Job description text
            
        Returns:
            Similarity score between 0 and 1
        """
        embeddings = self.model.encode([resume_text, job_text])
        resume_embedding = embeddings[0]
        job_embedding = embeddings[1]

        # Cosine similarity
        similarity = np.dot(resume_embedding, job_embedding) / (
            np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
        )

        return float(similarity)
    
    def build_job_index(self, job_descriptions: List[str], job_metadata: List[Dict] = None):
        """
        Build a FAISS index for efficient similarity search across multiple jobs.
        
        Args:
            job_descriptions: List of job description texts
            job_metadata: Optional list of metadata dictionaries for each job
        """
        if not job_descriptions:
            raise ValueError("job_descriptions cannot be empty")
        
        self.job_descriptions = job_descriptions
        self.job_metadata = job_metadata or [{}] * len(job_descriptions)
        
        # Generate embeddings for all job descriptions
        job_embeddings = self.model.encode(job_descriptions, show_progress_bar=True)
        dimension = job_embeddings.shape[1]
        
        # Create FAISS index (using L2 distance, we'll convert to cosine similarity)
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(job_embeddings)
        
        # Add embeddings to index
        self.index.add(job_embeddings.astype('float32'))
    
    def find_top_matches(
        self, 
        resume_text: str, 
        top_k: int = 5
    ) -> List[Tuple[int, float, str, Dict]]:
        """
        Find top K matching jobs for a resume.
        
        Args:
            resume_text: Resume text
            top_k: Number of top matches to return
            
        Returns:
            List of tuples: (job_index, similarity_score, job_text, metadata)
        """
        if self.index is None:
            raise ValueError("Job index not built. Call build_job_index() first.")
        
        # Generate embedding for resume
        resume_embedding = self.model.encode([resume_text])
        faiss.normalize_L2(resume_embedding)
        
        # Search in FAISS index
        similarities, indices = self.index.search(
            resume_embedding.astype('float32'), 
            min(top_k, len(self.job_descriptions))
        )
        
        # Format results
        results = []
        for i, (idx, sim) in enumerate(zip(indices[0], similarities[0])):
            if idx < len(self.job_descriptions):
                results.append((
                    int(idx),
                    float(sim),
                    self.job_descriptions[int(idx)],
                    self.job_metadata[int(idx)]
                ))
        
        return results
    
    def rank_jobs(
        self,
        resume_text: str,
        job_descriptions: List[str],
        job_metadata: List[Dict] = None
    ) -> List[Tuple[int, float, str, Dict]]:
        """
        Rank multiple jobs against a resume without building an index.
        Useful for small numbers of jobs.
        
        Args:
            resume_text: Resume text
            job_descriptions: List of job description texts
            job_metadata: Optional list of metadata dictionaries
            
        Returns:
            List of tuples sorted by similarity: (job_index, similarity_score, job_text, metadata)
        """
        if not job_descriptions:
            return []
        
        job_metadata = job_metadata or [{}] * len(job_descriptions)
        
        # Generate embeddings
        texts = [resume_text] + job_descriptions
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        resume_embedding = embeddings[0]
        job_embeddings = embeddings[1:]
        
        # Compute similarities
        similarities = []
        for i, job_emb in enumerate(job_embeddings):
            similarity = np.dot(resume_embedding, job_emb) / (
                np.linalg.norm(resume_embedding) * np.linalg.norm(job_emb)
            )
            similarities.append((i, float(similarity), job_descriptions[i], job_metadata[i]))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def save_index(self, filepath: str):
        """Save the FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        faiss.write_index(self.index, filepath + ".index")
        with open(filepath + ".metadata", "wb") as f:
            pickle.dump({
                'job_descriptions': self.job_descriptions,
                'job_metadata': self.job_metadata
            }, f)
    
    def load_index(self, filepath: str):
        """Load the FAISS index and metadata from disk."""
        self.index = faiss.read_index(filepath + ".index")
        with open(filepath + ".metadata", "rb") as f:
            data = pickle.load(f)
            self.job_descriptions = data['job_descriptions']
            self.job_metadata = data['job_metadata']


# Global instance for backward compatibility
_matcher_instance = None

def get_matcher() -> ResumeJobMatcher:
    """Get or create global matcher instance."""
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = ResumeJobMatcher()
    return _matcher_instance

def compute_similarity(resume_text: str, job_text: str) -> float:
    """
    Compute semantic similarity between resume and job description.
    Backward compatibility function.
    """
    matcher = get_matcher()
    return matcher.compute_similarity(resume_text, job_text)
