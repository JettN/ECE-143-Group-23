"""
Similarity Features Module

This module calculates semantic similarity features between prompts and responses
using sentence transformers. These features can be used to enhance the model's
ability to predict which response is better.

Features calculated:
1. prompt_response_a_similarity: Cosine similarity between prompt and response_a
2. prompt_response_b_similarity: Cosine similarity between prompt and response_b
3. response_a_response_b_similarity: Cosine similarity between response_a and response_b
4. similarity_diff: Difference between prompt-response similarities (a - b)
5. similarity_ratio: Ratio of similarities (a / b, with smoothing)

These features provide additional semantic information that the model can use
to make better predictions.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import torch


class SimilarityFeatureCalculator:
    """
    Calculates similarity features using sentence transformers.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize the similarity feature calculator.
        
        Args:
            model_name: Name of the sentence transformer model to use.
                       "all-MiniLM-L6-v2" is fast and lightweight.
            device: Device to run the model on (None = auto-detect).
            batch_size: Batch size for encoding.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.batch_size = batch_size
        print(f"Loading sentence transformer model: {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"✓ Model loaded")
    
    def calculate_features(
        self,
        prompts: List[str],
        responses_a: List[str],
        responses_b: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Calculate similarity features for a batch of examples.
        
        Args:
            prompts: List of prompt strings.
            responses_a: List of response A strings.
            responses_b: List of response B strings.
            
        Returns:
            Dictionary with similarity features as numpy arrays:
            - prompt_response_a_sim: Cosine similarity between prompt and response_a
            - prompt_response_b_sim: Cosine similarity between prompt and response_b
            - response_a_response_b_sim: Cosine similarity between response_a and response_b
            - similarity_diff: prompt_response_a_sim - prompt_response_b_sim
            - similarity_ratio: prompt_response_a_sim / (prompt_response_b_sim + 1e-8)
        """
        # Encode all texts
        all_texts = prompts + responses_a + responses_b
        embeddings = self.model.encode(
            all_texts,
            batch_size=self.batch_size,
            show_progress_bar=len(all_texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity
        )
        
        n = len(prompts)
        prompt_embeds = embeddings[:n]
        resp_a_embeds = embeddings[n:2*n]
        resp_b_embeds = embeddings[2*n:3*n]
        
        # Calculate cosine similarities (embeddings are already normalized)
        prompt_resp_a_sim = np.sum(prompt_embeds * resp_a_embeds, axis=1)
        prompt_resp_b_sim = np.sum(prompt_embeds * resp_b_embeds, axis=1)
        resp_a_resp_b_sim = np.sum(resp_a_embeds * resp_b_embeds, axis=1)
        
        # Additional features
        similarity_diff = prompt_resp_a_sim - prompt_resp_b_sim
        similarity_ratio = prompt_resp_a_sim / (prompt_resp_b_sim + 1e-8)
        
        return {
            "prompt_response_a_sim": prompt_resp_a_sim.astype(np.float32),
            "prompt_response_b_sim": prompt_resp_b_sim.astype(np.float32),
            "response_a_response_b_sim": resp_a_resp_b_sim.astype(np.float32),
            "similarity_diff": similarity_diff.astype(np.float32),
            "similarity_ratio": similarity_ratio.astype(np.float32),
        }
    
    def calculate_features_for_dataframe(
        self,
        df,
        prompt_col: str = "prompt",
        response_a_col: str = "response_a",
        response_b_col: str = "response_b",
    ) -> Dict[str, np.ndarray]:
        """
        Calculate similarity features for a pandas DataFrame.
        
        Args:
            df: DataFrame with prompt, response_a, and response_b columns.
            prompt_col: Name of the prompt column.
            response_a_col: Name of the response_a column.
            response_b_col: Name of the response_b column.
            
        Returns:
            Dictionary with similarity features as numpy arrays.
        """
        prompts = df[prompt_col].astype(str).tolist()
        responses_a = df[response_a_col].astype(str).tolist()
        responses_b = df[response_b_col].astype(str).tolist()
        
        return self.calculate_features(prompts, responses_a, responses_b)


def add_similarity_features_to_dataframe(
    df,
    similarity_calculator: Optional[SimilarityFeatureCalculator] = None,
    prompt_col: str = "prompt",
    response_a_col: str = "response_a",
    response_b_col: str = "response_b",
) -> tuple:
    """
    Add similarity features to a DataFrame.
    
    Args:
        df: DataFrame to add features to.
        similarity_calculator: Pre-initialized calculator (creates new one if None).
        prompt_col: Name of the prompt column.
        response_a_col: Name of the response_a column.
        response_b_col: Name of the response_b column.
        
    Returns:
        Tuple of (df_with_features, similarity_calculator)
    """
    if similarity_calculator is None:
        similarity_calculator = SimilarityFeatureCalculator()
    
    print("Calculating similarity features...")
    features = similarity_calculator.calculate_features_for_dataframe(
        df, prompt_col, response_a_col, response_b_col
    )
    
    # Add features to DataFrame
    df_with_features = df.copy()
    for feature_name, feature_values in features.items():
        df_with_features[feature_name] = feature_values
    
    print(f"✓ Added {len(features)} similarity features to DataFrame")
    
    return df_with_features, similarity_calculator

