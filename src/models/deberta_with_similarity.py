"""
DeBERTa Model with Similarity Features

This module extends the DeBERTa model to incorporate similarity features
calculated using sentence transformers. The similarity features are combined
with the DeBERTa embeddings using a multi-head attention mechanism.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional


class DeBERTaWithSimilarityFeatures(nn.Module):
    """
    DeBERTa model enhanced with similarity features.
    
    Architecture:
    1. DeBERTa base model processes the concatenated sequence
    2. Similarity features (5 features) are projected to match embedding dimension
    3. Multi-head attention combines DeBERTa embeddings with similarity features
    4. Classification head outputs 3 classes (A wins, B wins, Tie)
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 3,
        num_similarity_features: int = 5,
        hidden_dropout_prob: float = 0.1,
        similarity_feature_dim: int = 128,
    ):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the base DeBERTa model.
            num_labels: Number of classification labels (3: A wins, B wins, Tie).
            num_similarity_features: Number of similarity features (5).
            hidden_dropout_prob: Dropout probability.
            similarity_feature_dim: Dimension for projected similarity features.
        """
        super().__init__()
        
        # Base DeBERTa model
        self.deberta = AutoModel.from_pretrained(model_name)
        config = self.deberta.config
        hidden_size = config.hidden_size
        
        # Project similarity features to a higher dimension
        self.similarity_projection = nn.Sequential(
            nn.Linear(num_similarity_features, similarity_feature_dim),
            nn.LayerNorm(similarity_feature_dim),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(similarity_feature_dim, hidden_size),
        )
        
        # Multi-head attention to combine DeBERTa embeddings with similarity features
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=hidden_dropout_prob,
            batch_first=True,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(hidden_size // 2, num_labels),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for new layers."""
        for module in [self.similarity_projection, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        similarity_features: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs from tokenizer.
            attention_mask: Attention mask.
            similarity_features: Tensor of shape (batch_size, num_similarity_features).
            token_type_ids: Token type IDs (optional).
            labels: Ground truth labels (optional, for training).
            
        Returns:
            If labels provided: (loss, logits)
            Otherwise: logits
        """
        # Get DeBERTa embeddings
        deberta_outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # Use [CLS] token embedding (first token)
        cls_embedding = deberta_outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Project similarity features
        if similarity_features is not None:
            similarity_proj = self.similarity_projection(similarity_features)  # (batch_size, hidden_size)
            
            # Use attention to combine DeBERTa embeddings with similarity features
            # Query: DeBERTa embeddings, Key/Value: similarity features
            cls_embedding_expanded = cls_embedding.unsqueeze(1)  # (batch_size, 1, hidden_size)
            similarity_proj_expanded = similarity_proj.unsqueeze(1)  # (batch_size, 1, hidden_size)
            
            # Self-attention: combine both
            combined, _ = self.attention(
                cls_embedding_expanded,
                similarity_proj_expanded,
                similarity_proj_expanded,
            )
            combined = combined.squeeze(1)  # (batch_size, hidden_size)
            
            # Residual connection
            final_embedding = cls_embedding + combined
        else:
            # No similarity features provided, use only DeBERTa embeddings
            final_embedding = cls_embedding
        
        # Classification
        logits = self.classifier(final_embedding)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return (loss, logits)
        
        return logits


# For compatibility with Hugging Face Trainer
class DeBERTaWithSimilarityForSequenceClassification(nn.Module):
    """
    Wrapper class compatible with Hugging Face Trainer API.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 3,
        num_similarity_features: int = 5,
        hidden_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.model = DeBERTaWithSimilarityFeatures(
            model_name=model_name,
            num_labels=num_labels,
            num_similarity_features=num_similarity_features,
            hidden_dropout_prob=hidden_dropout_prob,
        )
        self.num_labels = num_labels
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        similarity_features: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass compatible with Trainer API.
        
        Returns:
            ModelOutput with logits (and loss if labels provided)
        """
        from transformers.modeling_outputs import SequenceClassifierOutput
        
        if labels is not None:
            loss, logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                similarity_features=similarity_features,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
            )
        else:
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                similarity_features=similarity_features,
                token_type_ids=token_type_ids,
                labels=None,
            )
            return SequenceClassifierOutput(logits=logits)

