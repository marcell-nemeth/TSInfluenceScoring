"""
Differentiable timestamp selector module with attention-based scoring.

This module implements a flexible selector that can score and select
timestamps from time-series data using attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class AttentionScorer(nn.Module):
    """
    Attention-based scoring mechanism for timestamps.
    
    This module computes attention scores over time dimensions,
    allowing the model to identify influential timestamps.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize the attention scorer.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden representations
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Project input to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Score computation
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores for timestamps.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            scores: Tensor of shape (batch_size, seq_len) with timestamp scores
        """
        # Project to hidden dimension
        h = self.input_projection(x)  # (batch, seq_len, hidden_dim)
        h = self.layer_norm(h)
        
        # Apply self-attention
        attn_output, _ = self.attention(h, h, h)
        attn_output = self.dropout(attn_output)
        
        # Residual connection
        h = h + attn_output
        h = self.layer_norm(h)
        
        # Compute scores
        scores = self.score_layer(h).squeeze(-1)  # (batch, seq_len)
        
        return scores


class GumbelSoftmaxSelector(nn.Module):
    """
    Differentiable selection using Gumbel-Softmax trick.
    
    Allows for sparse selection while maintaining differentiability.
    """
    
    def __init__(self, temperature: float = 1.0, hard: bool = False):
        """
        Initialize Gumbel-Softmax selector.
        
        Args:
            temperature: Temperature for Gumbel-Softmax (lower = more discrete)
            hard: If True, returns one-hot vectors (non-differentiable forward, differentiable backward)
        """
        super().__init__()
        self.temperature = temperature
        self.hard = hard
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply Gumbel-Softmax to logits.
        
        Args:
            logits: Input logits of shape (batch_size, seq_len)
            
        Returns:
            Soft or hard selection probabilities
        """
        return F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard, dim=-1)


class TopKSelector(nn.Module):
    """
    Top-K selection mechanism with straight-through estimator.
    
    Selects top-k timestamps while maintaining differentiability.
    """
    
    def __init__(self, k: int, hard: bool = True):
        """
        Initialize Top-K selector.
        
        Args:
            k: Number of timestamps to select
            hard: If True, returns binary mask
        """
        super().__init__()
        self.k = k
        self.hard = hard
        
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Select top-k timestamps based on scores.
        
        Args:
            scores: Score tensor of shape (batch_size, seq_len)
            
        Returns:
            Selection mask or probabilities
        """
        batch_size, seq_len = scores.shape
        k = min(self.k, seq_len)
        
        # Get top-k indices
        _, top_indices = torch.topk(scores, k, dim=-1)
        
        if self.hard:
            # Create binary mask
            mask = torch.zeros_like(scores)
            mask.scatter_(-1, top_indices, 1.0)
            
            # Straight-through estimator: use soft scores for backward pass
            # mask_ste = mask - scores.detach() + scores
            # For now, just return the mask with gradients from scores
            return mask + (scores - scores.detach())
        else:
            # Soft selection using softmax over scores
            soft_mask = torch.zeros_like(scores)
            soft_scores = F.softmax(scores, dim=-1)
            soft_mask.scatter_(-1, top_indices, soft_scores.gather(-1, top_indices))
            return soft_mask


class TimestampSelector(nn.Module):
    """
    Main differentiable selector module for influential timestamps.
    
    This module combines attention-based scoring with flexible selection
    mechanisms (Gumbel-Softmax, Top-K) to identify and select influential
    timestamps from time-series data.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        selection_method: str = "topk",
        k: Optional[int] = None,
        temperature: float = 1.0,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize the timestamp selector.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden representations
            num_heads: Number of attention heads
            selection_method: Selection method ('topk', 'gumbel', or 'threshold')
            k: Number of timestamps to select (for topk method)
            temperature: Temperature for Gumbel-Softmax
            dropout: Dropout rate
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.selection_method = selection_method
        self.k = k
        
        # Attention-based scorer
        self.scorer = AttentionScorer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Selection mechanism
        if selection_method == "gumbel":
            self.selector = GumbelSoftmaxSelector(temperature=temperature, hard=True)
        elif selection_method == "topk":
            if k is None:
                raise ValueError("k must be specified for topk selection method")
            self.selector = TopKSelector(k=k, hard=True)
        elif selection_method == "threshold":
            self.selector = None  # Will use threshold in forward
            self.threshold = kwargs.get("threshold", 0.5)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_scores: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Select influential timestamps from input time-series.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            return_scores: If True, return raw scores along with mask
            
        Returns:
            mask: Binary or soft selection mask of shape (batch_size, seq_len)
            scores: (Optional) Raw scores if return_scores=True
        """
        # Compute scores
        scores = self.scorer(x)  # (batch_size, seq_len)
        
        # Apply selection
        if self.selection_method == "threshold":
            # Threshold-based selection with sigmoid
            probs = torch.sigmoid(scores)
            mask = (probs > self.threshold).float()
            # Straight-through estimator
            mask = mask + (probs - probs.detach())
        else:
            mask = self.selector(scores)
        
        if return_scores:
            return mask, scores
        return mask, None
    
    def get_selected_timestamps(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract selected timestamps using the mask.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Selection mask of shape (batch_size, seq_len)
            
        Returns:
            Selected timestamps with shape (batch_size, seq_len, input_dim)
            where unselected positions are zeroed out
        """
        # Expand mask to match input dimensions
        mask_expanded = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        return x * mask_expanded
    
    def compute_selection_stats(self, mask: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics about the selection.
        
        Args:
            mask: Selection mask of shape (batch_size, seq_len)
            
        Returns:
            Dictionary with selection statistics
        """
        with torch.no_grad():
            binary_mask = (mask > 0.5).float()
            num_selected = binary_mask.sum(dim=-1).mean().item()
            selection_ratio = binary_mask.mean().item()
            
        return {
            "num_selected": num_selected,
            "selection_ratio": selection_ratio,
            "mean_score": mask.mean().item(),
            "max_score": mask.max().item(),
            "min_score": mask.min().item()
        }
