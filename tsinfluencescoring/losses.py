"""
Loss functions for training the timestamp selector.

Includes task consistency loss, diversity loss (DPP), and mutual information loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class TaskConsistencyLoss(nn.Module):
    """
    Task consistency loss to ensure selected timestamps are useful for the task.
    
    This loss encourages the selector to choose timestamps that help
    the downstream model make accurate predictions.
    """
    
    def __init__(self, task_loss_fn: Optional[Callable] = None):
        """
        Initialize task consistency loss.
        
        Args:
            task_loss_fn: Custom task loss function. If None, uses MSE for regression
        """
        super().__init__()
        self.task_loss_fn = task_loss_fn or nn.MSELoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute task consistency loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            mask: Optional mask to weight the loss
            
        Returns:
            Task loss value
        """
        loss = self.task_loss_fn(predictions, targets)
        
        if mask is not None:
            # Weight loss by selection density (encourage meaningful selections)
            selection_weight = mask.mean() + 1e-8
            loss = loss / selection_weight
            
        return loss


class DiversityLoss(nn.Module):
    """
    Diversity loss based on Determinantal Point Process (DPP).
    
    Encourages the selector to choose diverse timestamps rather than
    redundant ones, promoting better coverage of the time series.
    """
    
    def __init__(self, sigma: float = 1.0):
        """
        Initialize diversity loss.
        
        Args:
            sigma: Bandwidth parameter for the kernel
        """
        super().__init__()
        self.sigma = sigma
    
    def compute_kernel_matrix(
        self, 
        features: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RBF kernel matrix for selected features.
        
        Args:
            features: Feature tensor of shape (batch_size, seq_len, feature_dim)
            mask: Selection mask of shape (batch_size, seq_len)
            
        Returns:
            Kernel matrix weighted by selection mask
        """
        batch_size, seq_len, feature_dim = features.shape
        
        # Compute pairwise distances
        # features: (batch, seq_len, dim)
        features_expanded_1 = features.unsqueeze(2)  # (batch, seq_len, 1, dim)
        features_expanded_2 = features.unsqueeze(1)  # (batch, 1, seq_len, dim)
        
        # Squared Euclidean distance
        distances = torch.sum((features_expanded_1 - features_expanded_2) ** 2, dim=-1)
        
        # RBF kernel
        kernel = torch.exp(-distances / (2 * self.sigma ** 2))
        
        # Weight by selection mask
        mask_expanded_1 = mask.unsqueeze(2)  # (batch, seq_len, 1)
        mask_expanded_2 = mask.unsqueeze(1)  # (batch, 1, seq_len)
        mask_product = mask_expanded_1 * mask_expanded_2  # (batch, seq_len, seq_len)
        
        weighted_kernel = kernel * mask_product
        
        return weighted_kernel
    
    def forward(
        self, 
        features: torch.Tensor, 
        mask: torch.Tensor,
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute diversity loss using DPP log-determinant.
        
        Args:
            features: Feature representations (batch_size, seq_len, feature_dim)
            mask: Selection mask (batch_size, seq_len)
            epsilon: Small constant for numerical stability
            
        Returns:
            Negative log-determinant (to maximize diversity)
        """
        # Compute kernel matrix
        kernel = self.compute_kernel_matrix(features, mask)
        
        # Add small diagonal for numerical stability
        batch_size, seq_len = kernel.shape[0], kernel.shape[1]
        eye = torch.eye(seq_len, device=kernel.device).unsqueeze(0)
        kernel = kernel + epsilon * eye
        
        # Compute log determinant (log det promotes diversity)
        # For stability, use Cholesky decomposition
        try:
            chol = torch.linalg.cholesky(kernel)
            log_det = 2 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)), dim=-1)
        except RuntimeError:
            # Fallback to eigenvalues if Cholesky fails
            eigenvalues = torch.linalg.eigvalsh(kernel)
            log_det = torch.sum(torch.log(eigenvalues + epsilon), dim=-1)
        
        # Return negative log-det as loss (we want to maximize diversity)
        diversity_loss = -log_det.mean()
        
        return diversity_loss


class MutualInformationLoss(nn.Module):
    """
    Mutual information loss between selected timestamps and targets.
    
    Encourages selection of timestamps that contain maximal information
    about the prediction target.
    """
    
    def __init__(self, hidden_dim: int = 64):
        """
        Initialize mutual information loss.
        
        Args:
            hidden_dim: Hidden dimension for MI estimation network
        """
        super().__init__()
        
        # Simple MI estimation using a discriminator network
        # This is a simplified version - more sophisticated methods exist
        self.mi_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        selected_features: torch.Tensor,
        target_features: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate mutual information using contrastive learning.
        
        Args:
            selected_features: Features from selected timestamps (batch, seq_len, dim)
            target_features: Target features to predict (batch, dim)
            mask: Selection mask (batch, seq_len)
            
        Returns:
            Negative mutual information estimate (to maximize MI)
        """
        batch_size = selected_features.shape[0]
        
        # Aggregate selected features using mask
        mask_expanded = mask.unsqueeze(-1)
        masked_features = selected_features * mask_expanded
        aggregated = masked_features.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        
        # Positive samples: true pairs
        positive_pairs = torch.cat([aggregated, target_features], dim=-1)
        positive_scores = self.mi_estimator(positive_pairs)
        
        # Negative samples: shuffled pairs
        shuffled_idx = torch.randperm(batch_size)
        negative_pairs = torch.cat([aggregated, target_features[shuffled_idx]], dim=-1)
        negative_scores = self.mi_estimator(negative_pairs)
        
        # InfoNCE-style loss
        mi_estimate = positive_scores.mean() - torch.log(
            torch.exp(negative_scores).mean() + 1e-8
        )
        
        # Return negative MI as loss (we want to maximize MI)
        return -mi_estimate


class CombinedLoss(nn.Module):
    """
    Combined loss function for training the timestamp selector.
    
    Combines task consistency, diversity, and optional mutual information losses
    with configurable weights.
    """
    
    def __init__(
        self,
        task_loss_fn: Optional[Callable] = None,
        alpha_task: float = 1.0,
        alpha_diversity: float = 0.1,
        alpha_mi: float = 0.05,
        use_mi: bool = False,
        sigma: float = 1.0,
        mi_hidden_dim: int = 64
    ):
        """
        Initialize combined loss.
        
        Args:
            task_loss_fn: Task-specific loss function
            alpha_task: Weight for task consistency loss
            alpha_diversity: Weight for diversity loss
            alpha_mi: Weight for mutual information loss
            use_mi: Whether to include MI loss
            sigma: Sigma parameter for diversity loss
            mi_hidden_dim: Hidden dimension for MI estimator
        """
        super().__init__()
        
        self.task_loss = TaskConsistencyLoss(task_loss_fn)
        self.diversity_loss = DiversityLoss(sigma=sigma)
        self.mi_loss = MutualInformationLoss(hidden_dim=mi_hidden_dim) if use_mi else None
        
        self.alpha_task = alpha_task
        self.alpha_diversity = alpha_diversity
        self.alpha_mi = alpha_mi
        self.use_mi = use_mi
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
        target_features: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            features: Feature representations for diversity
            mask: Selection mask
            target_features: Optional target features for MI loss
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Task consistency loss
        task_loss_val = self.task_loss(predictions, targets, mask)
        
        # Diversity loss
        diversity_loss_val = self.diversity_loss(features, mask)
        
        # Combined loss
        total_loss = (
            self.alpha_task * task_loss_val +
            self.alpha_diversity * diversity_loss_val
        )
        
        loss_dict = {
            "task_loss": task_loss_val.item(),
            "diversity_loss": diversity_loss_val.item(),
            "total_loss": total_loss.item()
        }
        
        # Optional mutual information loss
        if self.use_mi and self.mi_loss is not None and target_features is not None:
            mi_loss_val = self.mi_loss(features, target_features, mask)
            total_loss = total_loss + self.alpha_mi * mi_loss_val
            loss_dict["mi_loss"] = mi_loss_val.item()
        
        return total_loss, loss_dict
