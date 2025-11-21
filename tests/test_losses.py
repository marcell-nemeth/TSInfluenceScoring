"""Tests for the loss functions module."""

import pytest
import torch
import torch.nn as nn
from tsinfluencescoring.losses import (
    TaskConsistencyLoss,
    DiversityLoss,
    MutualInformationLoss,
    CombinedLoss
)


def test_task_consistency_loss():
    """Test TaskConsistencyLoss module."""
    batch_size, output_dim = 4, 3
    
    loss_fn = TaskConsistencyLoss()
    predictions = torch.randn(batch_size, output_dim)
    targets = torch.randn(batch_size, output_dim)
    
    loss = loss_fn(predictions, targets)
    
    assert loss.item() >= 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_task_consistency_loss_with_mask():
    """Test TaskConsistencyLoss with selection mask."""
    batch_size, seq_len, output_dim = 4, 20, 3
    
    loss_fn = TaskConsistencyLoss()
    predictions = torch.randn(batch_size, output_dim)
    targets = torch.randn(batch_size, output_dim)
    mask = torch.rand(batch_size, seq_len)
    
    loss = loss_fn(predictions, targets, mask)
    
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_diversity_loss():
    """Test DiversityLoss module."""
    batch_size, seq_len, feature_dim = 4, 20, 10
    
    loss_fn = DiversityLoss(sigma=1.0)
    features = torch.randn(batch_size, seq_len, feature_dim)
    mask = torch.rand(batch_size, seq_len)
    mask = (mask > 0.5).float()  # Binary mask
    
    loss = loss_fn(features, mask)
    
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_diversity_loss_kernel():
    """Test kernel matrix computation in DiversityLoss."""
    batch_size, seq_len, feature_dim = 2, 10, 5
    
    loss_fn = DiversityLoss(sigma=1.0)
    features = torch.randn(batch_size, seq_len, feature_dim)
    mask = torch.ones(batch_size, seq_len)
    
    kernel = loss_fn.compute_kernel_matrix(features, mask)
    
    assert kernel.shape == (batch_size, seq_len, seq_len)
    # Kernel should be symmetric
    assert torch.allclose(kernel, kernel.transpose(-2, -1), atol=1e-5)
    # Diagonal should be close to 1 (RBF kernel property)
    diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)


def test_mutual_information_loss():
    """Test MutualInformationLoss module."""
    batch_size, seq_len, hidden_dim = 4, 20, 64
    
    loss_fn = MutualInformationLoss(hidden_dim=hidden_dim)
    selected_features = torch.randn(batch_size, seq_len, hidden_dim)
    target_features = torch.randn(batch_size, hidden_dim)
    mask = torch.rand(batch_size, seq_len)
    
    loss = loss_fn(selected_features, target_features, mask)
    
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_combined_loss():
    """Test CombinedLoss module."""
    batch_size, seq_len, feature_dim = 4, 20, 10
    output_dim = 3
    
    loss_fn = CombinedLoss(
        task_loss_fn=nn.MSELoss(),
        alpha_task=1.0,
        alpha_diversity=0.1,
        alpha_mi=0.05,
        use_mi=False
    )
    
    predictions = torch.randn(batch_size, output_dim)
    targets = torch.randn(batch_size, output_dim)
    features = torch.randn(batch_size, seq_len, feature_dim)
    mask = torch.rand(batch_size, seq_len)
    
    total_loss, loss_dict = loss_fn(predictions, targets, features, mask)
    
    assert total_loss.item() >= 0
    assert "task_loss" in loss_dict
    assert "diversity_loss" in loss_dict
    assert "total_loss" in loss_dict
    assert not torch.isnan(total_loss)


def test_combined_loss_with_mi():
    """Test CombinedLoss with mutual information loss."""
    batch_size, seq_len, feature_dim = 4, 20, 10
    output_dim = 3
    hidden_dim = 64
    
    loss_fn = CombinedLoss(
        task_loss_fn=nn.MSELoss(),
        alpha_task=1.0,
        alpha_diversity=0.1,
        alpha_mi=0.05,
        use_mi=True,
        mi_hidden_dim=hidden_dim
    )
    
    predictions = torch.randn(batch_size, output_dim)
    targets = torch.randn(batch_size, output_dim)
    features = torch.randn(batch_size, seq_len, hidden_dim)
    mask = torch.rand(batch_size, seq_len)
    target_features = torch.randn(batch_size, hidden_dim)
    
    total_loss, loss_dict = loss_fn(predictions, targets, features, mask, target_features)
    
    assert total_loss.item() >= 0
    assert "task_loss" in loss_dict
    assert "diversity_loss" in loss_dict
    assert "mi_loss" in loss_dict
    assert "total_loss" in loss_dict


def test_combined_loss_backward():
    """Test that combined loss is differentiable."""
    batch_size, seq_len, feature_dim = 4, 20, 10
    output_dim = 3
    
    loss_fn = CombinedLoss(alpha_task=1.0, alpha_diversity=0.1)
    
    predictions = torch.randn(batch_size, output_dim, requires_grad=True)
    targets = torch.randn(batch_size, output_dim)
    features = torch.randn(batch_size, seq_len, feature_dim, requires_grad=True)
    mask = torch.rand(batch_size, seq_len)
    
    total_loss, _ = loss_fn(predictions, targets, features, mask)
    total_loss.backward()
    
    assert predictions.grad is not None
    assert features.grad is not None
    assert not torch.isnan(predictions.grad).any()
    assert not torch.isnan(features.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
