"""Tests for the utility module."""

import pytest
import torch
import torch.nn as nn
from tsinfluencescoring.utils import (
    InfluenceFramework,
    ModelAgnosticWrapper,
    create_simple_framework
)


def test_influence_framework_creation():
    """Test creation of InfluenceFramework."""
    framework = InfluenceFramework(
        input_dim=10,
        hidden_dim=64,
        selection_method="topk",
        k=5
    )
    
    assert framework.selector is not None
    assert framework.loss_module is not None
    assert framework.counterfactual_generator is not None


def test_influence_framework_forward():
    """Test forward pass through InfluenceFramework."""
    batch_size, seq_len, input_dim = 4, 20, 10
    
    framework = InfluenceFramework(
        input_dim=input_dim,
        hidden_dim=64,
        selection_method="topk",
        k=5
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    
    outputs = framework(x, return_details=True)
    
    assert "mask" in outputs
    assert "selected" in outputs
    assert "scores" in outputs
    assert "stats" in outputs
    assert "counterfactual" in outputs
    
    assert outputs["mask"].shape == (batch_size, seq_len)
    assert outputs["selected"].shape == (batch_size, seq_len, input_dim)


def test_influence_framework_compute_loss():
    """Test loss computation in InfluenceFramework."""
    batch_size, seq_len, input_dim = 4, 20, 10
    output_dim = 3
    
    framework = InfluenceFramework(
        input_dim=input_dim,
        hidden_dim=64,
        selection_method="topk",
        k=5,
        task_loss_fn=nn.MSELoss()
    )
    
    x = torch.randn(batch_size, seq_len, input_dim)
    predictions = torch.randn(batch_size, output_dim)
    targets = torch.randn(batch_size, output_dim)
    
    total_loss, loss_dict = framework.compute_loss(x, predictions, targets)
    
    assert total_loss.item() >= 0
    assert "task_loss" in loss_dict
    assert "diversity_loss" in loss_dict
    assert not torch.isnan(total_loss)


def test_create_simple_framework_regression():
    """Test creation of simple framework for regression."""
    framework = create_simple_framework(
        input_dim=10,
        k=5,
        hidden_dim=64,
        task="regression"
    )
    
    assert framework.selector is not None
    assert framework.loss_module is not None


def test_create_simple_framework_classification():
    """Test creation of simple framework for classification."""
    framework = create_simple_framework(
        input_dim=10,
        k=5,
        hidden_dim=64,
        task="classification"
    )
    
    assert framework.selector is not None
    assert framework.loss_module is not None


def test_model_agnostic_wrapper():
    """Test ModelAgnosticWrapper class."""
    batch_size, seq_len, input_dim = 4, 20, 10
    output_dim = 3
    
    # Create a simple base model
    base_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(seq_len * input_dim, output_dim)
    )
    
    framework = InfluenceFramework(
        input_dim=input_dim,
        hidden_dim=64,
        selection_method="topk",
        k=5
    )
    
    wrapper = ModelAgnosticWrapper(base_model, framework, use_selected_only=True)
    
    x = torch.randn(batch_size, seq_len, input_dim)
    predictions = wrapper(x, return_selection=False)
    
    assert predictions.shape == (batch_size, output_dim)


def test_model_agnostic_wrapper_with_selection():
    """Test ModelAgnosticWrapper returning selection details."""
    batch_size, seq_len, input_dim = 4, 20, 10
    output_dim = 3
    
    base_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(seq_len * input_dim, output_dim)
    )
    
    framework = InfluenceFramework(
        input_dim=input_dim,
        hidden_dim=64,
        selection_method="topk",
        k=5
    )
    
    wrapper = ModelAgnosticWrapper(base_model, framework)
    
    x = torch.randn(batch_size, seq_len, input_dim)
    predictions, mask, selected = wrapper(x, return_selection=True)
    
    assert predictions.shape == (batch_size, output_dim)
    assert mask.shape == (batch_size, seq_len)
    assert selected.shape == (batch_size, seq_len, input_dim)


def test_model_agnostic_wrapper_train_step():
    """Test training step in ModelAgnosticWrapper."""
    batch_size, seq_len, input_dim = 4, 20, 10
    output_dim = 3
    
    base_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(seq_len * input_dim, output_dim)
    )
    
    framework = InfluenceFramework(
        input_dim=input_dim,
        hidden_dim=64,
        selection_method="topk",
        k=5,
        task_loss_fn=nn.MSELoss()
    )
    
    wrapper = ModelAgnosticWrapper(base_model, framework)
    
    # Create optimizer
    all_params = list(base_model.parameters()) + list(framework.parameters())
    optimizer = torch.optim.Adam(all_params, lr=0.001)
    
    x = torch.randn(batch_size, seq_len, input_dim)
    targets = torch.randn(batch_size, output_dim)
    
    loss_dict = wrapper.train_step(x, targets, optimizer)
    
    assert "task_loss" in loss_dict
    assert "diversity_loss" in loss_dict
    assert "total_loss" in loss_dict


def test_framework_end_to_end():
    """Test end-to-end training with framework."""
    batch_size, seq_len, input_dim = 8, 20, 10
    output_dim = 3
    num_steps = 3
    
    # Create framework
    framework = create_simple_framework(
        input_dim=input_dim,
        k=5,
        hidden_dim=64,
        task="regression"
    )
    
    # Create base model
    base_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(seq_len * input_dim, output_dim)
    )
    
    # Create wrapper
    wrapper = ModelAgnosticWrapper(base_model, framework)
    
    # Create optimizer
    all_params = list(base_model.parameters()) + list(framework.parameters())
    optimizer = torch.optim.Adam(all_params, lr=0.001)
    
    # Training loop
    initial_loss = None
    for step in range(num_steps):
        x = torch.randn(batch_size, seq_len, input_dim)
        targets = torch.randn(batch_size, output_dim)
        
        loss_dict = wrapper.train_step(x, targets, optimizer)
        
        if step == 0:
            initial_loss = loss_dict["total_loss"]
        
        assert not any(v != v for v in loss_dict.values())  # Check for NaN
    
    # Framework should be trainable (loss may not decrease in 3 steps with random data)
    assert initial_loss is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
