"""Tests for the counterfactual generation module."""

import pytest
import torch
from tsinfluencescoring.counterfactual import (
    CounterfactualGenerator,
    CounterfactualExplainer
)


def test_counterfactual_generator_perturbation():
    """Test CounterfactualGenerator with perturbation method."""
    batch_size, seq_len, input_dim = 4, 20, 10
    
    generator = CounterfactualGenerator(
        input_dim=input_dim,
        hidden_dim=64,
        generation_method="perturbation",
        perturbation_scale=0.1
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.rand(batch_size, seq_len)
    mask = (mask > 0.5).float()
    
    cf = generator(x, mask, intervention_type="modify_selected")
    
    assert cf.shape == x.shape
    assert not torch.allclose(cf, x)  # Should be different from original


def test_counterfactual_generator_replacement():
    """Test CounterfactualGenerator with replacement method."""
    batch_size, seq_len, input_dim = 4, 20, 10
    
    generator = CounterfactualGenerator(
        input_dim=input_dim,
        hidden_dim=64,
        generation_method="replacement"
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.rand(batch_size, seq_len)
    mask = (mask > 0.5).float()
    
    cf = generator(x, mask, intervention_type="modify_selected")
    
    assert cf.shape == x.shape


def test_counterfactual_generator_removal():
    """Test CounterfactualGenerator with removal method."""
    batch_size, seq_len, input_dim = 4, 20, 10
    
    generator = CounterfactualGenerator(
        input_dim=input_dim,
        hidden_dim=64,
        generation_method="removal"
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.rand(batch_size, seq_len)
    mask = (mask > 0.5).float()
    
    cf = generator(x, mask, intervention_type="remove_selected")
    
    assert cf.shape == x.shape
    # Check that selected timestamps are zeroed
    mask_expanded = mask.unsqueeze(-1)
    assert torch.allclose(cf, x * (1.0 - mask_expanded))


def test_intervention_types():
    """Test different intervention types."""
    batch_size, seq_len, input_dim = 4, 20, 10
    
    generator = CounterfactualGenerator(
        input_dim=input_dim,
        generation_method="perturbation"
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.rand(batch_size, seq_len)
    mask = (mask > 0.5).float()
    
    # Test modify_selected
    cf1 = generator(x, mask, intervention_type="modify_selected")
    assert cf1.shape == x.shape
    
    # Test modify_unselected
    cf2 = generator(x, mask, intervention_type="modify_unselected")
    assert cf2.shape == x.shape
    
    # Test remove_selected
    cf3 = generator(x, mask, intervention_type="remove_selected")
    mask_expanded = mask.unsqueeze(-1)
    assert torch.allclose(cf3, x * (1.0 - mask_expanded))
    
    # Test remove_unselected
    cf4 = generator(x, mask, intervention_type="remove_unselected")
    assert torch.allclose(cf4, x * mask_expanded)


def test_generate_multiple_counterfactuals():
    """Test generation of multiple counterfactual samples."""
    batch_size, seq_len, input_dim = 4, 20, 10
    num_samples = 5
    
    generator = CounterfactualGenerator(
        input_dim=input_dim,
        generation_method="perturbation"
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.rand(batch_size, seq_len)
    
    cfs = generator.generate_multiple_counterfactuals(
        x, mask, num_samples=num_samples, noise_scale=0.1
    )
    
    assert cfs.shape == (num_samples, batch_size, seq_len, input_dim)
    # Check that samples are different
    assert not torch.allclose(cfs[0], cfs[1])


def test_compute_counterfactual_distance():
    """Test counterfactual distance computation."""
    batch_size, seq_len, input_dim = 4, 20, 10
    
    generator = CounterfactualGenerator(
        input_dim=input_dim,
        generation_method="perturbation"
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.rand(batch_size, seq_len)
    cf = generator(x, mask)
    
    # Test L2 distance
    dist_l2 = generator.compute_counterfactual_distance(x, cf, mask, metric="l2")
    assert dist_l2.shape == (batch_size,)
    assert (dist_l2 >= 0).all()
    
    # Test L1 distance
    dist_l1 = generator.compute_counterfactual_distance(x, cf, mask, metric="l1")
    assert dist_l1.shape == (batch_size,)
    assert (dist_l1 >= 0).all()
    
    # Test cosine distance
    dist_cos = generator.compute_counterfactual_distance(x, cf, mask, metric="cosine")
    assert dist_cos.shape == (batch_size,)


def test_counterfactual_explainer():
    """Test CounterfactualExplainer class."""
    batch_size, seq_len, input_dim = 4, 20, 10
    output_dim = 3
    
    # Create a simple model
    model = torch.nn.Linear(input_dim * seq_len, output_dim)
    
    def model_wrapper(x):
        # Flatten input for the linear model
        return model(x.reshape(x.shape[0], -1))
    
    generator = CounterfactualGenerator(
        input_dim=input_dim,
        generation_method="perturbation"
    )
    explainer = CounterfactualExplainer(generator, model_wrapper)
    
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.rand(batch_size, seq_len)
    mask = (mask > 0.5).float()
    
    explanation = explainer.explain_prediction(x, mask)
    
    assert "original_prediction" in explanation
    assert "interventions" in explanation
    assert len(explanation["interventions"]) > 0


def test_compute_attribution_scores():
    """Test attribution score computation."""
    batch_size, seq_len, input_dim = 2, 10, 5
    output_dim = 3
    
    # Create a simple model
    model = torch.nn.Linear(input_dim * seq_len, output_dim)
    
    def model_wrapper(x):
        return model(x.reshape(x.shape[0], -1))
    
    generator = CounterfactualGenerator(
        input_dim=input_dim,
        generation_method="removal"  # Use removal for clearer attribution
    )
    explainer = CounterfactualExplainer(generator, model_wrapper)
    
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len)  # Select all
    
    attribution = explainer.compute_attribution_scores(x, mask)
    
    assert attribution.shape == (batch_size, seq_len)
    assert (attribution >= 0).all()


def test_counterfactual_backward():
    """Test that counterfactual generation is differentiable."""
    batch_size, seq_len, input_dim = 4, 20, 10
    
    generator = CounterfactualGenerator(
        input_dim=input_dim,
        generation_method="perturbation"
    )
    x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    mask = torch.rand(batch_size, seq_len)
    
    cf = generator(x, mask)
    loss = cf.sum()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
