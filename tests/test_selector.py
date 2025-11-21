"""Tests for the timestamp selector module."""

import pytest
import torch
from tsinfluencescoring.selector import (
    AttentionScorer,
    GumbelSoftmaxSelector,
    TopKSelector,
    TimestampSelector
)


def test_attention_scorer():
    """Test AttentionScorer module."""
    batch_size, seq_len, input_dim = 4, 20, 10
    hidden_dim = 64
    
    scorer = AttentionScorer(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=4)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    scores = scorer(x)
    
    assert scores.shape == (batch_size, seq_len)
    assert not torch.isnan(scores).any()
    assert not torch.isinf(scores).any()


def test_gumbel_softmax_selector():
    """Test GumbelSoftmaxSelector module."""
    batch_size, seq_len = 4, 20
    
    selector = GumbelSoftmaxSelector(temperature=1.0, hard=True)
    logits = torch.randn(batch_size, seq_len)
    
    mask = selector(logits)
    
    assert mask.shape == (batch_size, seq_len)
    assert (mask >= 0).all() and (mask <= 1).all()


def test_topk_selector():
    """Test TopKSelector module."""
    batch_size, seq_len = 4, 20
    k = 5
    
    selector = TopKSelector(k=k, hard=True)
    scores = torch.randn(batch_size, seq_len)
    
    mask = selector(scores)
    
    assert mask.shape == (batch_size, seq_len)
    # Check that k timestamps are selected
    assert torch.allclose(mask.sum(dim=1), torch.tensor(float(k)), atol=1e-5)


def test_timestamp_selector_topk():
    """Test TimestampSelector with top-k selection."""
    batch_size, seq_len, input_dim = 4, 20, 10
    k = 5
    
    selector = TimestampSelector(
        input_dim=input_dim,
        hidden_dim=64,
        selection_method="topk",
        k=k
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    
    mask, scores = selector(x, return_scores=True)
    
    assert mask.shape == (batch_size, seq_len)
    assert scores.shape == (batch_size, seq_len)
    assert torch.allclose(mask.sum(dim=1), torch.tensor(float(k)), atol=1e-5)


def test_timestamp_selector_gumbel():
    """Test TimestampSelector with Gumbel-Softmax selection."""
    batch_size, seq_len, input_dim = 4, 20, 10
    
    selector = TimestampSelector(
        input_dim=input_dim,
        hidden_dim=64,
        selection_method="gumbel",
        temperature=1.0
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    
    mask, scores = selector(x, return_scores=True)
    
    assert mask.shape == (batch_size, seq_len)
    assert scores.shape == (batch_size, seq_len)


def test_timestamp_selector_threshold():
    """Test TimestampSelector with threshold selection."""
    batch_size, seq_len, input_dim = 4, 20, 10
    
    selector = TimestampSelector(
        input_dim=input_dim,
        hidden_dim=64,
        selection_method="threshold",
        threshold=0.5
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    
    mask, _ = selector(x, return_scores=False)
    
    assert mask.shape == (batch_size, seq_len)
    assert (mask >= 0).all() and (mask <= 1).all()


def test_get_selected_timestamps():
    """Test extraction of selected timestamps."""
    batch_size, seq_len, input_dim = 4, 20, 10
    k = 5
    
    selector = TimestampSelector(
        input_dim=input_dim,
        hidden_dim=64,
        selection_method="topk",
        k=k
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    
    mask, _ = selector(x, return_scores=False)
    selected = selector.get_selected_timestamps(x, mask)
    
    assert selected.shape == x.shape
    # Check that unselected positions are zeroed
    mask_expanded = mask.unsqueeze(-1)
    assert torch.allclose(selected, x * mask_expanded)


def test_compute_selection_stats():
    """Test computation of selection statistics."""
    batch_size, seq_len, input_dim = 4, 20, 10
    k = 5
    
    selector = TimestampSelector(
        input_dim=input_dim,
        hidden_dim=64,
        selection_method="topk",
        k=k
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    
    mask, _ = selector(x, return_scores=False)
    stats = selector.compute_selection_stats(mask)
    
    assert "num_selected" in stats
    assert "selection_ratio" in stats
    assert "mean_score" in stats
    assert abs(stats["num_selected"] - k) < 1.0  # Should be close to k


def test_selector_backward():
    """Test that selector is differentiable."""
    batch_size, seq_len, input_dim = 4, 20, 10
    k = 5
    
    selector = TimestampSelector(
        input_dim=input_dim,
        hidden_dim=64,
        selection_method="topk",
        k=k
    )
    x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    
    mask, _ = selector(x, return_scores=False)
    loss = mask.sum()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
