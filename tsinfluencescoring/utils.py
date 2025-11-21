"""
Utility functions and helper classes for the TSInfluenceScoring framework.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
from .selector import TimestampSelector
from .losses import CombinedLoss
from .counterfactual import CounterfactualGenerator


class InfluenceFramework(nn.Module):
    """
    Complete framework combining selector, losses, and counterfactual generation.
    
    This is a high-level interface that integrates all components for easy use.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        selection_method: str = "topk",
        k: Optional[int] = None,
        task_loss_fn: Optional[Callable] = None,
        alpha_task: float = 1.0,
        alpha_diversity: float = 0.1,
        alpha_mi: float = 0.05,
        use_mi: bool = False,
        use_counterfactual: bool = True,
        generation_method: str = "perturbation",
        **kwargs
    ):
        """
        Initialize the complete influence framework.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden representations
            num_heads: Number of attention heads
            selection_method: Selection method ('topk', 'gumbel', 'threshold')
            k: Number of timestamps to select (for topk)
            task_loss_fn: Task-specific loss function
            alpha_task: Weight for task consistency loss
            alpha_diversity: Weight for diversity loss
            alpha_mi: Weight for mutual information loss
            use_mi: Whether to use mutual information loss
            use_counterfactual: Whether to include counterfactual generator
            generation_method: Method for counterfactual generation
            **kwargs: Additional arguments
        """
        super().__init__()
        
        # Timestamp selector
        self.selector = TimestampSelector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            selection_method=selection_method,
            k=k,
            **kwargs
        )
        
        # Combined loss function
        self.loss_module = CombinedLoss(
            task_loss_fn=task_loss_fn,
            alpha_task=alpha_task,
            alpha_diversity=alpha_diversity,
            alpha_mi=alpha_mi,
            use_mi=use_mi,
            **kwargs
        )
        
        # Counterfactual generator (optional)
        self.use_counterfactual = use_counterfactual
        if use_counterfactual:
            self.counterfactual_generator = CounterfactualGenerator(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                generation_method=generation_method,
                **kwargs
            )
        else:
            self.counterfactual_generator = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the framework.
        
        Args:
            x: Input time series (batch_size, seq_len, input_dim)
            return_details: If True, return detailed outputs
            
        Returns:
            Dictionary containing:
                - mask: Selection mask
                - selected: Selected timestamps
                - scores: (optional) Raw scores
                - counterfactual: (optional) Counterfactual time series
        """
        # Select influential timestamps
        mask, scores = self.selector(x, return_scores=True)
        
        # Get selected timestamps
        selected = self.selector.get_selected_timestamps(x, mask)
        
        outputs = {
            "mask": mask,
            "selected": selected
        }
        
        if return_details:
            outputs["scores"] = scores
            outputs["stats"] = self.selector.compute_selection_stats(mask)
            
            # Generate counterfactual if available
            if self.use_counterfactual:
                counterfactual = self.counterfactual_generator(
                    x, mask, intervention_type="modify_selected"
                )
                outputs["counterfactual"] = counterfactual
        
        return outputs
    
    def compute_loss(
        self,
        x: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_features: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss for training.
        
        Args:
            x: Input time series
            predictions: Model predictions
            targets: Ground truth targets
            target_features: Optional target features for MI loss
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Get selection mask
        mask, _ = self.selector(x, return_scores=False)
        
        # Compute loss
        total_loss, loss_dict = self.loss_module(
            predictions=predictions,
            targets=targets,
            features=x,
            mask=mask,
            target_features=target_features
        )
        
        return total_loss, loss_dict


def create_simple_framework(
    input_dim: int,
    k: int,
    hidden_dim: int = 128,
    task: str = "regression"
) -> InfluenceFramework:
    """
    Create a simple framework with reasonable defaults.
    
    Args:
        input_dim: Dimension of input features
        k: Number of timestamps to select
        hidden_dim: Hidden dimension
        task: Task type ('regression' or 'classification')
        
    Returns:
        Configured InfluenceFramework
    """
    if task == "regression":
        task_loss_fn = nn.MSELoss()
    elif task == "classification":
        task_loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown task type: {task}")
    
    return InfluenceFramework(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=4,
        selection_method="topk",
        k=k,
        task_loss_fn=task_loss_fn,
        alpha_task=1.0,
        alpha_diversity=0.1,
        use_mi=False,
        use_counterfactual=True,
        generation_method="perturbation"
    )


class ModelAgnosticWrapper:
    """
    Wrapper to integrate the framework with any PyTorch model.
    
    This makes the framework truly model-agnostic by wrapping
    any existing model with timestamp selection capabilities.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        framework: InfluenceFramework,
        use_selected_only: bool = True
    ):
        """
        Initialize model-agnostic wrapper.
        
        Args:
            base_model: Existing PyTorch model
            framework: InfluenceFramework instance
            use_selected_only: If True, only pass selected timestamps to model
        """
        self.base_model = base_model
        self.framework = framework
        self.use_selected_only = use_selected_only
    
    def __call__(
        self,
        x: torch.Tensor,
        return_selection: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through wrapper.
        
        Args:
            x: Input time series
            return_selection: If True, also return selection details
            
        Returns:
            Model predictions (and optionally selection details)
        """
        # Get selection
        outputs = self.framework(x, return_details=False)
        mask = outputs["mask"]
        selected = outputs["selected"]
        
        # Pass through base model
        if self.use_selected_only:
            predictions = self.base_model(selected)
        else:
            predictions = self.base_model(x)
        
        if return_selection:
            return predictions, mask, selected
        return predictions
    
    def train_step(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            x: Input time series
            targets: Ground truth targets
            optimizer: Optimizer
            
        Returns:
            Dictionary with loss values
        """
        optimizer.zero_grad()
        
        # Forward pass
        predictions = self(x)
        
        # Compute loss
        total_loss, loss_dict = self.framework.compute_loss(
            x=x,
            predictions=predictions,
            targets=targets
        )
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return loss_dict


def visualize_selection(
    x: torch.Tensor,
    mask: torch.Tensor,
    sample_idx: int = 0,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize selected timestamps (requires matplotlib).
    
    Args:
        x: Input time series (batch_size, seq_len, input_dim)
        mask: Selection mask (batch_size, seq_len)
        sample_idx: Index of sample to visualize
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed. Skipping visualization.")
        return
    
    # Extract single sample
    x_sample = x[sample_idx].cpu().numpy()  # (seq_len, input_dim)
    mask_sample = mask[sample_idx].cpu().numpy()  # (seq_len,)
    
    seq_len, input_dim = x_sample.shape
    
    # Plot each dimension
    fig, axes = plt.subplots(input_dim, 1, figsize=(12, 2 * input_dim))
    if input_dim == 1:
        axes = [axes]
    
    for dim in range(input_dim):
        ax = axes[dim]
        
        # Plot time series
        ax.plot(x_sample[:, dim], label=f"Dimension {dim}", alpha=0.7)
        
        # Highlight selected timestamps
        selected_indices = mask_sample > 0.5
        ax.scatter(
            range(seq_len)[selected_indices],
            x_sample[selected_indices, dim],
            color="red",
            s=100,
            marker="o",
            label="Selected",
            zorder=5
        )
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
