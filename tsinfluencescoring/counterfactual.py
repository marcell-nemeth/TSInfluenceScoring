"""
Counterfactual generation module using selected timestamps.

This module generates counterfactual time series by modifying
the selected influential timestamps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any


class CounterfactualGenerator(nn.Module):
    """
    Generate counterfactual time series based on selected timestamps.
    
    This module can modify selected timestamps to create counterfactual
    scenarios, useful for understanding causal relationships and model
    behavior.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        generation_method: str = "perturbation",
        perturbation_scale: float = 0.1,
        **kwargs
    ):
        """
        Initialize counterfactual generator.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden representations
            generation_method: Method for generating counterfactuals
                - 'perturbation': Add learned perturbations
                - 'replacement': Replace with generated values
                - 'removal': Remove selected timestamps
            perturbation_scale: Scale of perturbations
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.generation_method = generation_method
        self.perturbation_scale = perturbation_scale
        
        if generation_method == "perturbation":
            # Learn perturbations for selected timestamps
            self.perturbation_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.Tanh()  # Bounded perturbations
            )
        elif generation_method == "replacement":
            # Learn replacement values
            self.replacement_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
        elif generation_method == "removal":
            # No learned parameters for removal
            pass
        else:
            raise ValueError(f"Unknown generation method: {generation_method}")
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        intervention_type: str = "modify_selected"
    ) -> torch.Tensor:
        """
        Generate counterfactual time series.
        
        Args:
            x: Original time series (batch_size, seq_len, input_dim)
            mask: Selection mask (batch_size, seq_len)
            intervention_type: Type of intervention
                - 'modify_selected': Modify selected timestamps
                - 'modify_unselected': Modify unselected timestamps
                - 'remove_selected': Remove selected timestamps
                - 'remove_unselected': Remove unselected timestamps
                
        Returns:
            Counterfactual time series with same shape as input
        """
        batch_size, seq_len, input_dim = x.shape
        mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        
        if intervention_type == "modify_selected":
            target_mask = mask_expanded
        elif intervention_type == "modify_unselected":
            target_mask = 1.0 - mask_expanded
        elif intervention_type == "remove_selected":
            # Zero out selected timestamps
            return x * (1.0 - mask_expanded)
        elif intervention_type == "remove_unselected":
            # Zero out unselected timestamps
            return x * mask_expanded
        else:
            raise ValueError(f"Unknown intervention type: {intervention_type}")
        
        # Generate counterfactual based on method
        if self.generation_method == "perturbation":
            # Add learned perturbations to selected timestamps
            perturbations = self.perturbation_net(x) * self.perturbation_scale
            counterfactual = x + perturbations * target_mask
            
        elif self.generation_method == "replacement":
            # Replace selected timestamps with generated values
            replacements = self.replacement_net(x)
            counterfactual = x * (1.0 - target_mask) + replacements * target_mask
            
        elif self.generation_method == "removal":
            # Zero out target timestamps
            counterfactual = x * (1.0 - target_mask)
            
        return counterfactual
    
    def generate_multiple_counterfactuals(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        num_samples: int = 5,
        noise_scale: float = 0.1
    ) -> torch.Tensor:
        """
        Generate multiple counterfactual samples with variations.
        
        Args:
            x: Original time series (batch_size, seq_len, input_dim)
            mask: Selection mask (batch_size, seq_len)
            num_samples: Number of counterfactual samples to generate
            noise_scale: Scale of random noise for variation
            
        Returns:
            Multiple counterfactuals (num_samples, batch_size, seq_len, input_dim)
        """
        counterfactuals = []
        
        for _ in range(num_samples):
            # Generate base counterfactual
            cf = self.forward(x, mask, intervention_type="modify_selected")
            
            # Add random noise for variation
            if noise_scale > 0:
                noise = torch.randn_like(cf) * noise_scale
                mask_expanded = mask.unsqueeze(-1)
                cf = cf + noise * mask_expanded
            
            counterfactuals.append(cf)
        
        return torch.stack(counterfactuals, dim=0)
    
    def compute_counterfactual_distance(
        self,
        x: torch.Tensor,
        counterfactual: torch.Tensor,
        mask: torch.Tensor,
        metric: str = "l2"
    ) -> torch.Tensor:
        """
        Compute distance between original and counterfactual time series.
        
        Args:
            x: Original time series
            counterfactual: Counterfactual time series
            mask: Selection mask
            metric: Distance metric ('l2', 'l1', or 'cosine')
            
        Returns:
            Distance value(s)
        """
        mask_expanded = mask.unsqueeze(-1)
        
        if metric == "l2":
            diff = (x - counterfactual) ** 2
            distance = (diff * mask_expanded).sum(dim=(1, 2)) / (mask.sum(dim=1) + 1e-8)
        elif metric == "l1":
            diff = torch.abs(x - counterfactual)
            distance = (diff * mask_expanded).sum(dim=(1, 2)) / (mask.sum(dim=1) + 1e-8)
        elif metric == "cosine":
            # Cosine distance on flattened selected features
            x_flat = (x * mask_expanded).reshape(x.shape[0], -1)
            cf_flat = (counterfactual * mask_expanded).reshape(counterfactual.shape[0], -1)
            
            cosine_sim = F.cosine_similarity(x_flat, cf_flat, dim=-1)
            distance = 1.0 - cosine_sim
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return distance


class CounterfactualExplainer:
    """
    Utility class for explaining model predictions using counterfactuals.
    
    Generates and analyzes counterfactuals to understand which timestamps
    are most influential for model predictions.
    """
    
    def __init__(
        self,
        generator: CounterfactualGenerator,
        model: Callable[[torch.Tensor], torch.Tensor]
    ):
        """
        Initialize counterfactual explainer.
        
        Args:
            generator: Counterfactual generator module
            model: Prediction model (callable)
        """
        self.generator = generator
        self.model = model
    
    def explain_prediction(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        intervention_types: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Explain prediction by comparing original and counterfactual predictions.
        
        Args:
            x: Original time series
            mask: Selection mask
            intervention_types: List of intervention types to test
            
        Returns:
            Dictionary with explanations and prediction changes
        """
        if intervention_types is None:
            intervention_types = ["modify_selected", "remove_selected"]
        
        # Get original prediction
        with torch.no_grad():
            original_pred = self.model(x)
        
        explanations = {
            "original_prediction": original_pred.cpu(),
            "interventions": {}
        }
        
        # Test each intervention type
        for intervention_type in intervention_types:
            counterfactual = self.generator(x, mask, intervention_type=intervention_type)
            
            with torch.no_grad():
                cf_pred = self.model(counterfactual)
            
            # Compute prediction change
            pred_change = torch.abs(cf_pred - original_pred).mean().item()
            
            explanations["interventions"][intervention_type] = {
                "counterfactual_prediction": cf_pred.cpu(),
                "prediction_change": pred_change
            }
        
        return explanations
    
    def compute_attribution_scores(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attribution scores for each timestamp.
        
        Args:
            x: Original time series (batch_size, seq_len, input_dim)
            mask: Selection mask (batch_size, seq_len)
            
        Returns:
            Attribution scores (batch_size, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Get original prediction
        with torch.no_grad():
            original_pred = self.model(x)
        
        attribution_scores = torch.zeros(batch_size, seq_len, device=x.device)
        
        # Test removing each timestamp individually
        for t in range(seq_len):
            # Create mask that removes timestamp t
            temp_mask = mask.clone()
            temp_mask[:, t] = 0.0
            
            # Generate counterfactual
            counterfactual = self.generator(x, temp_mask, intervention_type="remove_unselected")
            
            # Compute prediction change
            with torch.no_grad():
                cf_pred = self.model(counterfactual)
                pred_change = torch.abs(cf_pred - original_pred).mean(dim=-1)
            
            attribution_scores[:, t] = pred_change
        
        return attribution_scores
