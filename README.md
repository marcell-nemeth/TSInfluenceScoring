# TSInfluenceScoring

A modular PyTorch framework for selecting influential timestamps from time-series data using attention-based models. This framework enables you to identify which timestamps in your time series are most important for downstream tasks, generate counterfactuals, and understand model behavior.

## Features

- **Attention-Based Selection**: Uses multi-head self-attention to score and identify influential timestamps
- **Multiple Selection Methods**: 
  - Top-K selection (hard selection of k timestamps)
  - Gumbel-Softmax (differentiable soft selection)
  - Threshold-based selection
- **Differentiable**: All components are fully differentiable for end-to-end training
- **Rich Loss Functions**:
  - Task consistency loss for prediction accuracy
  - Diversity loss using Determinantal Point Process (DPP)
  - Optional mutual information loss
- **Counterfactual Generation**: Generate counterfactuals by modifying selected timestamps
- **Model-Agnostic Design**: Works with any PyTorch model as the downstream predictor
- **Explainability Tools**: Attribution scores and counterfactual explanations

## Installation

### From Source

```bash
git clone https://github.com/marcell-nemeth/TSInfluenceScoring.git
cd TSInfluenceScoring
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.20.0

## Interactive Demo

**📓 [Try the Jupyter Notebook Demo](demo.ipynb)** - A comprehensive interactive tutorial covering:
- Basic setup and data preparation
- Timestamp selection with different methods
- Training end-to-end models
- Counterfactual generation and analysis
- Visualization and interpretation

## Quick Start

```python
import torch
from tsinfluencescoring import create_simple_framework, ModelAgnosticWrapper

# Your time series data
X = torch.randn(32, 50, 10)  # (batch_size, seq_len, input_dim)
y = torch.randn(32, 1)        # targets

# Create framework
framework = create_simple_framework(
    input_dim=10,      # Feature dimension
    k=10,              # Select 10 most influential timestamps
    hidden_dim=64,
    task="regression"
)

# Wrap your existing model
import torch.nn as nn
base_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(50 * 10, 1)
)

wrapper = ModelAgnosticWrapper(base_model, framework, use_selected_only=True)

# Training
optimizer = torch.optim.Adam(wrapper.parameters(), lr=0.001)
loss_dict = wrapper.train_step(X, y, optimizer)

# Inference with selection details
predictions, mask, selected = wrapper(X, return_selection=True)
print(f"Selected timestamps: {mask.sum(dim=1).mean().item():.1f} per sample")
```

## Core Components

### 1. TimestampSelector

The main module for scoring and selecting timestamps:

```python
from tsinfluencescoring import TimestampSelector

selector = TimestampSelector(
    input_dim=10,
    hidden_dim=128,
    num_heads=4,
    selection_method="topk",  # or "gumbel", "threshold"
    k=10,                      # for topk method
    dropout=0.1
)

x = torch.randn(8, 50, 10)  # (batch, seq_len, features)
mask, scores = selector(x, return_scores=True)

# Get selected timestamps
selected = selector.get_selected_timestamps(x, mask)
```

### 2. Loss Functions

Multiple loss components for training:

```python
from tsinfluencescoring import CombinedLoss
import torch.nn as nn

loss_fn = CombinedLoss(
    task_loss_fn=nn.MSELoss(),  # Your task-specific loss
    alpha_task=1.0,              # Weight for task loss
    alpha_diversity=0.1,         # Weight for diversity loss
    alpha_mi=0.05,               # Weight for MI loss
    use_mi=False,                # Enable mutual information loss
    sigma=1.0                    # Kernel bandwidth for DPP
)

# Compute loss
predictions = model(x)
total_loss, loss_dict = loss_fn(predictions, targets, features=x, mask=mask)
```

### 3. Counterfactual Generation

Generate counterfactuals by modifying selected timestamps:

```python
from tsinfluencescoring import CounterfactualGenerator

generator = CounterfactualGenerator(
    input_dim=10,
    hidden_dim=128,
    generation_method="perturbation",  # or "replacement", "removal"
    perturbation_scale=0.1
)

# Generate counterfactual
counterfactual = generator(x, mask, intervention_type="modify_selected")

# Generate multiple samples
cf_samples = generator.generate_multiple_counterfactuals(
    x, mask, num_samples=5, noise_scale=0.1
)
```

### 4. Explainability

Explain predictions using counterfactuals:

```python
from tsinfluencescoring import CounterfactualExplainer

explainer = CounterfactualExplainer(generator, model)

# Explain a prediction
explanation = explainer.explain_prediction(
    x, mask, 
    intervention_types=["modify_selected", "remove_selected"]
)

# Compute attribution scores
attribution = explainer.compute_attribution_scores(x, mask)
```

## Architecture

The framework consists of several modular components:

```
┌─────────────────────────────────────────────────────────┐
│                   Input Time Series                      │
│                  (batch, seq_len, dim)                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              TimestampSelector                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │  AttentionScorer (Multi-head Self-Attention)     │   │
│  │         → Scores each timestamp                   │   │
│  └──────────────────────────────────────────────────┘   │
│                     │                                    │
│  ┌──────────────────▼──────────────────────────────┐   │
│  │  Selection Method                                │   │
│  │  • Top-K (hard selection)                        │   │
│  │  • Gumbel-Softmax (soft, differentiable)        │   │
│  │  • Threshold (sigmoid-based)                     │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │   Binary/Soft Mask      │
        │   (batch, seq_len)      │
        └────────┬────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌──────────────┐  ┌──────────────────┐
│   Selected   │  │  Counterfactual  │
│  Timestamps  │  │    Generator     │
└──────┬───────┘  └────────┬─────────┘
       │                   │
       ▼                   ▼
┌──────────────┐  ┌──────────────────┐
│ Downstream   │  │  Explanations &  │
│    Model     │  │   Attribution    │
└──────────────┘  └──────────────────┘
```

## Advanced Usage

### Custom Selection Method

You can implement custom selection mechanisms:

```python
import torch.nn as nn

class CustomSelector(nn.Module):
    def forward(self, scores):
        # Your custom selection logic
        # Must return a mask of same shape as scores
        return custom_mask

selector = TimestampSelector(
    input_dim=10,
    selection_method="topk",  # Use built-in, then replace
    k=10
)
selector.selector = CustomSelector()
```

### Integration with Existing Models

The framework is designed to be model-agnostic:

```python
# Your existing model
class MyTimeSeriesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=64, num_layers=2)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x.transpose(0, 1))
        return self.fc(out[-1])

# Wrap with selection framework
my_model = MyTimeSeriesModel()
framework = create_simple_framework(input_dim=10, k=10)
wrapper = ModelAgnosticWrapper(my_model, framework)
```

### Custom Loss Function

Define your own task loss:

```python
def custom_task_loss(predictions, targets):
    # Your custom loss logic
    return loss

loss_fn = CombinedLoss(
    task_loss_fn=custom_task_loss,
    alpha_task=1.0,
    alpha_diversity=0.1
)
```

## Examples

### Interactive Demo (Recommended)

**📓 [Jupyter Notebook Demo](demo.ipynb)** - Interactive tutorial with visualizations:
- Step-by-step walkthrough of all features
- Visual analysis of timestamp selections
- Training progress visualization
- Counterfactual generation and analysis
- Attribution scoring examples

Launch with:
```bash
jupyter notebook demo.ipynb
```

### Basic Usage

See `examples/basic_usage.py` for a complete example:

```bash
python examples/basic_usage.py
```

This demonstrates:
- Creating synthetic time series data
- Training with timestamp selection
- Analyzing which timestamps are selected
- Generating counterfactuals

### Advanced Features

See `examples/advanced_features.py`:

```bash
python examples/advanced_features.py
```

This shows:
- Comparison of different selection methods
- Various counterfactual generation approaches
- Explainability and attribution analysis
- Different loss function configurations

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_selector.py -v

# Run with coverage
pytest tests/ --cov=tsinfluencescoring --cov-report=html
```

## API Reference

### TimestampSelector

Main class for timestamp selection.

**Parameters:**
- `input_dim` (int): Dimension of input features
- `hidden_dim` (int, default=128): Dimension of hidden representations
- `num_heads` (int, default=4): Number of attention heads
- `selection_method` (str): Selection method ('topk', 'gumbel', 'threshold')
- `k` (int, optional): Number of timestamps to select (for topk)
- `temperature` (float, default=1.0): Temperature for Gumbel-Softmax
- `dropout` (float, default=0.1): Dropout rate

**Methods:**
- `forward(x, return_scores)`: Select timestamps and return mask
- `get_selected_timestamps(x, mask)`: Extract selected timestamps
- `compute_selection_stats(mask)`: Compute selection statistics

### CombinedLoss

Combined loss function for training.

**Parameters:**
- `task_loss_fn` (callable): Task-specific loss function
- `alpha_task` (float, default=1.0): Weight for task loss
- `alpha_diversity` (float, default=0.1): Weight for diversity loss
- `alpha_mi` (float, default=0.05): Weight for MI loss
- `use_mi` (bool, default=False): Whether to use MI loss
- `sigma` (float, default=1.0): Kernel bandwidth for DPP

### CounterfactualGenerator

Generate counterfactuals from selected timestamps.

**Parameters:**
- `input_dim` (int): Dimension of input features
- `hidden_dim` (int, default=128): Hidden dimension
- `generation_method` (str): Method ('perturbation', 'replacement', 'removal')
- `perturbation_scale` (float, default=0.1): Scale of perturbations

**Methods:**
- `forward(x, mask, intervention_type)`: Generate counterfactual
- `generate_multiple_counterfactuals(x, mask, num_samples)`: Generate multiple samples
- `compute_counterfactual_distance(x, cf, mask, metric)`: Compute distance

### InfluenceFramework

High-level interface combining all components.

**Parameters:**
- `input_dim` (int): Input feature dimension
- `hidden_dim` (int, default=128): Hidden dimension
- `selection_method` (str): Selection method
- `k` (int, optional): Number of timestamps to select
- `task_loss_fn` (callable, optional): Task loss function
- `use_counterfactual` (bool, default=True): Whether to include counterfactual generator

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{tsinfluencescoring2025,
  title = {TSInfluenceScoring: A Framework for Influential Timestamp Selection},
  author = {TSInfluenceScoring Contributors},
  year = {2025},
  url = {https://github.com/marcell-nemeth/TSInfluenceScoring}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This framework implements ideas from various research papers on attention mechanisms, time series analysis, and counterfactual explanations.