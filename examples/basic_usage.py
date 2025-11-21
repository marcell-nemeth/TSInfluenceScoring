"""
Basic example demonstrating the use of TSInfluenceScoring framework.

This example shows how to:
1. Create a simple time series dataset
2. Initialize the framework
3. Train a model with timestamp selection
4. Generate counterfactuals
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tsinfluencescoring import (
    create_simple_framework,
    ModelAgnosticWrapper
)


def create_synthetic_data(num_samples=100, seq_len=50, input_dim=5, output_dim=1):
    """
    Create synthetic time series data.
    
    The target is influenced more by certain timestamps (20-30) than others.
    """
    X = torch.randn(num_samples, seq_len, input_dim)
    
    # Target depends more heavily on timesteps 20-30
    influential_window = X[:, 20:30, :].mean(dim=(1, 2))  # (num_samples,)
    noise = torch.randn(num_samples) * 0.1
    y = (influential_window + noise).unsqueeze(-1)  # (num_samples, 1)
    
    return X, y


def main():
    print("=" * 60)
    print("TSInfluenceScoring Framework - Basic Example")
    print("=" * 60)
    
    # Hyperparameters
    input_dim = 5
    output_dim = 1
    seq_len = 50
    k = 10  # Number of timestamps to select
    hidden_dim = 64
    num_epochs = 20
    batch_size = 16
    learning_rate = 0.001
    
    print(f"\nConfiguration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Timestamps to select: {k}")
    print(f"  Hidden dimension: {hidden_dim}")
    
    # Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    X_train, y_train = create_synthetic_data(
        num_samples=200,
        seq_len=seq_len,
        input_dim=input_dim,
        output_dim=output_dim
    )
    X_test, y_test = create_synthetic_data(
        num_samples=50,
        seq_len=seq_len,
        input_dim=input_dim,
        output_dim=output_dim
    )
    print(f"   Train set: {X_train.shape}, {y_train.shape}")
    print(f"   Test set: {X_test.shape}, {y_test.shape}")
    
    # Create influence framework
    print("\n2. Creating influence framework...")
    framework = create_simple_framework(
        input_dim=input_dim,
        k=k,
        hidden_dim=hidden_dim,
        task="regression"
    )
    print(f"   Framework initialized with {k} timestamp selection")
    
    # Create a simple prediction model
    print("\n3. Creating prediction model...")
    base_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(seq_len * input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, output_dim)
    )
    print("   Simple MLP model created")
    
    # Wrap model with framework
    print("\n4. Wrapping model with framework...")
    model_wrapper = ModelAgnosticWrapper(
        base_model=base_model,
        framework=framework,
        use_selected_only=True
    )
    print("   Model wrapped - will use only selected timestamps")
    
    # Create optimizer
    all_params = list(base_model.parameters()) + list(framework.parameters())
    optimizer = optim.Adam(all_params, lr=learning_rate)
    
    # Training loop
    print(f"\n5. Training for {num_epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Mini-batch training
        num_batches = len(X_train) // batch_size
        epoch_losses = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            # Training step
            loss_dict = model_wrapper.train_step(X_batch, y_batch, optimizer)
            epoch_losses.append(loss_dict["total_loss"])
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        if (epoch + 1) % 5 == 0:
            # Evaluate on test set
            with torch.no_grad():
                test_pred = model_wrapper(X_test)
                test_loss = nn.MSELoss()(test_pred, y_test).item()
            
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {avg_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f}")
    
    print("-" * 60)
    
    # Analyze selections
    print("\n6. Analyzing timestamp selections...")
    with torch.no_grad():
        outputs = framework(X_test[:5], return_details=True)
        
        print(f"   Selection statistics:")
        stats = outputs["stats"]
        for key, value in stats.items():
            print(f"     {key}: {value:.4f}")
        
        # Show which timestamps are selected
        mask = outputs["mask"]
        print(f"\n   Selected timestamps for first test sample:")
        selected_indices = torch.where(mask[0] > 0.5)[0].tolist()
        print(f"     Indices: {selected_indices}")
        print(f"     (Ground truth influential window: 20-30)")
    
    # Generate counterfactuals
    print("\n7. Generating counterfactuals...")
    with torch.no_grad():
        sample_idx = 0
        x_sample = X_test[sample_idx:sample_idx+1]
        
        # Get original prediction
        pred_original = model_wrapper(x_sample)
        
        # Get mask and generate counterfactual
        outputs = framework(x_sample, return_details=True)
        mask = outputs["mask"]
        counterfactual = outputs["counterfactual"]
        
        # Predict on counterfactual
        pred_cf = model_wrapper(counterfactual)
        
        print(f"   Original prediction: {pred_original.item():.4f}")
        print(f"   Counterfactual prediction: {pred_cf.item():.4f}")
        print(f"   Prediction change: {abs(pred_cf.item() - pred_original.item()):.4f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
