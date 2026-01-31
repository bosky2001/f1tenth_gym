"""
Collect conformal prediction scores by comparing trained model predictions
with ground truth calibration data
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from data_collection.train_perception_map import PerceptionMap

def compute_residuals(model_path, calibration_data_path, n_input=360, n_hidden=256,
                      use_pos_encoding=False, dropout=0.2, device='cpu'):
    """
    Load trained model and compute residuals on calibration dataset

    Args:
        model_path: Path to saved model (.pth file)
        calibration_data_path: Path to calibration dataset (.npz file)
        n_input: Number of LiDAR input points (360 or 1080)
        n_hidden: Hidden layer size (must match trained model)
        use_pos_encoding: Whether positional encoding was used during training
        dropout: Dropout rate (must match trained model)
        device: 'cpu' or 'cuda'

    Returns:
        R: Array of residuals (absolute differences) between predictions and ground truth
           Shape: (N, 3) for [x_error, y_error, theta_error]
    """
    print(f"\n{'='*60}")
    print("Conformal Score Collection")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Calibration data: {calibration_data_path}")
    print(f"Positional encoding: {use_pos_encoding}")
    print(f"Hidden size: {n_hidden}")
    print(f"{'='*60}\n")

    # Load calibration data
    print("Loading calibration data...")
    data = np.load(calibration_data_path)

    # Handle both data formats
    if 'data_record' in data:
        # Concatenated format: [x, y, theta, lidar...]
        data_record = data['data_record'].astype(np.float32)
        poses = data_record[150000:, :3]
        lidar_data = data_record[150000:, 3:]

        # Check if data is downsampled (360 points) or full (1080 points)
        if lidar_data.shape[1] == 360 and n_input == 1080:
            # Data is downsampled but model expects full scan - need to upsample
            # For now, repeat each point 3 times to match expected input size
            lidar_scans = np.repeat(lidar_data, 3, axis=1)
        elif lidar_data.shape[1] == 1080 and n_input == 360:
            # Data is full but model expects downsampled - downsample it
            lidar_scans = lidar_data[:, ::3]
        else:
            lidar_scans = lidar_data
    else:
        # Separate arrays format
        lidar_scans = data['lidar_scans'].astype(np.float32)
        poses = data['poses'].astype(np.float32)

    print(f"Loaded {len(lidar_scans)} calibration samples")
    print(f"LiDAR shape: {lidar_scans.shape}")
    print(f"Poses shape: {poses.shape}")

    # Initialize model with same architecture as training
    n_output = poses.shape[1]  # 3
    model = PerceptionMap(n_input, n_hidden, n_output,
                         use_pos_encoding=use_pos_encoding, dropout=dropout)

    # Load trained weights
    print(f"\nLoading model from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    # Handle torch.compile() prefix if present
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Convert data to PyTorch tensors
    X = torch.from_numpy(lidar_scans).to(device)
    y_true = torch.from_numpy(poses).to(device)

    # Get predictions
    print("\nComputing predictions...")
    with torch.no_grad():
        y_pred = model(X)

    # Compute residuals (absolute differences)
    residuals = torch.abs(y_pred - y_true)

    # Convert back to numpy
    R = residuals.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    # Print statistics
    print(f"\n{'='*60}")
    print("Residual Statistics")
    print(f"{'='*60}")
    print(f"Residuals shape: {R.shape}")
    print(f"\nX position errors:")
    print(f"  Mean: {R[:, 0].mean():.6f}")
    print(f"  Std:  {R[:, 0].std():.6f}")
    print(f"  Max:  {R[:, 0].max():.6f}")
    print(f"\nY position errors:")
    print(f"  Mean: {R[:, 1].mean():.6f}")
    print(f"  Std:  {R[:, 1].std():.6f}")
    print(f"  Max:  {R[:, 1].max():.6f}")
    print(f"\nTheta (orientation) errors:")
    print(f"  Mean: {R[:, 2].mean():.6f} rad ({np.rad2deg(R[:, 2].mean()):.3f}°)")
    print(f"  Std:  {R[:, 2].std():.6f} rad ({np.rad2deg(R[:, 2].std()):.3f}°)")
    print(f"  Max:  {R[:, 2].max():.6f} rad ({np.rad2deg(R[:, 2].max()):.3f}°)")
    print(f"{'='*60}\n")

    return R, y_pred_np, y_true_np


def visualize_conformal_predictions(y_true, y_pred, error_bounds, conformal_band, output_prefix='conformal'):
    """
    Visualize model predictions with conformal error bounds

    Args:
        y_true: Ground truth poses (N, 3)
        y_pred: Predicted poses (N, 3)
        error_bounds: Error bounds for each dimension (3,) - [x_bound, y_bound, theta_bound]
        conformal_band: Confidence level (e.g., 0.9)
        output_prefix: Prefix for output filenames
    """
    # Use first 100 samples for clearer visualization
    n_samples = min(100, len(y_true))
    indices = np.arange(n_samples)

    quantile_str = f"{int(conformal_band*100)}"

    # X position plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(indices, y_true[:n_samples, 0], 'b-', linewidth=0.8, alpha=0.7, label='Ground Truth', zorder=3)
    ax.plot(indices, y_pred[:n_samples, 0], 'r-', linewidth=1.5, label='Model Prediction', zorder=2)
    ax.fill_between(indices,
                     y_pred[:n_samples, 0] - error_bounds[0],
                     y_pred[:n_samples, 0] + error_bounds[0],
                     alpha=0.3, color='orange', label=f'Error Bound (±{error_bounds[0]:.4f})', zorder=1)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('X Position (m)', fontsize=12)
    ax.set_title('X Position: Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = f'{output_prefix}_x_q{quantile_str}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"X position plot saved to {output_path}")
    plt.close()

    # Y position plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(indices, y_true[:n_samples, 1], 'b-', linewidth=0.8, alpha=0.7, label='Ground Truth', zorder=3)
    ax.plot(indices, y_pred[:n_samples, 1], 'r-', linewidth=1.5, label='Model Prediction', zorder=2)
    ax.fill_between(indices,
                     y_pred[:n_samples, 1] - error_bounds[1],
                     y_pred[:n_samples, 1] + error_bounds[1],
                     alpha=0.3, color='orange', label=f'Error Bound (±{error_bounds[1]:.4f})', zorder=1)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Y Position: Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = f'{output_prefix}_y_q{quantile_str}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Y position plot saved to {output_path}")
    plt.close()

    # Theta (orientation) plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    y_true_deg = np.rad2deg(y_true[:n_samples, 2])
    y_pred_deg = np.rad2deg(y_pred[:n_samples, 2])
    error_bound_deg = np.rad2deg(error_bounds[2])
    ax.plot(indices, y_true_deg, 'b-', linewidth=0.8, alpha=0.7, label='Ground Truth', zorder=3)
    ax.plot(indices, y_pred_deg, 'r-', linewidth=1.5, label='Model Prediction', zorder=2)
    ax.fill_between(indices,
                     y_pred_deg - error_bound_deg,
                     y_pred_deg + error_bound_deg,
                     alpha=0.3, color='orange', label=f'Error Bound (±{error_bound_deg:.2f}°)', zorder=1)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Theta (degrees)', fontsize=12)
    ax.set_title('Orientation (Theta): Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = f'{output_prefix}_theta_q{quantile_str}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Theta plot saved to {output_path}")
    plt.close()

    print(f"Showing first {n_samples} samples for clarity")


def visualize_residual_distributions(R, error_bounds, conformal_band, output_path='residual_distributions.png'):
    """
    Visualize residual distributions and error bounds

    Args:
        R: Residuals array (N, 3)
        error_bounds: Error bounds for each dimension (3,)
        conformal_band: Confidence level (e.g., 0.9)
        output_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    labels = ['X Position Error (m)', 'Y Position Error (m)', 'Theta Error (degrees)']

    for i, ax in enumerate(axes):
        residuals = R[:, i] if i < 2 else np.rad2deg(R[:, i])
        error_bound = error_bounds[i] if i < 2 else np.rad2deg(error_bounds[i])

        # Histogram
        n, bins, patches = ax.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')

        # Add vertical line for error bound
        ax.axvline(error_bound, color='red', linestyle='--', linewidth=2,
                  label=f'{conformal_band*100:.0f}% Quantile: {error_bound:.4f}')

        # Add mean line
        mean_val = residuals.mean()
        ax.axvline(mean_val, color='green', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.4f}')

        ax.set_xlabel(labels[i], fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{labels[i]} Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics text box
        stats_text = f'Max: {residuals.max():.4f}\nStd: {residuals.std():.4f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Residual distribution visualization saved to {output_path}")


def main():
    # Configuration - MUST match your training configuration!
    model_path = 'data_collection/perception_model.pth'
    calibration_data_path = 'data_collection/Monza_100k.npz'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model architecture parameters - MUST match the saved model!
    n_input = 360  # Full LiDAR scan (model was trained on 1080 points, not 360)
    n_hidden = 2048  # Hidden layer size used during training
    use_pos_encoding = False  # No positional encoding in old model
    dropout = 0.0  # Dropout used during training

    # Compute residuals and get predictions
    R, y_pred, y_true = compute_residuals(
        model_path, calibration_data_path,
        n_input=n_input,
        n_hidden=n_hidden,
        use_pos_encoding=use_pos_encoding,
        dropout=dropout,
        device=device
    )

    print("Conformal score collection complete!")

    # Compute conformal error bounds
    conformal_band = 0.5
    error_bounds = np.quantile(R, conformal_band, axis=0)

    print(f"\n{'='*60}")
    print(f"Conformal Prediction Error Bounds ({conformal_band*100:.0f}% confidence)")
    print(f"{'='*60}")
    print(f"X position:  ±{error_bounds[0]:.6f} m")
    print(f"Y position:  ±{error_bounds[1]:.6f} m")
    print(f"Theta:       ±{error_bounds[2]:.6f} rad (±{np.rad2deg(error_bounds[2]):.3f}°)")
    print(f"{'='*60}\n")

    # Create visualizations
    print("Creating visualizations...")
    visualize_conformal_predictions(y_true, y_pred, error_bounds, conformal_band,
                                   output_prefix='conformal_predictions')

    print("\nAll visualizations complete!")

if __name__ == '__main__':
    main()
