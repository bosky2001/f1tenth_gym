#!/usr/bin/env python3
"""
This file is a temporary commit file.
Ineffieciently converts the cartesian labels to frenet labels and then trains 
in frenet mode.

Run:
# Cartesian mode
python train_comparable.py \
  --mode cartesian \
  --data ./Monza_100k.npz \
  --wpt ../Monza/Monza_centerline.csv \
  --pos_encoding \
  --n_samples 60000

# Frenet mode
python train_comparable.py \
  --mode frenet \
  --data ./Monza_100k.npz \
  --wpt ../Monza/Monza_centerline.csv \
  --pos_encoding \
  --n_samples 60000
"""

import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# --------------------------
# Utils
# --------------------------
def wrap_to_pi(a):
    """Wrap angle to [-pi, pi]"""
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_waypoints(waypoint_path):
    """Load centerline waypoints (x, y columns)"""
    return np.loadtxt(waypoint_path, delimiter=",", skiprows=1)


def compute_cumulative_lengths(trajectory_xy):
    """Compute arc length along trajectory"""
    n = trajectory_xy.shape[0]
    cum = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        cum[i] = cum[i - 1] + np.linalg.norm(trajectory_xy[i] - trajectory_xy[i - 1])
    return cum


def compute_psi_rad(trajectory_xy):
    """Compute heading angle at each waypoint"""
    diffs = trajectory_xy[1:] - trajectory_xy[:-1]
    psi = np.zeros(len(trajectory_xy), dtype=np.float64)
    psi[:-1] = np.arctan2(diffs[:, 1], diffs[:, 0])
    psi[-1] = psi[-2]
    return psi


def cartesian_to_frenet(x, y, theta, trajectory, cumulative_lengths, psi_rad):
    """
    Project (x,y,theta) to Frenet frame (s, d, theta_error).
    
    Returns:
        s: arc length along track (meters)
        d: signed lateral offset (left positive, meters)
        theta_error: wrap(theta - path_heading) (radians)
    """
    px = np.array([x, y], dtype=np.float64)

    best_dist2 = 1e18
    best_s = 0.0
    best_path_theta = 0.0
    best_d = 0.0

    for i in range(len(trajectory) - 1):
        p0 = trajectory[i]
        p1 = trajectory[i + 1]
        v = p1 - p0
        seg_len2 = float(v[0] * v[0] + v[1] * v[1])
        if seg_len2 < 1e-12:
            continue

        t = float(((px - p0) @ v) / seg_len2)
        t = max(0.0, min(1.0, t))
        proj = p0 + t * v

        dvec = px - proj
        dist2 = float(dvec[0] * dvec[0] + dvec[1] * dvec[1])
        
        if dist2 < best_dist2:
            best_dist2 = dist2
            seg_len = np.sqrt(seg_len2)
            best_s = float(cumulative_lengths[i] + t * seg_len)

            # Interpolate heading with wrap handling
            psi0 = float(psi_rad[i])
            psi1 = float(psi_rad[i + 1])
            dpsi = psi1 - psi0
            while dpsi > np.pi:
                dpsi -= 2 * np.pi
            while dpsi < -np.pi:
                dpsi += 2 * np.pi
            path_theta = psi0 + t * dpsi
            best_path_theta = float(path_theta)

            # Signed d: dot with left normal
            left_normal = np.array([-np.sin(path_theta), np.cos(path_theta)], dtype=np.float64)
            best_d = float(dvec @ left_normal)

    theta_error = wrap_to_pi(theta - best_path_theta)
    return best_s, best_d, float(theta_error)


# --------------------------
# Model
# --------------------------
class PerceptionMLP(nn.Module):
    """
    Simple MLP for LiDAR to pose prediction.
    Supports optional positional encoding.
    """
    
    def __init__(self, n_input=360, n_hidden=2048, n_output=3, use_pos_encoding=False, dropout=0.1):
        super().__init__()
        self.use_pos_encoding = use_pos_encoding
        self.n_lidar_points = n_input

        actual_input = n_input * 3 if use_pos_encoding else n_input

        self.net = nn.Sequential(
            nn.Linear(actual_input, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden // 2, n_hidden // 4),
            nn.ReLU(),
            nn.Linear(n_hidden // 4, n_hidden // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden // 8, n_output),
        )

    def add_positional_encoding(self, scans):
        """
        Add sin/cos of beam angles as features.
        scans: (B, N) -> output: (B, N*3)
        """
        B, N = scans.shape
        fov = 4.712388980384690  # 270 degrees
        angles = torch.linspace(-fov / 2, fov / 2, N, device=scans.device)
        angles = angles.unsqueeze(0).expand(B, -1)
        sin_a = torch.sin(angles)
        cos_a = torch.cos(angles)
        encoded = torch.stack([scans, sin_a, cos_a], dim=2).reshape(B, -1)
        return encoded

    def forward(self, x):
        if self.use_pos_encoding:
            x = self.add_positional_encoding(x)
        return self.net(x)


# --------------------------
# Training
# --------------------------
def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, device="cpu"):
    """Train model with simple MSE loss"""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(x_batch)
                val_loss += loss_fn(y_pred, y_batch).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step()

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

    return train_losses, val_losses


# --------------------------
# Evaluation
# --------------------------
def evaluate_cartesian(model, test_loader, device="cpu"):
    """Evaluate Cartesian predictions (x, y, theta)"""
    model.eval()
    model = model.to(device)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_pred = model(x_batch)
            all_predictions.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

    # Position errors
    x_error = predictions[:, 0] - targets[:, 0]
    y_error = predictions[:, 1] - targets[:, 1]
    xy_error = np.sqrt(x_error**2 + y_error**2)
    
    # Angle error with wrapping
    theta_error = wrap_to_pi(predictions[:, 2] - targets[:, 2])
    theta_error_deg = np.rad2deg(np.abs(theta_error))

    # Overall MSE
    mse = np.mean((predictions - targets)**2)

    print("\n" + "=" * 70)
    print("Test Evaluation (CARTESIAN)")
    print("=" * 70)
    print(f"Overall MSE: {mse:.6f}")
    print(f"\nPosition Errors:")
    print(f"  X RMSE:      {np.sqrt(np.mean(x_error**2)):.4f} m")
    print(f"  Y RMSE:      {np.sqrt(np.mean(y_error**2)):.4f} m")
    print(f"  XY RMSE:     {np.sqrt(np.mean(xy_error**2)):.4f} m")
    print(f"  XY Mean:     {np.mean(xy_error):.4f} m")
    print(f"  XY Median:   {np.median(xy_error):.4f} m")
    print(f"  XY 95%ile:   {np.percentile(xy_error, 95):.4f} m")
    print(f"\nOrientation Errors:")
    print(f"  Theta RMSE:  {np.sqrt(np.mean(theta_error**2)) * 180/np.pi:.2f}°")
    print(f"  Theta Mean:  {np.mean(theta_error_deg):.2f}°")
    print(f"  Theta Median:{np.median(theta_error_deg):.2f}°")
    print(f"  Theta 95%ile:{np.percentile(theta_error_deg, 95):.2f}°")
    print("=" * 70 + "\n")

    return predictions, targets, mse


def evaluate_frenet(model, test_loader, track_length, device="cpu"):
    """Evaluate Frenet predictions (s, d, theta_error)"""
    model.eval()
    model = model.to(device)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_pred = model(x_batch)
            all_predictions.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

    # s error (handle wraparound)
    s_error = predictions[:, 0] - targets[:, 0]
    s_error = np.minimum(np.abs(s_error), track_length - np.abs(s_error))
    
    # d error (no wrapping)
    d_error = predictions[:, 1] - targets[:, 1]
    
    # theta_error with wrapping
    theta_error = wrap_to_pi(predictions[:, 2] - targets[:, 2])
    theta_error_deg = np.rad2deg(np.abs(theta_error))

    # Overall MSE (note: this includes wrapped errors)
    mse = np.mean((predictions - targets)**2)

    print("\n" + "=" * 70)
    print("Test Evaluation (FRENET)")
    print("=" * 70)
    print(f"Overall MSE: {mse:.6f}")
    print(f"\nAlong-Track Errors (s):")
    print(f"  s RMSE (wrapped): {np.sqrt(np.mean(s_error**2)):.4f} m")
    print(f"  s Mean:           {np.mean(s_error):.4f} m")
    print(f"  s Median:         {np.median(s_error):.4f} m")
    print(f"  s 95%ile:         {np.percentile(s_error, 95):.4f} m")
    print(f"\nLateral Errors (d):")
    print(f"  d RMSE:      {np.sqrt(np.mean(d_error**2)):.4f} m")
    print(f"  d Mean abs:  {np.mean(np.abs(d_error)):.4f} m")
    print(f"  d Median:    {np.median(np.abs(d_error)):.4f} m")
    print(f"  d 95%ile:    {np.percentile(np.abs(d_error), 95):.4f} m")
    print(f"\nHeading Errors (theta_error):")
    print(f"  θ_err RMSE:  {np.sqrt(np.mean(theta_error**2)) * 180/np.pi:.2f}°")
    print(f"  θ_err Mean:  {np.mean(theta_error_deg):.2f}°")
    print(f"  θ_err Median:{np.median(theta_error_deg):.2f}°")
    print(f"  θ_err 95%ile:{np.percentile(theta_error_deg, 95):.2f}°")
    print("=" * 70 + "\n")

    return predictions, targets, mse


def plot_training_history(train_losses, val_losses, mode):
    """Plot training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"Training History ({mode.upper()})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.savefig(f"training_history_{mode}.png", dpi=150, bbox_inches="tight")
    print(f"Saved training_history_{mode}.png")


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["cartesian", "frenet"], required=True,
                       help="Coordinate system: cartesian (x,y,theta) or frenet (s,d,theta_error)")
    parser.add_argument("--data", type=str, required=True,
                       help="NPZ file with data_record=[x,y,theta,scan...]")
    parser.add_argument("--wpt", type=str, required=True,
                       help="Centerline CSV file")
    parser.add_argument("--n_samples", type=int, default=60000,
                       help="Number of samples to use from dataset")
    parser.add_argument("--pos_encoding", action="store_true",
                       help="Use positional encoding (sin/cos of beam angles)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--r_max", type=float, default=30.0,
                       help="Clip and normalize LiDAR to [0, r_max]")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 70)
    print(f"Training LiDAR → {args.mode.upper()} Pose Prediction")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Data: {args.data}")
    print(f"Centerline: {args.wpt}")
    print(f"Mode: {args.mode}")
    print(f"Positional Encoding: {args.pos_encoding}")
    print(f"Samples: {args.n_samples}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data = np.load(args.data)
    
    if "data_record" in data:
        data_record = data["data_record"][:args.n_samples].astype(np.float32)
        cartesian_poses = data_record[:, :3].astype(np.float64)
        scans = data_record[:, 3:].astype(np.float32)
    else:
        cartesian_poses = data["cartesian_labels"][:args.n_samples].astype(np.float64)
        scans = data["scans"][:args.n_samples].astype(np.float32)

    # Normalize scans
    scans = np.clip(scans, 0.0, args.r_max) / args.r_max

    # Load centerline
    wpts = load_waypoints(args.wpt)
    trajectory = wpts[:, 0:2].astype(np.float64)
    cumulative_lengths = compute_cumulative_lengths(trajectory)
    psi_rad = compute_psi_rad(trajectory)
    track_length = float(cumulative_lengths[-1])

    print(f"Track length: {track_length:.2f} m")
    print(f"LiDAR scans: {scans.shape}")

    # Prepare labels based on mode
    if args.mode == "cartesian":
        labels = cartesian_poses.astype(np.float32)
        print(f"Cartesian labels: {labels.shape}")
        print(f"  X range: [{labels[:, 0].min():.2f}, {labels[:, 0].max():.2f}]")
        print(f"  Y range: [{labels[:, 1].min():.2f}, {labels[:, 1].max():.2f}]")
        print(f"  θ range: [{np.rad2deg(labels[:, 2].min()):.1f}°, {np.rad2deg(labels[:, 2].max()):.1f}°]")
    else:  # frenet
        print("Converting to Frenet coordinates...")
        frenet_labels = []
        for i in range(len(cartesian_poses)):
            x, y, theta = cartesian_poses[i]
            s, d, theta_error = cartesian_to_frenet(x, y, theta, trajectory, cumulative_lengths, psi_rad)
            frenet_labels.append([s, d, theta_error])
        
        labels = np.array(frenet_labels, dtype=np.float32)
        print(f"Frenet labels: {labels.shape}")
        print(f"  s range: [{labels[:, 0].min():.2f}, {labels[:, 0].max():.2f}] m")
        print(f"  d range: [{labels[:, 1].min():.2f}, {labels[:, 1].max():.2f}] m")
        print(f"  θ_err range: [{np.rad2deg(labels[:, 2].min()):.1f}°, {np.rad2deg(labels[:, 2].max()):.1f}°]")

    # Split data
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        scans, labels, test_size=0.2, random_state=args.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=args.seed
    )

    print(f"\nData split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")

    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Initialize model
    model = PerceptionMLP(
        n_input=scans.shape[1],
        n_hidden=args.hidden,
        n_output=3,
        use_pos_encoding=args.pos_encoding,
        dropout=0.1
    )

    print(f"\nModel:")
    print(f"  Input: {scans.shape[1]} {'(with pos encoding)' if args.pos_encoding else ''}")
    print(f"  Hidden: {args.hidden}")
    print(f"  Output: 3 ({args.mode})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device
    )

    # Plot
    plot_training_history(train_losses, val_losses, args.mode)

    # Evaluate
    print("\nEvaluating on test set...")
    if args.mode == "cartesian":
        predictions, targets, mse = evaluate_cartesian(model, test_loader, device)
    else:
        predictions, targets, mse = evaluate_frenet(model, test_loader, track_length, device)

    # Save
    model_path = f"perception_{args.mode}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "mode": args.mode,
        "n_input": scans.shape[1],
        "n_hidden": args.hidden,
        "use_pos_encoding": args.pos_encoding,
        "track_length": track_length if args.mode == "frenet" else None,
        "test_mse": mse,
    }, model_path)
    print(f"Saved model: {model_path}")

    # Sample predictions
    print(f"\nSample predictions (first 10):")
    if args.mode == "cartesian":
        print(f"{'Predicted':<30} {'Actual':<30} {'Error'}")
        print("-" * 75)
        for i in range(min(10, len(predictions))):
            pred, tgt = predictions[i], targets[i]
            xy_err = np.sqrt((pred[0]-tgt[0])**2 + (pred[1]-tgt[1])**2)
            th_err = np.rad2deg(np.abs(wrap_to_pi(pred[2]-tgt[2])))
            print(f"[{pred[0]:6.2f}, {pred[1]:6.2f}, {np.rad2deg(pred[2]):6.1f}°]  "
                  f"[{tgt[0]:6.2f}, {tgt[1]:6.2f}, {np.rad2deg(tgt[2]):6.1f}°]  "
                  f"Δxy={xy_err:.3f}m Δθ={th_err:.1f}°")
    else:
        print(f"{'Predicted':<35} {'Actual':<35} {'Error'}")
        print("-" * 85)
        for i in range(min(10, len(predictions))):
            pred, tgt = predictions[i], targets[i]
            s_err = min(abs(pred[0]-tgt[0]), track_length - abs(pred[0]-tgt[0]))
            d_err = abs(pred[1]-tgt[1])
            th_err = np.rad2deg(np.abs(wrap_to_pi(pred[2]-tgt[2])))
            print(f"[s:{pred[0]:6.1f}, d:{pred[1]:5.2f}, θ:{np.rad2deg(pred[2]):6.1f}°]  "
                  f"[s:{tgt[0]:6.1f}, d:{tgt[1]:5.2f}, θ:{np.rad2deg(tgt[2]):6.1f}°]  "
                  f"Δs={s_err:.2f}m Δd={d_err:.3f}m Δθ={th_err:.1f}°")

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()