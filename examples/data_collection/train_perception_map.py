import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import wandb

# Neural net to predict pose from lidar scan
class PerceptionMap(nn.Module):
    def __init__(self, n_input=1080, n_hidden=256, n_output=3, use_pos_encoding=True,
                 n_frequencies=4, dropout=0.0):
        super(PerceptionMap, self).__init__()

        self.use_pos_encoding = use_pos_encoding
        self.n_lidar_points = n_input
        self.n_frequencies = n_frequencies
        self.n_scale = 1000

        # Multi-frequency positional encoding:
        # For each beam: [range, sin(angle*1), cos(angle*1), sin(angle*2), cos(angle*2), ...]
        # Input dimension = n_input * (1 + 2 * n_frequencies)
        if use_pos_encoding:
            actual_input = n_input * (1 + 2 * n_frequencies)
        else:
            actual_input = n_input

        self.nn = nn.Sequential(
            nn.Linear(actual_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden//2),
            nn.ReLU(),
            nn.Linear(n_hidden//2, n_hidden//4),
            nn.ReLU(),
            nn.Linear(n_hidden//4, n_hidden//8),
            nn.ReLU(),
            nn.Linear(n_hidden//8, n_output)
        )
        # init bias to 0
        self.init_bias(self.nn)


    def init_bias(self, m):
        if isinstance(m, nn.Linear):
            m.bias.data.fill_(0.0)

    def add_positional_encoding(self, lidar_scans):
        """
        Transformer-style positional encoding for LiDAR beam angles.

        Formula: sin(pos / scale^(2i/d)), cos(pos / scale^(2i/d))
        Where pos = beam angle, scale = 10000 (or self.n_scale), d = n_frequencies * 2
        """
        batch_size = lidar_scans.shape[0]
        n_points = lidar_scans.shape[1]

        # F1TENTH LiDAR field of view
        fov = 4.7  # radians
        start_angle = -fov / 2
        end_angle = fov / 2

        # Create base angles for each beam (these are our "positions")
        angles = torch.linspace(start_angle, end_angle, n_points)
        angles = angles.to(lidar_scans.device).unsqueeze(0).repeat(batch_size, 1)

        # Transformer-style encoding: pos / scale^(2i/d)
        d = self.n_frequencies * 2  # total encoding dimensions
        features = [lidar_scans]

        for i in range(self.n_frequencies):
            # scale^(2i/d) creates wavelengths from 1 to scale
            div_term = self.n_scale ** (2 * i / d)
            features.append(torch.sin(angles / div_term))
            features.append(torch.cos(angles / div_term))

        # Stack: (batch, n_points, 1 + 2*n_frequencies)
        encoded = torch.stack(features, dim=2)
        encoded = encoded.reshape(batch_size, -1)

        return encoded

    def forward(self, x):
        if self.use_pos_encoding:
            x = self.add_positional_encoding(x)
        return self.nn(x)


# func to train model
def train_model(model, train_loader, val_loader, epochs=500, learning_rate=1e-3, device='cpu'):

    model = model.to(device)
    print(model.parameters())
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        # training loop
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)

            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)


        # Validation 
        model.eval()
        val_loss = 0.0
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            y_pred = model(x_val)

            loss = loss_fn(y_pred, y_val)

            val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Step the LR scheduler
        scheduler.step()

        # Log to wandb
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "epoch": epoch
        })

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    return train_losses, val_losses

# model eval
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    model = model.to(device)
    loss_fn = nn.MSELoss()

    test_loss = 0.0
    all_predictions = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            test_loss += loss.item()

            all_predictions.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    test_loss /= len(test_loader)

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

    # Calculate per-dimension errors
    x_error = np.mean((predictions[:, 0] - targets[:, 0])**2)
    y_error = np.mean((predictions[:, 1] - targets[:, 1])**2)
    theta_error = np.mean((predictions[:, 2] - targets[:, 2])**2)

    print(f"\n{'='*60}")
    print("Test Set Evaluation")
    print(f"{'='*60}")
    print(f"Overall MSE: {test_loss:.6f}")
    print(f"X MSE: {x_error:.6f}")
    print(f"Y MSE: {y_error:.6f}")
    print(f"Theta MSE: {theta_error:.6f}")
    print(f"{'='*60}\n")

    return test_loss, predictions, targets


def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("Training history saved to training_history.png")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", help="number of samples from dataset for training",
                    type=int, default=150000)
    parser.add_argument("--data", "-d", help="path to data file (.npz)",
                    type=str, default='./Monza_200k_wods.npz')

    args = parser.parse_args()

    # Configuration
    file_path = args.data
    batch_size = 32
    epochs = 1000
    learning_rate = 1e-3
    test_size = 0.2
    val_size = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*60}")
    print("Training LiDAR to Pose Perception Network")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Data file: {file_path}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    data = np.load(file_path)
    data_record = data['data_record'].astype(np.float32)

    print(data_record.shape)
    poses = data_record[:args.n_samples, :3]
    lidar_scans = data_record[:args.n_samples, 3:]

    # Downsample LiDAR from 1080 to 360 (pick every 3rd reading)
    lidar_scans = lidar_scans[:, ::3]

    # Normalize LiDAR scans by max range (30m)
    lidar_scans = lidar_scans / 30.0

    test_scans = np.hstack((lidar_scans, poses))
    print(f"Loaded {lidar_scans.shape[0]} samples")
    print(f"LiDAR shape: {lidar_scans.shape}")
    print(f"Poses shape: {poses.shape}")

    print(f"test scans shape: {test_scans.shape}")
    # Split into train/val/test
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        lidar_scans, poses, test_size=test_size, random_state=42
    )
    # testing
    # X_trainval, X_test, y_trainval, y_test = train_test_split(
    #     test_scans, poses, test_size=test_size, random_state=42
    # )

    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size/(1-test_size), random_state=42
    )

    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Convert to PyTorch tensors and create DataLoaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    n_input = lidar_scans.shape[1]  
    # n_input = test_scans.shape[1]  
    # n_hidden = 
    n_hidden = 720
    n_output = poses.shape[1]  # 3
    use_pos_encoding = False  # Multi-frequency trigonometric positional encoding
    n_frequencies = 4  # Number of frequency bands (1, 2, 4, 8)
    dropout = 0.0

    model = PerceptionMap(n_input, n_hidden, n_output, use_pos_encoding=use_pos_encoding,
                          n_frequencies=n_frequencies, dropout=dropout)

    # Initialize wandb
    wandb.init(
        project="f1tenth-perception",
        name="Levine map",
        config={
            "n_input": n_input,
            "n_hidden": n_hidden,
            "n_output": n_output,
            "use_pos_encoding": use_pos_encoding,
            "n_frequencies": n_frequencies,
            "dropout": dropout,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "n_samples": args.n_samples,
            "data_file": file_path,
        }
    )

    # torch compile
    model = torch.compile(model)
    print(f"\nPositional encoding: {'ENABLED' if use_pos_encoding else 'DISABLED'}")
    if use_pos_encoding:
        print(f"Frequencies: {n_frequencies} bands (1, 2, 4, ... {2**(n_frequencies-1)})")

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Train model
    print(f"\nStarting training for {epochs} epochs...\n")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device
    )

    # Plot training history
    plot_training_history(train_losses, val_losses)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, predictions, targets = evaluate_model(model, test_loader, device)

    # Log test metrics to wandb
    wandb.log({"test_loss": test_loss})

    # Save model
    model_path_pe = 'perception_model_pe.pth'
    model_path = 'perception_model.pth'

    if use_pos_encoding:
        torch.save(model.state_dict(), model_path_pe)
        print(f"Model saved to {model_path_pe}")
    
    else:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Show some predictions
    print("\nSample predictions (first 5 test samples):")
    print(f"{'Predicted':<30} {'Actual':<30}")
    print("-" * 60)
    for i in range(min(5, len(predictions))):
        pred = predictions[i]
        target = targets[i]
        print(f"[{pred[0]:6.3f}, {pred[1]:6.3f}, {np.rad2deg(pred[2]):7.2f}°]  "
              f"[{target[0]:6.3f}, {target[1]:6.3f}, {np.rad2deg(target[2]):7.2f}°]")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}\n")

    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    main()
