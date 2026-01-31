"""
Visualize collected data points on the actual map
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import yaml
from argparse import Namespace


def load_map_image(map_path, map_ext):
    """Load the map image"""
    if map_ext == '.png':
        img = mpimg.imread(map_path + map_ext)
    else:
        print(f"Map extension {map_ext} not supported for visualization")
        return None
    return img


def visualize_data_on_map(data_file, config_file, max_points=None):
    """
    Visualize collected data points overlaid on the map

    Args:
        data_file: Path to .npz file with collected data
        config_file: Path to YAML config file with map info
        max_points: Maximum number of points to plot (for performance)
    """
    # Load configuration
    with open(config_file) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # Load data
    print(f"Loading data from {data_file}...")
    data = np.load(data_file)
    data_record = data['data_record']

    print(f"Data shape: {data_record.shape}")
    print(f"Total samples: {len(data_record)}")

    # Extract positions (first 3 columns are x, y, theta)
    positions = data_record[:, :3]
    x_positions = positions[:, 0]
    y_positions = positions[:, 1]

    # y_positions = positions[:, 0]
    # x_positions = positions[:, 1]
    theta_positions = positions[:, 2]

    print(f"\nPosition statistics:")
    print(f"X: min={x_positions.min():.2f}, max={x_positions.max():.2f}")
    print(f"Y: min={y_positions.min():.2f}, max={y_positions.max():.2f}")

    # Load map image
    print(f"\nLoading map from {conf.map_path}{conf.map_ext}...")
    map_img = load_map_image(conf.map_path, conf.map_ext)

    if map_img is None:
        print("Could not load map image, showing positions only")
        plot_positions_only(x_positions, y_positions, theta_positions, max_points)
        return

    # Get map metadata from YAML
    map_yaml_path = conf.map_path + '.yaml'
    try:
        with open(map_yaml_path) as f:
            map_metadata = yaml.safe_load(f)

        resolution = map_metadata['resolution']
        origin = map_metadata['origin']

        print(f"\nMap metadata:")
        print(f"Resolution: {resolution} m/pixel")
        print(f"Origin: {origin}")

    except FileNotFoundError:
        print(f"Map metadata file {map_yaml_path} not found")
        print("Using default resolution and origin")
        resolution = 0.05
        origin = conf.origin if hasattr(conf, 'origin') else [0, 0, 0]

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Helper function to convert world coords to pixel coords (same as frenet_debug)
    img_height = map_img.shape[0]
    def world_to_pixel(x, y):
        px = (x - origin[0]) / resolution
        py = img_height - (y - origin[1]) / resolution
        return px, py

    # Subsample points if too many
    if max_points and len(x_positions) > max_points:
        indices = np.random.choice(len(x_positions), max_points, replace=False)
        x_plot = x_positions[indices]
        y_plot = y_positions[indices]
        theta_plot = theta_positions[indices]
        print(f"\nSubsampling to {max_points} points for visualization")
    else:
        x_plot = x_positions
        y_plot = y_positions
        theta_plot = theta_positions

    # Convert to pixel coordinates
    x_px, y_px = world_to_pixel(x_plot, y_plot)

    # Left plot: All points on map
    ax = axes[0]
    ax.imshow(map_img, cmap='gray')  # Default origin='upper'

    # Plot points in pixel coordinates
    scatter = ax.scatter(x_px, y_px, c=theta_plot, s=1, alpha=0.5,
                        cmap='hsv', vmin=0, vmax=2*np.pi)

    ax.set_xlabel('Pixel X', fontsize=12)
    ax.set_ylabel('Pixel Y', fontsize=12)
    ax.set_title(f'Data Points on Map ({len(x_positions)} total samples)',
                fontsize=14, fontweight='bold')

    # Add colorbar for orientation
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Orientation (rad)', fontsize=10)

    # Right plot: Hexbin density plot (avoids histogram orientation issues)
    ax = axes[1]
    ax.imshow(map_img, cmap='gray', alpha=0.3)  # Default origin='upper'

    # Convert all positions to pixel coords
    all_x_px, all_y_px = world_to_pixel(x_positions, y_positions)

    # Use hexbin for density visualization
    hb = ax.hexbin(all_x_px, all_y_px, gridsize=50, cmap='hot', alpha=0.6, mincnt=1)

    ax.set_xlabel('Pixel X', fontsize=12)
    ax.set_ylabel('Pixel Y', fontsize=12)
    ax.set_title('Point Density Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlim(0, map_img.shape[1])
    ax.set_ylim(map_img.shape[0], 0)  # Invert y to match image

    cbar2 = plt.colorbar(hb, ax=ax)
    cbar2.set_label('Sample Density', fontsize=10)

    plt.tight_layout()

    # Save figure
    output_file = 'data_visualization_on_map.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_file}")

    plt.show()


def plot_positions_only(x_positions, y_positions, theta_positions, max_points=None):
    """Plot positions without map background"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Subsample if needed
    if max_points and len(x_positions) > max_points:
        indices = np.random.choice(len(x_positions), max_points, replace=False)
        x_plot = x_positions[indices]
        y_plot = y_positions[indices]
        theta_plot = theta_positions[indices]
    else:
        x_plot = x_positions
        y_plot = y_positions
        theta_plot = theta_positions

    # Left: Scatter colored by orientation
    ax = axes[0]
    scatter = ax.scatter(x_plot, y_plot, c=theta_plot, s=2, alpha=0.5,
                        cmap='hsv', vmin=0, vmax=2*np.pi)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Collected Data Points ({len(x_positions)} samples)',
                fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Orientation (rad)', fontsize=10)

    # Right: Density heatmap
    ax = axes[1]
    hist, xedges, yedges = np.histogram2d(x_positions, y_positions, bins=100)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(hist.T, origin='lower', extent=extent,
                   cmap='hot', interpolation='gaussian')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Point Density Heatmap', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    cbar2 = plt.colorbar(im, ax=ax)
    cbar2.set_label('Sample Density', fontsize=10)

    plt.tight_layout()
    output_file = 'data_visualization_positions.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_file}")
    plt.show()


def main():
    # Configuration
    data_file = 'Monza_100k.npz'  # Update path as needed
    config_file = '../Monza/Monza_map.yaml'  # Update path as needed
    max_points = 10000  # Limit points for faster rendering

    visualize_data_on_map(data_file, config_file, max_points)


if __name__ == '__main__':
    main()
