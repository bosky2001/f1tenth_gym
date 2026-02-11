import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from tqdm import tqdm
from numba import njit


@njit(cache=True)
def frenet_to_cartesian(s, d, theta_error, trajectory, cumulative_lengths, psi_rad):
    """
    Convert Frenet coordinates (s, d, theta_error) back to Cartesian (x, y, theta).

    Args:
        s (float): Arc length along the path (meters)
        d (float): Signed lateral offset (meters, positive = left)
        theta_error (float): Heading error relative to path (radians)
        trajectory (np.ndarray): Reference path as (N, 2) array of [x, y] points
        cumulative_lengths (np.ndarray): Arc lengths from waypoint file (N,)
        psi_rad (np.ndarray): Path heading at each waypoint (N,) from waypoint file

    Returns:
        x (float): World x position
        y (float): World y position
        theta (float): World heading (radians)
    """
    n_points = trajectory.shape[0]
    n_segments = n_points - 1

    # Compute segment vectors
    diffs = trajectory[1:, :] - trajectory[:-1, :]

    # Compute segment lengths from the s values (consistent with cumulative_lengths)
    segment_lengths = cumulative_lengths[1:] - cumulative_lengths[:-1]

    # Find which segment s falls on
    segment_idx = 0
    for i in range(n_segments):
        if cumulative_lengths[i + 1] >= s:
            segment_idx = i
            break
        if i == n_segments - 1:
            segment_idx = i

    # Compute parameter t along the segment (0 to 1)
    s_on_segment = s - cumulative_lengths[segment_idx]
    t = s_on_segment / segment_lengths[segment_idx] if segment_lengths[segment_idx] > 0 else 0.0
    t = max(0.0, min(1.0, t))  # Manual clip for numba compatibility

    # Point on the path at arc length s (interpolate between waypoints)
    path_point = trajectory[segment_idx] + t * diffs[segment_idx]

    # Interpolate path heading from waypoint psi_rad values
    # This is more accurate than computing from segment vectors
    psi_start = psi_rad[segment_idx]
    psi_end = psi_rad[segment_idx + 1] if segment_idx + 1 < n_points else psi_rad[segment_idx]

    # Handle angle wrapping for interpolation
    delta_psi = psi_end - psi_start
    while delta_psi > np.pi:
        delta_psi -= 2.0 * np.pi
    while delta_psi < -np.pi:
        delta_psi += 2.0 * np.pi

    path_theta = psi_start + t * delta_psi

    # Normal vector (perpendicular to path, pointing left)
    # Standard left normal: rotate path direction 90° CCW
    normal = np.array([-np.sin(path_theta), np.cos(path_theta)])

    # Cartesian position: path_point + d * normal
    x = path_point[0] + d * normal[0]
    y = path_point[1] + d * normal[1]

    # Cartesian heading = path heading + heading error
    theta = path_theta + theta_error

    # Normalize theta to [-pi, pi]
    while theta > np.pi:
        theta -= 2.0 * np.pi
    while theta < -np.pi:
        theta += 2.0 * np.pi

    return x, y, theta


def load_waypoints(waypoint_path, delimiter=';', skiprows=3):
    """
    Load waypoints from CSV file.

    Expected format: s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
    """
    waypoints = np.loadtxt(waypoint_path, delimiter=delimiter, skiprows=skiprows)
    return waypoints


def compute_cumulative_lengths(trajectory):
    """
    Compute cumulative arc lengths along the trajectory.

    Args:
        trajectory: (N, 2) array of [x, y] points

    Returns:
        cumulative_lengths: (N,) array where cumulative_lengths[i] is the
                           arc length from start to point i
    """
    n_points = trajectory.shape[0]
    cumulative_lengths = np.zeros(n_points)

    for i in range(1, n_points):
        segment_length = np.linalg.norm(trajectory[i] - trajectory[i-1])
        cumulative_lengths[i] = cumulative_lengths[i-1] + segment_length

    return cumulative_lengths


def main():
    """
    Sample positions in Frenet frame, convert to Cartesian, and collect LiDAR observations.
    Stores both Frenet and Cartesian coordinates for experimentation

    Frenet sampling allows us to:
    1. Sample uniformly along the track (s coordinate)
    2. Sample lateral offsets within track bounds (d coordinate)
    3. Sample heading variations relative to track direction (theta_error)
    """

    with open('../Levine/levine_slam.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # Add required fields not in yaml
    conf.map_path = '../Levine/levine_slam'
    conf.map_ext = '.pgm'
    conf.sample_num = 200000
    conf.save_filename = 'Levine_200k.npz'

    # Load waypoints for Frenet conversion
    # Levine format: s_m; x_m; y_m; psi_rad; kappa; vx; ax
    waypoint_path = '../Levine/levine_centerline.csv'
    waypoints = load_waypoints(waypoint_path, delimiter=';', skiprows=3)

    # Extract trajectory data
    trajectory = waypoints[:, 1:3].astype(np.float64)  # x, y (columns 1, 2)
    cumulative_lengths = waypoints[:, 0].astype(np.float64)  # s (column 0)
    total_track_length = cumulative_lengths[-1]
    psi_rad = waypoints[:, 3].astype(np.float64)  # psi (column 3)

    print(f"Track length: {total_track_length:.2f} m")
    print(f"Number of waypoints: {len(trajectory)}")

    # Frenet sampling parameters
    s_min = 0.0
    s_max = total_track_length
    d_min = -0.7   # 1 meter right of centerline
    d_max = 0.7    # 1 meter left of centerline
    theta_error_min = -1*np.pi/4         # Full rotation range
    theta_error_max = 1 * np.pi/4    # 0 to 360 degrees

    print(f"\nFrenet sampling bounds:")
    print(f"  s: [{s_min:.1f}, {s_max:.1f}] m")
    print(f"  d: [{d_min:.1f}, {d_max:.1f}] m")
    print(f"  theta_error: [{np.rad2deg(theta_error_min):.1f}, {np.rad2deg(theta_error_max):.1f}] deg")

    # Initialize environment
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, disable_env_checker=True,)

    start = time.time()
    cnt = 0

    frenet_coords = []      # [s, d, theta_error]
    cartesian_poses = []    # [x, y, theta]
    lidar_scans = []        # LiDAR data

    with tqdm(total=conf.sample_num, desc="Collecting data") as pbar:
        while cnt < conf.sample_num:
            # Sample in Frenet frame
            s = np.random.uniform(s_min, s_max)
            d = np.random.uniform(d_min, d_max)
            theta_error = np.random.uniform(theta_error_min, theta_error_max)

            # Convert to Cartesian using waypoint headings
            x, y, theta = frenet_to_cartesian(s, d, theta_error, trajectory, cumulative_lengths, psi_rad)

            sample_pos = [x, y, theta]

            # Reset environment at sampled position
            obs, step_reward, done, info = env.reset(poses=np.array([sample_pos]))

            # Check if position is valid (not in collision)
            if not done:
                # Get actual pose from simulator (may differ slightly due to discretization)
                actual_x = obs['poses_x'][0]
                actual_y = obs['poses_y'][0]
                actual_theta = obs['poses_theta'][0]
                
                # Get LiDAR scan
                lidar_scan = np.array(obs['scans'][0])  # Full resolution (1080), no downsample
                
                # Store both coordinate systems
                frenet_coords.append([s, d, theta_error])
                cartesian_poses.append([actual_x, actual_y, actual_theta])
                lidar_scans.append(lidar_scan)

                cnt += 1
                pbar.update(1)

    frenet_coords = np.array(frenet_coords, dtype=np.float32)
    cartesian_poses = np.array(cartesian_poses, dtype=np.float32)
    lidar_scans = np.array(lidar_scans, dtype=np.float32)

    print(f"\nData collection summary:")
    print(f"  Frenet coordinates shape: {frenet_coords.shape}")
    print(f"  Cartesian poses shape: {cartesian_poses.shape}")
    print(f"  LiDAR scans shape: {lidar_scans.shape}")
    print(f"\nFrenet coordinate statistics:")
    print(f"  s range: [{frenet_coords[:, 0].min():.2f}, {frenet_coords[:, 0].max():.2f}] m")
    print(f"  d range: [{frenet_coords[:, 1].min():.2f}, {frenet_coords[:, 1].max():.2f}] m")
    print(f"  theta_error range: [{np.rad2deg(frenet_coords[:, 2].min()):.2f}, {np.rad2deg(frenet_coords[:, 2].max()):.2f}]°")

    # Save data with multiple representations
    save_path = conf.save_filename
    np.savez_compressed(
        save_path,
        # LiDAR observations
        scans=lidar_scans,
        
        # Frenet frame labels 
        frenet_labels=frenet_coords,      # [s, d, theta_error]
        
        # Cartesian frame labels (for absolute localization)
        cartesian_labels=cartesian_poses,  # [x, y, theta]
        
        # Prev format -> so that train_perception_map.py continues to work
        data_record=np.concatenate([cartesian_poses, lidar_scans], axis=1),
        
        # Track metadata
        track_length=np.array([total_track_length], dtype=np.float32),
        trajectory=trajectory.astype(np.float32),
        cumulative_lengths=cumulative_lengths.astype(np.float32),
        psi_rad=psi_rad.astype(np.float32)
    )

    print(f"\nSaved to {save_path}")
    print(f"File contents:")
    print(f"  - scans: LiDAR observations ({lidar_scans.shape})")
    print(f"  - frenet_labels: [s, d, theta_error] ({frenet_coords.shape})")
    print(f"  - cartesian_labels: [x, y, theta] ({cartesian_poses.shape})")
    print(f"  - data_record: prev commit format (cartesian + scans)")
    print(f"  - track_length: scalar ({total_track_length:.2f} m)")
    print(f"  - trajectory, cumulative_lengths, psi_rad: for conversions")

    print(f'Real elapsed time: {time.time()-start:.2f}s')


def debug_visualization():
    """Debug: visualize sampled Frenet points on the map"""
    import matplotlib.pyplot as plt
    from PIL import Image

    with open('../Levine/levine_slam.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # Load centerline (Levine format: s; x; y; psi; ...)
    waypoint_path = '../Levine/levine_centerline.csv'
    waypoints = load_waypoints(waypoint_path, delimiter=';', skiprows=3)
    trajectory = waypoints[:, 1:3].astype(np.float64)  # x, y
    cumulative_lengths = waypoints[:, 0].astype(np.float64)  # s
    total_track_length = cumulative_lengths[-1]
    psi_rad = waypoints[:, 3].astype(np.float64)  # psi

    # Load map
    map_img = Image.open('../Levine/levine_slam.pgm')
    map_array = np.array(map_img)

    # Map params
    resolution = conf.resolution
    origin_x, origin_y = conf.origin[0], conf.origin[1]
    img_height = map_array.shape[0]
    d_min = -0.8
    d_max = 0.8
    # Sample points along centerline (d=0) and with offsets
    sampled_points = []
    for s in np.linspace(0, total_track_length * 0.99, 50):
        for d in [d_min, 0.0, d_max]:
            x, y, theta = frenet_to_cartesian(s, d, 0.0, trajectory, cumulative_lengths, psi_rad)
            sampled_points.append([x, y, d])

    sampled_points = np.array(sampled_points)

    # Convert to pixels
    def world_to_pixel(x, y):
        px = (x - origin_x) / resolution
        py = img_height - (y - origin_y) / resolution
        return px, py

    # Plot
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(map_array, cmap='gray')

    # Plot centerline from file (ground truth)
    cl_px, cl_py = world_to_pixel(trajectory[:, 0], trajectory[:, 1])
    ax.plot(cl_px, cl_py, 'g-', linewidth=2, label='Centerline (from file)', alpha=0.7)

    # Plot sampled points
    colors = {d_min: 'blue', 0.0: 'red', d_max: 'cyan'}
    for d_val in [d_min, 0.0, d_max]:
        mask = sampled_points[:, 2] == d_val
        pts = sampled_points[mask]
        px, py = world_to_pixel(pts[:, 0], pts[:, 1])
        label = f'd={d_val} ({"right" if d_val < 0 else "center" if d_val == 0 else "left"})'
        ax.scatter(px, py, c=colors[d_val], s=20, label=label, alpha=0.8)

    ax.legend(loc='upper right')
    ax.set_title('Frenet Sampling Debug: Sampled Points vs Centerline')
    plt.savefig('frenet_debug.png', dpi=150)
    print("Saved debug visualization to frenet_debug.png")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'debug':
        debug_visualization()
    else:
        main()
