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


def load_waypoints(waypoint_path):
    """
    Load waypoints from CSV file.

    Expected format: s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
    """
    # waypoints = np.loadtxt(waypoint_path, delimiter=';', skiprows=3)
    waypoints = np.loadtxt(waypoint_path, delimiter=',', skiprows=1)
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

    Frenet sampling allows us to:
    1. Sample uniformly along the track (s coordinate)
    2. Sample lateral offsets within track bounds (d coordinate)
    3. Sample heading variations relative to track direction (theta_error)
    """

    with open('../Monza/Monza_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)


    # Load waypoints for Frenet conversion
    waypoint_path = conf.wpt_path if hasattr(conf, 'wpt_path') else '../Monza/Monza_centerline.csv'
    waypoints = load_waypoints(waypoint_path)

    # Extract trajectory data
    # Centerline format: [x, y, w_tr_right, w_tr_left]
    trajectory = waypoints[:, 0:2].astype(np.float64)  # x, y

    # Compute cumulative arc lengths from trajectory
    cumulative_lengths = compute_cumulative_lengths(trajectory)
    total_track_length = cumulative_lengths[-1]

    # Compute heading (psi) from trajectory direction
    diffs = trajectory[1:] - trajectory[:-1]
    psi_rad = np.zeros(len(trajectory))
    psi_rad[:-1] = np.arctan2(diffs[:, 1], diffs[:, 0])
    psi_rad[-1] = psi_rad[-2]  # Last point same as second-to-last
    psi_rad = psi_rad.astype(np.float64)

    print(f"Track length: {total_track_length:.2f} m")
    print(f"Number of waypoints: {len(trajectory)}")

    # Frenet sampling parameters
    s_min = 0.0
    s_max = total_track_length
    d_min = -1.0   # 1 meter right of centerline
    d_max = 1.0    # 1 meter left of centerline
    theta_error_min = -1*np.pi/4         # Full rotation range
    theta_error_max = 1 * np.pi/4    # 0 to 360 degrees

    print(f"\nFrenet sampling bounds:")
    print(f"  s: [{s_min:.1f}, {s_max:.1f}] m")
    print(f"  d: [{d_min:.1f}, {d_max:.1f}] m")
    print(f"  theta_error: [{np.rad2deg(theta_error_min):.1f}, {np.rad2deg(theta_error_max):.1f}] deg")

    # Initialize environment
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)

    start = time.time()
    cnt = 0
    data_record = []

    with tqdm(total=conf.sample_num) as pbar:
        while cnt < conf.sample_num:
            # Sample in Frenet frame
            s = np.random.uniform(s_min, s_max)
            d = np.random.uniform(d_min, d_max)
            theta_error = np.random.uniform(theta_error_min, theta_error_max)

            # Convert to Cartesian using waypoint headings
            x, y, theta = frenet_to_cartesian(s, d, theta_error, trajectory, cumulative_lengths, psi_rad)

            sample_pos = [x, y, theta]

            # Reset environment at sampled position
            obs, step_reward, done, info = env.reset(np.array([sample_pos]))

            # Check if position is valid (not in collision)
            if not done:
                # Record Cartesian pose and LiDAR scan
                pose = np.array([obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]])
                lidar_scan = np.array(obs['scans'][0])[np.arange(0, 1080, 3)]  # Downsample to 360

                data_record.append(np.concatenate([pose, lidar_scan]))

                cnt += 1
                pbar.update(1)

    data_record = np.array(data_record)

    print(f"\nData shape: {data_record.shape}")

    # Save data
    np.savez_compressed(
        conf.save_filename,
        data_record=data_record,
    )
    print(f"Saved to {conf.save_filename}")

    print(f'Real elapsed time: {time.time()-start:.2f}s')


def debug_visualization():
    """Debug: visualize sampled Frenet points on the map"""
    import matplotlib.pyplot as plt
    from PIL import Image

    with open('../Monza/Monza_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # Load centerline
    waypoint_path = '../Monza/Monza_centerline.csv'
    waypoints = load_waypoints(waypoint_path)
    trajectory = waypoints[:, 0:2].astype(np.float64)
    cumulative_lengths = compute_cumulative_lengths(trajectory)
    total_track_length = cumulative_lengths[-1]

    # Compute heading
    diffs = trajectory[1:] - trajectory[:-1]
    psi_rad = np.zeros(len(trajectory))
    psi_rad[:-1] = np.arctan2(diffs[:, 1], diffs[:, 0])
    psi_rad[-1] = psi_rad[-2]
    psi_rad = psi_rad.astype(np.float64)

    # Load map
    map_img = Image.open('../Monza/Monza_map.png')
    map_array = np.array(map_img)

    # Map params
    resolution = conf.resolution
    origin_x, origin_y = conf.origin[0], conf.origin[1]
    img_height = map_array.shape[0]

    # Sample points along centerline (d=0) and with offsets
    sampled_points = []
    for s in np.linspace(0, total_track_length * 0.99, 50):
        for d in [-1.0, 0.0, 1.0]:
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
    colors = {-1.0: 'blue', 0.0: 'red', 1.0: 'cyan'}
    for d_val in [-1.0, 0.0, 1.0]:
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
