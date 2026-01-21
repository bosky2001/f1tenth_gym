"""
MPPI Planner for F1TENTH Gym Environment
Gym-compatible version without ROS dependencies
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from mppi import MPPI
from mppi_helper import MPPIEnv, oneLineJaxRNG
import yaml


class MPPIConfig:
    """Configuration class for MPPI planner"""
    def __init__(self):
        # MPPI parameters
        self.n_iterations = 1
        self.n_steps = 20
        self.n_samples = 1000
        self.control_dim = 2
        self.adaptive_covariance = False
        self.normalization_param = [3.2, 9.51]  # [steering_vel, acceleration]

        # Vehicle parameters
        self.lf = 0.15875  # front axle distance
        self.lr = 0.17145  # rear axle distance

    def load_from_yaml(self, yaml_path):
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Update parameters from YAML if they exist
        if 'mppi' in config_dict:
            mppi_config = config_dict['mppi']
            self.n_iterations = mppi_config.get('n_iterations', self.n_iterations)
            self.n_steps = mppi_config.get('n_steps', self.n_steps)
            self.n_samples = mppi_config.get('n_samples', self.n_samples)
            self.normalization_param = mppi_config.get('normalization_param', self.normalization_param)


class MPPIPlanner:
    """
    MPPI (Model Predictive Path Integral) Planner for F1TENTH
    Compatible with gym environment interface
    """
    def __init__(self, waypoint_path, config_path=None, wheelbase=None):
        """
        Initialize MPPI planner

        Args:
            waypoint_path: Path to waypoint CSV file
            config_path: Optional path to YAML config file
            wheelbase: Vehicle wheelbase (lf + lr)
        """
        # Load configuration
        self.config = MPPIConfig()
        if config_path:
            try:
                self.config.load_from_yaml(config_path)
            except FileNotFoundError:
                print(f"Config file {config_path} not found, using defaults")

        # Load waypoints
        self.waypoints = self.load_waypoints(waypoint_path)

        # MPPI parameters
        self.n_iterations = self.config.n_iterations
        self.n_steps = self.config.n_steps
        self.n_samples = self.config.n_samples
        self.DT = 0.1  # timestep

        # Initialize JAX random number generator
        self.jRNG = oneLineJaxRNG(1337)

        # Normalization parameters for control inputs
        self.normalization_param = np.array(self.config.normalization_param).T
        norm_param = self.normalization_param / 2
        self.norm_param = norm_param

        # Initialize MPPI environment and controller
        self.mppi_env = MPPIEnv(self.waypoints, norm_param, self.n_steps, mode='ks', DT=self.DT)
        self.mppi = MPPI(self.config, jRNG=self.jRNG, a_noise=1.0, scan=False)

        # Control state
        self.a_opt = None
        self.a_cov = None
        self.mppi_distrib = None

        # Target velocity
        self.target_vel = 3.0

        # Previous control inputs (for integration)
        self.prev_steering_angle = 0.0
        self.prev_speed = 0.0

        # Initialize MPPI state
        self.init_state()

        print("MPPI Planner initialized successfully")

    def load_waypoints(self, path):
        """
        Load waypoints from CSV file

        Args:
            path: Path to waypoint file

        Returns:
            numpy array of waypoints
        """
        try:
            points = np.loadtxt(path, delimiter=';', skiprows=3, dtype=np.float64)
            # Adjust theta by pi/2 if needed (track-dependent)
            points[:, 3] += 0.5 * np.pi
            print(f"Loaded {len(points)} waypoints from {path}")
            return points
        except Exception as e:
            print(f"Error loading waypoints: {e}")
            print("Using Pure Pursuit waypoint format instead...")
            # Try Pure Pursuit format (no delimiter, different structure)
            points = np.loadtxt(path, delimiter=',', skiprows=0)
            # Reformat to MPPI expected format if needed
            # Assuming columns: [s, x, y, theta, velocity]
            if points.shape[1] < 6:
                # Pad with zeros or duplicate velocity column
                points = np.column_stack([
                    np.arange(len(points)),  # s
                    points[:, 0],  # x
                    points[:, 1],  # y
                    points[:, 2] if points.shape[1] > 2 else np.zeros(len(points)),  # theta
                    np.zeros(len(points)),  # placeholder
                    points[:, 3] if points.shape[1] > 3 else np.ones(len(points)) * 2.0  # velocity
                ])
            return points

    def init_state(self):
        """Initialize MPPI state"""
        self.mppi.init_state(self.mppi_env)
        self.a_opt = self.mppi.a_opt
        self.a_cov = self.mppi.a_cov
        self.mppi_distrib = (self.a_opt, self.a_cov)

    def plan(self, pose_x, pose_y, pose_theta, current_speed=None, current_steering=None):
        """
        Plan control action using MPPI

        Args:
            pose_x: Current x position
            pose_y: Current y position
            pose_theta: Current yaw angle
            current_speed: Current velocity (optional, uses previous if None)
            current_steering: Current steering angle (optional, uses previous if None)

        Returns:
            speed: Target speed
            steering_angle: Target steering angle
        """
        # Use previous values if not provided
        if current_speed is None:
            current_speed = self.prev_speed
        if current_steering is None:
            current_steering = self.prev_steering_angle

        # Construct state vector: [x, y, steering_angle, velocity, yaw]
        state = np.array([pose_x, pose_y, current_steering, current_speed, pose_theta])

        # Get reference trajectory
        ref_traj, _ = self.mppi_env.get_refernece_traj(
            state,
            target_speed=self.target_vel,
            vind=5,
            speed_factor=1.0
        )

        # Run MPPI optimization
        self.mppi_distrib, sampled_traj, s_opt = self.mppi.update(
            self.mppi_env,
            state.copy(),
            self.jRNG.new_key()
        )

        # Extract optimal control
        a_opt = self.mppi_distrib[0]
        control = a_opt[0]  # First control in horizon
        scaled_control = np.multiply(self.norm_param, control)

        # Unpack control: [steering_velocity, acceleration]
        steer_vel = scaled_control[0]
        accl = scaled_control[1]

        # Integrate to get absolute commands
        cmd_steering_angle = current_steering + steer_vel * self.DT
        cmd_speed = current_speed + accl * self.DT

        # Apply constraints
        cmd_steering_angle = np.clip(cmd_steering_angle, -0.4189, 0.4189)
        cmd_speed = np.clip(cmd_speed, 0.0, 6.0)

        # Store for next iteration
        self.prev_steering_angle = cmd_steering_angle
        self.prev_speed = cmd_speed

        return cmd_speed, cmd_steering_angle

    def render_waypoints(self, env_renderer):
        """
        Render waypoints for visualization (optional, gym compatible)

        Args:
            env_renderer: Gym environment renderer
        """
        # This can be implemented similar to PurePursuitPlanner if needed
        # For now, just pass
        pass
