"""
Example script for MPPI planner with F1TENTH Gym
Similar to waypoint_follow.py but uses MPPI instead of Pure Pursuit
"""

import time
import sys
import os
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import numpy as np
from argparse import Namespace

# Add mppi_jax to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mppi_jax'))

from mppi_jax.mppi_planner_gym import MPPIPlanner


def main():
    """
    Main entry point
    """
    # Load configuration
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # Initialize MPPI planner
    # You can create waypoints or use the same format as Pure Pursuit
    waypoint_path = conf.wpt_path  # Use same waypoints as Pure Pursuit
    wheelbase = 0.17145 + 0.15875

    print("\n" + "="*60)
    print("Initializing MPPI Planner")
    print("="*60)

    planner = MPPIPlanner(
        waypoint_path=waypoint_path,
        config_path=None,  # Optional: path to MPPI config YAML
        wheelbase=wheelbase
    )

    # Rendering callback
    def render_callback(env_renderer):
        e = env_renderer

        # Update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        planner.render_waypoints(env_renderer)

    # Create gym environment
    env = gym.make(
        'f110_gym:f110-v0',
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4
    )
    env.add_render_callback(render_callback)

    # Reset environment
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    laptime = 0.0
    start = time.time()

    # Control parameters
    current_speed = 0.0
    current_steering = 0.0

    print("\n" + "="*60)
    print("Starting MPPI Control Loop")
    print("="*60)

    step_count = 0
    compute_times = []

    while not done:
        step_start = time.time()

        # Get control action from MPPI
        speed, steering = planner.plan(
            obs['poses_x'][0],
            obs['poses_y'][0],
            obs['poses_theta'][0],
            current_speed=current_speed,
            current_steering=current_steering
        )

        # Store for next iteration
        current_speed = speed
        current_steering = steering

        # Step environment
        obs, step_reward, done, info = env.step(np.array([[steering, speed]]))
        laptime += step_reward
        env.render(mode='human')

        # Track compute time
        compute_time = time.time() - step_start
        compute_times.append(compute_time)

        step_count += 1

        # Print status every 100 steps
        if step_count % 100 == 0:
            avg_compute_time = np.mean(compute_times[-100:])
            print(f"Step {step_count}: Speed={speed:.2f}, Steering={steering:.3f}, "
                  f"Avg compute time={avg_compute_time*1000:.2f}ms")

    total_time = time.time() - start

    print("\n" + "="*60)
    print("Run Complete")
    print("="*60)
    print(f'Sim elapsed time: {laptime:.2f}s')
    print(f'Real elapsed time: {total_time:.2f}s')
    print(f'Total steps: {step_count}')
    print(f'Average compute time: {np.mean(compute_times)*1000:.2f}ms')
    print(f'Max compute time: {np.max(compute_times)*1000:.2f}ms')
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
