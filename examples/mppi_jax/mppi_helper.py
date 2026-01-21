#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np



from numba import njit

from functools import partial



# TODO CHECK: include needed ROS msg type headers and libraries
import math

import time


import yaml
from pathlib import Path

from mppi import MPPI

class oneLineJaxRNG:
    def __init__(self, init_num=0) -> None:
        self.rng = jax.random.PRNGKey(init_num)
        
    def new_key(self):
        self.rng, key = jax.random.split(self.rng)
        return key


class MPPIEnv():
    def __init__(self, waypoints, norm_param, n_steps, mode='st', DT=0.1) -> None:
        self.a_shape = 2

        self.waypoints = np.array(waypoints)
        # self.frenet_coord = FrenetCoord(jnp.array(waypoints))
        # self.diff = self.waypoints[1:, 1:3] - self.waypoints[:-1, 1:3]
        self.waypoints_distances = np.linalg.norm(self.waypoints[1:, (1, 2)] - self.waypoints[:-1, (1, 2)], axis=1)
        # print(self.waypoints_distances)
        self.n_steps = n_steps
        self.reference = None
        self.DT = DT
        self.dlk = self.waypoints[1,0] - self.waypoints[0, 0]

        self.params = {
                "lf": 0.15875,  # Distance from the center of gravity to the front axle [m]
                "lr": 0.17145,  # Distance from the center of gravity to the rear axle [m]
                "s_min": -0.4189,  # Minimum steering angle [rad]
                "s_max": 0.4189,  # Maximum steering angle [rad]
                "sv_min": -3.2,  # Minimum steering velocity [rad/s]
                "sv_max": 3.2,  # Maximum steering velocity [rad/s]
                "v_switch": 7.319,  # Switching velocity [m/s]
                "a_max": 9.51,  # Maximum acceleration [m/s^2]
                "v_min": -5.0,  # Minimum velocity [m/s]
                "v_max": 20.0,  # Maximum velocity [m/s]
                "width": 0.31,  # Vehicle width [m]
                "length": 0.58,  # Vehicle length [m]
            }
        
        # config.load_file(config.savedir + 'config.json')
        self.normalization_param = norm_param
        self.mode = mode
        # self.mb_dyna_pre = None
        def update_fn(x, u):
            x1 = x.copy()
            Ddt = 0.05
            def step_fn(i, x0):
                # # Forward euler
                # return x0 + vehicle_dynamics_st_trap([x0, u]) * Ddt

                # RK45
                k1 = self.vehicle_dynamics_ks(x0, u)
                k2 = self.vehicle_dynamics_ks(x0 + k1 * 0.5 * Ddt, u)
                k3 = self.vehicle_dynamics_ks(x0 + k2 * 0.5 * Ddt, u)
                k4 = self.vehicle_dynamics_ks(x0 + k3 * Ddt, u)
                return x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * Ddt
                
            x1 = jax.lax.fori_loop(0, int(self.DT/Ddt), step_fn, x1)
            return (x1, 0, x1-x)
        self.update_fn = update_fn


    
    ## Constraints handling
    def accl_constraints(self, vel, accl, v_switch, a_max, v_min, v_max):
        """
        Acceleration constraints, adjusts the acceleration based on constraints

            Args:
                vel (float): current velocity of the vehicle
                accl (float): unconstraint desired acceleration
                v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
                a_max (float): maximum allowed acceleration
                v_min (float): minimum allowed velocity
                v_max (float): maximum allowed velocity

            Returns:
                accl (float): adjusted acceleration
        """

        # positive accl limit
        # if vel > v_switch:
        #     pos_limit = a_max*v_switch/vel
        # else:
        #     pos_limit = a_max
        pos_limit = jax.lax.select(vel > v_switch, a_max*v_switch/vel, a_max)

        # accl limit reached?
        # accl = jax.lax.select(vel <= v_min and accl <= 0, 0., accl)
        # accl = jax.lax.select(vel >= v_max and accl >= 0, 0., accl)
        accl = jax.lax.select(jnp.all(jnp.asarray([vel <= v_min, accl <= 0])), 0., accl)
        accl = jax.lax.select(jnp.all(jnp.asarray([vel >= v_max, accl >= 0])), 0., accl)
        
        accl = jax.lax.select(accl <= -a_max, -a_max, accl)
        accl = jax.lax.select(accl >= pos_limit, pos_limit, accl)

        return accl
    
    def steering_constraint(self, steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
        """
        Steering constraints, adjusts the steering velocity based on constraints

            Args:
                steering_angle (float): current steering_angle of the vehicle
                steering_velocity (float): unconstraint desired steering_velocity
                s_min (float): minimum steering angle
                s_max (float): maximum steering angle
                sv_min (float): minimum steering velocity
                sv_max (float): maximum steering velocity

            Returns:
                steering_velocity (float): adjusted steering velocity
        """

        # constraint steering velocity
        steering_velocity = jax.lax.select(jnp.all(jnp.asarray([steering_angle <= s_min, steering_velocity <= 0])), 0., steering_velocity)
        steering_velocity = jax.lax.select(jnp.all(jnp.asarray([steering_angle >= s_max, steering_velocity >= 0])), 0., steering_velocity)
        # steering_velocity = jax.lax.select(steering_angle >= s_max and steering_velocity >= 0, 0., steering_velocity)
        steering_velocity = jax.lax.select(steering_velocity <= sv_min, sv_min, steering_velocity)
        steering_velocity = jax.lax.select(steering_velocity >= sv_max, sv_max, steering_velocity)
        # if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
        #     steering_velocity = 0.
        # elif steering_velocity <= sv_min:
        #     steering_velocity = sv_min
        # elif steering_velocity >= sv_max:
        #     steering_velocity = sv_max

        return steering_velocity
    
    ##Vehicle Dynamics models
    @partial(jax.jit, static_argnums=(0))
    def vehicle_dynamics_ks(self, x, u_init):
        """
        Single Track Kinematic Vehicle Dynamics.

            Args:
                x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5)
                    x1: x position in global coordinates
                    x2: y position in global coordinates
                    x3: steering angle of front wheels
                    x4: velocity in x direction
                    x5: yaw angle
                u (numpy.ndarray (2, )): control input vector (u1, u2)
                    u1: steering angle velocity of front wheels
                    u2: longitudinal acceleration

            Returns:
                f (numpy.ndarray): right hand side of differential equations
        """
        # wheelbase
        lf = self.params["lf"]
        lr = self.params["lr"]
        lwb = lf + lr
        # constraints
        s_min = self.params["s_min"]  # minimum steering angle [rad]
        s_max = self.params["s_max"]  # maximum steering angle [rad]
        # longitudinal constraints
        v_min = self.params["v_min"]  # minimum velocity [m/s]
        v_max = self.params["v_max"] # minimum velocity [m/s]
        sv_min = self.params["sv_min"] # minimum steering velocity [rad/s]
        sv_max = self.params["sv_max"] # maximum steering velocity [rad/s]
        v_switch = self.params["v_switch"]  # switching velocity [m/s]
        a_max = self.params["a_max"] # maximum absolute acceleration [m/s^2]

        # constraints
        u = jnp.array([self.steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), self.accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

        # system dynamics
        f = jnp.array([x[3]*jnp.cos(x[4]),
            x[3]*jnp.sin(x[4]), 
            u[0],
            u[1],
            x[3]/lwb*jnp.tan(x[2])])
        return f
                
    # @partial(jax.jit, static_argnums=(0,))
    def step(self, x, u):
        return self.update_fn(x, u * self.normalization_param)
        # return self.update_fn(x, u)
    
    
    @partial(jax.jit, static_argnums=(0,))
    def reward_fn(self, s, reference):
        

        xy_cost = -jnp.linalg.norm(reference[1:, :2] - s[:, :2], ord=1, axis=1)
        # vel_cost = -jnp.linalg.norm(reference[1:, 5] - s[:, 3])
        yaw_cost = -jnp.abs(jnp.sin(reference[1:, 3]) - jnp.sin(s[:, 4])) - \
            jnp.abs(jnp.cos(reference[1:, 3]) - jnp.cos(s[:, 4]))
        
        # adding terminal cost( useful sometimes)
        terminal_cost = -jnp.linalg.norm(reference[-1, :2] - s[-1, :2])
        return 15*xy_cost + 20*yaw_cost  
            
    
    # @partial(jax.jit, static_argnums=(0,))
    def reward(self, x):
        return 0
    
    def get_refernece_traj(self, state, target_speed=None, vind=5, speed_factor=1.0):
        _, dist, _, _, ind = nearest_point(np.array([state[0], state[1]]), 
                                           self.waypoints[:, (1, 2)].copy())
        
        if target_speed is None:
            # speed = self.waypoints[ind, vind] * speed_factor
            speed = np.minimum(self.waypoints[ind, vind] * speed_factor, 20.)
            # speed = state[3]
        else:
            speed = target_speed
        
        # if ind < self.waypoints.shape[0] - self.n_steps:
        #     speeds = self.waypoints[ind:ind+self.n_steps, vind]
        # else:
        speeds = np.ones(self.n_steps) * speed
        
        reference = self.get_reference_trajectory(speeds, dist, ind, 
                                            self.waypoints.copy(), int(self.n_steps),
                                            self.waypoints_distances.copy(), DT=self.DT)
        orientation = state[4]
        angle_thres = 5.0
        reference[3, :][reference[3, :] - orientation > angle_thres] = np.abs(
            reference[3, :][reference[3, :] - orientation > angle_thres] - (2 * np.pi))
        reference[3, :][reference[3, :] - orientation < -angle_thres] = np.abs(
            reference[3, :][reference[3, :] - orientation < -angle_thres] + (2 * np.pi))
        
        # reference[2] = np.where(reference[2] - speed > 5.0, speed + 5.0, reference[2])
        self.reference = reference.T
        return reference.T, ind
    
    def get_reference_trajectory(self, predicted_speeds, dist_from_segment_start, idx, 
                                waypoints, n_steps, waypoints_distances, DT):
        s_relative = np.zeros((n_steps + 1,))
        s_relative[0] = dist_from_segment_start
        s_relative[1:] = predicted_speeds * DT
        s_relative = np.cumsum(s_relative)

        waypoints_distances_relative = np.cumsum(np.roll(waypoints_distances, -idx))

        index_relative = np.int_(np.ones((n_steps + 1,)))
        for i in range(n_steps + 1):
            index_relative[i] = (waypoints_distances_relative <= s_relative[i]).sum()
        index_absolute = np.mod(idx + index_relative, waypoints.shape[0] - 1)

        segment_part = s_relative - (
                waypoints_distances_relative[index_relative] - waypoints_distances[index_absolute])

        t = (segment_part / waypoints_distances[index_absolute])
        # print(np.all(np.logical_and((t < 1.0), (t > 0.0))))

        position_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, (1, 2)] -
                            waypoints[index_absolute][:, (1, 2)])
        orientation_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 3] -
                                waypoints[index_absolute][:, 3])
        speed_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 5] -
                        waypoints[index_absolute][:, 5])

        interpolated_positions = waypoints[index_absolute][:, (1, 2)] + (t * position_diffs.T).T
        interpolated_orientations = waypoints[index_absolute][:, 3] + (t * orientation_diffs)
        interpolated_orientations = (interpolated_orientations + np.pi) % (2 * np.pi) - np.pi
        interpolated_speeds = waypoints[index_absolute][:, 5] + (t * speed_diffs)
        
        reference = np.array([
            # Sort reference trajectory so the order of reference match the order of the states
            interpolated_positions[:, 0],
            interpolated_positions[:, 1],
            interpolated_speeds,
            interpolated_orientations,
            # Fill zeros to the rest so number of references mathc number of states (x[k] - ref[k])
            np.zeros(len(interpolated_speeds)),
            np.zeros(len(interpolated_speeds)),
            np.zeros(len(interpolated_speeds))
        ])
        return reference

@njit(cache=True)
def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world
    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    dist_from_segment_start = np.linalg.norm(diffs[min_dist_segment] * t[min_dist_segment])
    return projections[min_dist_segment], dist_from_segment_start, dists[min_dist_segment], t[
        min_dist_segment], min_dist_segment

