#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
import networkx as nx
from graph import ParticleGraph

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class Environment(object):
    def __init__(self):
        self.box_size = 5.0
        self._delta_T = 0.001
        self.dimensions = 2
        self._positions = []
        self._velocities = []

    def reset(self):
        self._positions.clear()
        self._velocities.clear()

    def add_a_particle(self):
        pass

    def get_positions(self):
        return self._positions

    def get_velocities(self):
        return self._velocities


class SpringSystem(Environment):
    def __init__(self):
        super().__init__()
        self.p_graph = ParticleGraph()
        self.noise_variance = 0.5
        self.init_velocity_mean_sd = (0.01, 1)
        self.k = np.asarray([])
        self.num_particles = 0
        self.noise = (0.5, 1)
        # In space the periodic interactions between particles go on to infinity
        # In real world however system would loose energy slows down by damp_factor.
        # (rapid) 0.0 < damp < 1.0 (no damp)
        self.damp_factor = 1.0

    def set_initial_velocity_mean_sd(self, m_sd):
        self.init_velocity_mean_sd = m_sd

    def set_damp_factor(self, value):
        self.damp_factor = value

    def set_noise_mean_sd(self, m_sd):
        self.noise = m_sd

    def get_particle_names(self):
        return self.p_graph.get_node_names()

    def get_particles_count(self):
        return self.num_particles

    def get_column_names(self):
        self.p_graph.get_node_names()

    def add_particles(self, num_of_particles=0):
        logging.debug(f'Creating a spring particle system with {num_of_particles} particles')
        for _ in range(num_of_particles):
            self.p_graph.add_particle_node_to_graph()
        self.num_particles = self.p_graph.get_total_number_of_particles()
        logging.debug(f'Initialized springs to 0.0')
        # initialize springs
        self.k = np.zeros((self.num_particles, self.num_particles))
        logging.info(f'Created a spring particle system with {num_of_particles} particles')

    def show_graph(self):
        self.p_graph.show()

    def add_a_spring(self, particle_a, particle_b, spring_constant):
        num_of_particles = self.p_graph.get_total_number_of_particles()
        if num_of_particles == 0:
            logging.error('Environment has no particles to add a spring')
            return

        self.k[particle_a][particle_b] = spring_constant
        self.k[particle_b][particle_a] = spring_constant
        self.p_graph.add_spring_to_graph(particle_a=particle_a,
                                         particle_b=particle_b,
                                         spring_constant=spring_constant)
        logging.info(f'Added spring to a {particle_a} {particle_b} : {spring_constant}')

    def add_springs(self, spring_constants_matrix):

        num_of_particles = self.p_graph.get_total_number_of_particles()

        if num_of_particles == 0:
            logging.error('Environment has no particles to add a spring')
            return

        if spring_constants_matrix.shape != (num_of_particles, num_of_particles):
            logging.error('Shapes of spring constants matrix and number of particles wont match')
            return

        # Establish symmetry
        spring_constants_matrix = np.tril(spring_constants_matrix) + np.tril(spring_constants_matrix, -1).T

        # Nullify self interaction or causality
        np.fill_diagonal(spring_constants_matrix, 0)
        self.k = spring_constants_matrix
        self.p_graph.add_springs_to_graph(spring_constant_matrix=self.k)
        logging.info(f'Added springs to a spring particle system')

    def remove_spring(self, particle_a, particle_b):
        self.k[particle_a][particle_b] = 0.0
        logging.info(f'Removed Spring p_{particle_a} p_{particle_b}')

    def simulate(self, total_time_steps, sample_freq, observations, spring_observations, traj_id):
        num_particles = self.p_graph.get_total_number_of_particles()
        if num_particles == 0:
            logging.warning('Nothing to simulate, add particles')
            return

        def get_init_pos_velocity():
            """
            This function samples position and velocity from a distribution.
            These position and velocity will be used as
            initial position and velocity for all particles.
            :return: initial position and velocity
            """
            loc_std = 0.5
            vel_norm = 0.5
            _position = np.random.randn(2, num_particles) * loc_std
            # sample initial velocity from normal distribution
            _mv = np.random.normal(self.init_velocity_mean_sd[0],
                                   self.init_velocity_mean_sd[1], 1)
            logging.info(f'Initial velocity set to {_mv}')
            _velocity = (_mv + np.random.randn(2, num_particles)) * 0.01
            # Compute magnitude of this velocity vector and format to right shape
            #v_norm = np.linalg.norm(_position, axis=0)
            # Scale by magnitude
            #_velocity = _position * vel_norm / v_norm
            return _position, _velocity

        def get_force1(_edges, current_positions):
            """
            :param _edges: Adjacency matrix representing mutual causality
            :param current_positions: current coordinates of all particles
            :return: net forces acting on all particles.
            """
            force_matrix = - 0.5 * _edges
            np.fill_diagonal(force_matrix, 0)
            x_cords, y_cords = current_positions[0, :], current_positions[1, :]

            #x_diffs = np.subtract.outer(x_cords, x_cords)
            #y_diffs = np.subtract.outer(y_cords, y_cords)
            #distance_matrix = np.sqrt(np.square(x_diffs) + np.square(y_diffs))
            # By Hooke's law Force = -k * dx
            #force_matrix = np.multiply(-0.5 * _edges, distance_matrix)

            x_diffs = np.subtract.outer(x_cords, x_cords).reshape(1, self.num_particles, self.num_particles)
            y_diffs = np.subtract.outer(y_cords, y_cords).reshape(1, self.num_particles, self.num_particles)
            force_matrix = force_matrix.reshape(1, self.num_particles, self.num_particles)
            _force = (force_matrix * np.concatenate((x_diffs, y_diffs))).sum(axis=-1)
            return _force

        def get_force(k, current_positions):
            """
            :param k: Adjacency matrix representing mutual causality
            :param current_positions: current coordinates of all particles
            :return: net forces acting on all particles.
            TODO: Re verify this force computation
            """
            np.fill_diagonal(k, 0)
            x_cords, y_cords = current_positions[0, :], current_positions[1, :]

            # we are interested in distance between particles not direction
            x_diffs = np.subtract.outer(x_cords, x_cords)
            y_diffs = np.subtract.outer(y_cords, y_cords)
            distance_matrix = np.sqrt(np.square(x_diffs) + np.square(y_diffs))

            # By Hooke's law Force = -k * dx
            force = np.multiply(-k, distance_matrix)
            force_direction = np.full(force.shape, -2)
            force_direction = np.tril(force_direction, k=0)
            np.fill_diagonal(force_direction, 0)
            force_direction = np.add(force_direction, np.ones(force.shape))
            force = np.multiply(force, force_direction)

            # get force components
            poc_vec = current_positions / np.linalg.norm(current_positions, axis=0)
            x_intep, y_intep = poc_vec[0], poc_vec[1]

            # slope = (y2-y1)/(x2-x1)
            # tan(theta) = slope
            dif_y = np.subtract.outer(y_intep, y_intep)
            dif_x = np.subtract.outer(x_intep, x_intep)
            slopes = np.divide(dif_y, dif_x)
            slopes = np.nan_to_num(slopes)
            theta = np.arctan(slopes)

            horizontal_components = np.multiply(force, np.cos(theta))
            vertical_components = np.multiply(force, np.sin(theta))

            # net forces acting on each particle along x dimension
            nfx = np.reshape(horizontal_components.sum(axis=0), (1, self.num_particles))
            # net forces acting on each particle along y dimension
            nfy = np.reshape(vertical_components.sum(axis=0), (1, self.num_particles))
            # package the results as (2 * num of particles)
            _force = np.concatenate((nfx, nfy), axis=0)
            return _force

        # Initialize the first position and velocity from a distribution
        init_position, init_velocity = get_init_pos_velocity()

        # Compute initial forces between particles.
        init_force_between_particles = get_force(self.k, init_position)

        # Compute new velocity.
        '''
        F = m * a, for unit mass force is acceleration
        F = a = dv/dt
        dv = dt * F
        current_velocity - velocity = dt * F
        velocity = current_velocity - (self._delta_T * F)
        '''
        velocity = init_velocity + (self._delta_T * init_force_between_particles)
        current_position = init_position
        step = 0
        for i in range(total_time_steps):
            # Compute forces between particles
            force_between_particles = get_force1(self.k, current_position)

            # Compute new velocity based on current velocity and forces between particles.
            new_velocity = velocity + (self._delta_T * force_between_particles)

            # Compute new position based on current velocity and positions.
            # dx/dt = v
            # (current_position - new_position) = dt * v
            new_position = current_position + (self._delta_T * new_velocity)

            # Update velocity and position
            velocity = new_velocity
            current_position = new_position
            # Add noise to observations
            #current_position += np.random.randn(2, self.num_particles) * self.noise_variance
            #velocity += np.random.randn(2, self.num_particles) * self.noise_variance
            # add to observations

            if i % sample_freq == 0:
                self.add_observation(observations, spring_observations, traj_id, step, self.k, current_position, velocity)
                step += 1

    def add_observation(self, observations, spring_observations, traj_id, step, springs, positions, velocity):
        # local dict to collect readings
        observation = {}
        sp_observation = {}

        # Adding all positions, velocity
        for i in range(len(positions)):
            for j in range(len(positions[0])):
                particle_id = j
                if i == 0:
                    # x axis
                    observation[f'p_{particle_id}_x_position'] = positions[i][j]
                    observation[f'p_{particle_id}_x_velocity'] = velocity[i][j]
                else:
                    # y axis
                    observation[f'p_{particle_id}_y_position'] = positions[i][j]
                    observation[f'p_{particle_id}_y_velocity'] = velocity[i][j]

        # Calculate and store distance
        x_cords, y_cords = positions[0, :], positions[1, :]
        x_diffs = np.subtract.outer(x_cords, x_cords)
        y_diffs = np.subtract.outer(y_cords, y_cords)
        distance_matrix = np.sqrt(np.square(x_diffs) + np.square(y_diffs))
        for i in range(len(distance_matrix)):
            for j in range(len(distance_matrix[0])):
                observation[f'p_{i}_{j}_distance'] = distance_matrix[i][j]

        for i in range(self.num_particles):
            for j in range(self.num_particles):
                sp_observation[f's_{i}_{j}'] = springs[i][j]

        observation[f'trajectory_step'] = f'{traj_id}_{step}'
        sp_observation[f'trajectory_step'] = f'{traj_id}_{step}'
        # Add observation to global record
        observations.add_an_observation(observation)
        spring_observations.add_an_observation(sp_observation)

def test():
    sp = SpringSystem()
    sp.add_particles(num_of_particles=3)
    spring_constants_matrix = np.asarray([[0, 0, 1],
                                          [0, 0, 0],
                                          [1, 0, 0]])
    #spring_constants_matrix = np.random.rand(5, 5)
    sp.add_springs(spring_constants_matrix=spring_constants_matrix)
    sp.show_graph()


if __name__ == "__main__":
    test()
