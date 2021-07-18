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
        self.k = []
        self.num_particles = 0

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
        logging.info(f'Created a spring particle system with {num_of_particles} particles')

    def show_graph(self):
        self.p_graph.show()

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

    def simulate(self, total_time_steps, sample_freq, observations):
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
            _velocity = np.random.randn(2, num_particles)
            # Compute magnitude of this velocity vector and format to right shape
            v_norm = np.linalg.norm(_position, axis=0)
            # Scale by magnitude
            _velocity = _position * vel_norm / v_norm
            return _position, _velocity

        def get_force(k, current_positions):
            """
            :param k: Adjacency matrix representing mutual causality
            :param current_positions: current coordinates of all particles
            :return: net forces acting on all particles.
            TODO: Re verify this force computation
            """
            k = -1 * k
            np.fill_diagonal(k, 0)
            x_cords, y_cords = current_positions[0, :], current_positions[1, :]
            # we are interested in distance between particles not direction
            x_diffs = np.abs(np.subtract.outer(x_cords, x_cords))
            y_diffs = np.abs(np.subtract.outer(y_cords, y_cords))
            k = np.reshape(k, (self.num_particles, self.num_particles))

            # By Hooke's law Force = -k*dx
            fx = np.multiply(k, x_diffs)
            fy = np.multiply(k, y_diffs)
            # net forces acting on each particle along x dimension
            nfx = np.reshape(fx.sum(axis=0), (1, self.num_particles))
            # net forces acting on each particle along y dimension
            nfy = np.reshape(fy.sum(axis=0), (1, self.num_particles))
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
        velocity - current_velocity = dt * F
        velocity = current_velocity + (self._delta_T * F)
        '''
        velocity = init_velocity + (self._delta_T * init_force_between_particles)
        current_position = init_position
        for i in range(total_time_steps):
            # Compute new position based on current velocity and positions.
            new_position = current_position + (self._delta_T * velocity)
            # Compute forces between particles
            force_between_particles = get_force(self.k, new_position)
            # Compute new velocity based on current velocity and forces between particles.
            new_velocity = velocity + (self._delta_T * force_between_particles)
            # Update velocity and position
            velocity = new_velocity
            current_position = new_position
            # Add noise to observations
            current_position += np.random.randn(2, self.num_particles) * self.noise_variance
            velocity += np.random.randn(2, self.num_particles) * self.noise_variance
            # add to observations
            if i % sample_freq == 0:
                observation = {}
                # Adding all positions
                for i in range(len(current_position)):
                    for j in range(len(current_position[0])):
                        particle_id = j
                        if i == 0:
                            # x axis
                            observation[f'p_{particle_id}_x_position'] = current_position[i][j]
                        else:
                            # y axis
                            observation[f'p_{particle_id}_y_position'] = current_position[i][j]
                observations.add_an_observation(observation)


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
