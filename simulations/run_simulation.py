#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulates a chosen system
(Spring Particle, Charge Particle or Gravity Particles)
writes observational data and schema to /data
"""
import os
import logging
import pandas as pd
import numpy as np
from particle_system import SpringSystem

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


class Observations(object):
    def __init__(self):
        self.column_names = []
        self.observations = dict()

    def set_column_names(self, columns):
        self.column_names = columns
        self.observations = {label: [] for label in columns}

    def add_an_observation(self, observation):
        for col_name in observation.keys():
            self.observations[col_name].append(observation[col_name])

    def get_observations(self):
        df = pd.DataFrame(self.observations)
        return df

    def save_observations(self, name):
        logging.info("*** Saving: observations")
        df = pd.DataFrame(self.observations).set_index('trajectory_step')
        _dir = os.path.split(os.getcwd())[0]
        df.to_csv(os.path.join(_dir, 'data', f'{name}.csv'))
        logging.info(f"*** Saved: observations {name}.csv")


def run_spring_particle_simulation(number_of_simulations=1):
    num_of_particles = 3

    # Create Observation records
    particle_observations = Observations()
    spring_observations = Observations()

    # Configure the observations for recording
    column_names = ['trajectory_step']
    column_names.extend([f'p_{particle_id}_x_position' for particle_id in range(num_of_particles)])
    column_names.extend([f'p_{particle_id}_y_position' for particle_id in range(num_of_particles)])
    column_names.extend([f'p_{particle_id}_x_velocity' for particle_id in range(num_of_particles)])
    column_names.extend([f'p_{particle_id}_y_velocity' for particle_id in range(num_of_particles)])
    for i in range(num_of_particles):
        for j in range(num_of_particles):
            column_names.append(f'p_{i}_{j}_distance')
    particle_observations.set_column_names(columns=column_names)

    spring_observation_columns = ['trajectory_step']
    for i in range(num_of_particles):
        for j in range(num_of_particles):
            spring_observation_columns.append(f's_{i}_{j}')
    spring_observations.set_column_names(columns=spring_observation_columns)

    # Run simulation
    for i in range(number_of_simulations):
        # Configure the particle system
        sp = SpringSystem()
        sp.add_particles(num_of_particles)
        sp.set_initial_velocity_mean_sd((0.0, 0.0001))
        logging.info(f'*** Running: Simulation {i}')
        sp.add_a_spring(particle_a=1, particle_b=2, spring_constant=np.random.normal(2, 0.5, 1))
        # total_time_steps: run simulation with the current configuration for total_time_steps
        # sample_freq : make an observation for every sample_freq step.
        # For a good trajectory longer time_steps recommended
        sp.simulate(total_time_steps=100000,
                    observations=particle_observations,
                    spring_observations=spring_observations,
                    sample_freq=100,
                    traj_id=i)
    logging.info(f'*** Complete: Simulation')

    # Save observations to a csv file
    particle_observations.save_observations(name='observations')
    spring_observations.save_observations(name='springs')


def main():
    run_spring_particle_simulation(number_of_simulations=100)


if __name__ == "__main__":
    main()
