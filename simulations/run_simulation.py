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
    def __init__(self, columns):
        self.column_names = columns
        self.observations = dict()
        self.observations = {label: [] for label in columns}

    def add_an_observation(self, observation):
        for col_name in observation.keys():
            self.observations[col_name].append(observation[col_name])

    def get_observations(self):
        df = pd.DataFrame(self.observations)
        return df

    def save_observations(self):
        logging.info("*** Saving: observations")
        df = pd.DataFrame(self.observations).set_index('trajectory_step')
        _dir = os.path.split(os.getcwd())[0]
        df.to_csv(os.path.join(_dir, 'data', 'observations.csv'))
        logging.info("*** Saved: observations")


def run_spring_particle_simulation(number_of_simulations=1):
    sp = SpringSystem()

    # Configure the particle system
    num_of_particles = 3
    sp.add_particles(num_of_particles)
    sp.set_initial_velocity_mean_sd((0.5, 1))

    # Configure the observations for recording
    column_names = ['trajectory_step']
    column_names.extend([f'p_{particle_id}_x_position' for particle_id in range(sp.get_particles_count())])
    column_names.extend([f'p_{particle_id}_y_position' for particle_id in range(sp.get_particles_count())])
    obs = Observations(columns=column_names)

    # Run simulation
    for i in range(number_of_simulations):
        logging.info(f'*** Running: Simulation {i}')

        spring_constants_matrix = np.random.rand(num_of_particles, num_of_particles)
        sp.add_springs(spring_constants_matrix=spring_constants_matrix)
        sp.remove_spring(particle_a=0, particle_b=1)
        # total_time_steps: run simulation with the current configuration for total_time_steps
        # sample_freq : make an observation for every sample_freq step.
        sp.simulate(total_time_steps=10000,
                    observations=obs,
                    sample_freq=20,
                    traj_id=i)
    logging.info(f'*** Complete: Simulation')

    # Save observations to a csv file
    obs.save_observations()


def main():
    run_spring_particle_simulation(number_of_simulations=5)


if __name__ == "__main__":
    main()
