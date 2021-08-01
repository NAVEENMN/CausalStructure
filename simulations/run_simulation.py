#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulates a chosen system
(Spring Particle, Charge Particle or Gravity Particles)
writes observational data and schema to /data
"""
import os
import time
import logging
import pandas as pd
import numpy as np
from particle_system import SpringSystem
from multiprocessing import Process

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
        # index represents most reset observation
        df = df[::-1]
        _dir = os.path.split(os.getcwd())[0]
        df.to_csv(os.path.join(_dir, 'data/simulations', f'{name}.csv'))
        logging.info(f"*** Saved: observations {name}.csv")


def run_spring_particle_simulation(_id=0):
    # *** Control Variables ***
    num_of_particles = 4
    trajectory_length = 100000
    sample_freq = 50
    # Zero implies static edges
    period = 500
    # ********

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
    # Configure the particle system
    sp = SpringSystem()
    sp.add_particles(num_of_particles)
    sp.set_initial_velocity_mean_sd((0.0, 0.0001))
    logging.info(f'*** Running: Simulation {_id}')

    # *** Control Variable ***
    sp.add_a_spring(particle_a=0,
                    particle_b=1,
                    spring_constant=np.random.normal(2, 0.5, 1))
    # ********

    # total_time_steps: run simulation with the current configuration for total_time_steps
    # sample_freq : make an observation for every sample_freq step.
    # For a good trajectory longer time_steps recommended
    sp.simulate(total_time_steps=trajectory_length,
                period=period,
                observations=particle_observations,
                spring_observations=spring_observations,
                sample_freq=sample_freq,
                traj_id=_id)
    logging.info(f'*** Complete: Simulation')

    # Save observations to a csv file
    particle_observations.save_observations(name=f'observations_{_id}')
    spring_observations.save_observations(name=f'springs_{_id}')


def main():
    start = time.time()
    from multiprocessing import Pool
    number_of_simulations = range(10)
    with Pool(4) as p:
        p.map(run_spring_particle_simulation, number_of_simulations)
    print(f'Total time taken: {time.time() - start}')


if __name__ == "__main__":
    main()
