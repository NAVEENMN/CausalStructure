#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merges all observations dumped inside data/simulations
"""
import os
import glob
import pandas as pd

observations = None
flag = False
for name in glob.glob('../data/simulations/observations_*.csv'):
    print(f'*** Reading {name}')
    if not flag:
        observations = pd.read_csv(name)
        flag = True
    else:
        observations = observations.append(pd.read_csv(name))
    print(f'*** Deleting {name}')
    os.remove(name)
print(f"*** Saving: observations {observations.shape}")
df = pd.DataFrame(observations).set_index('trajectory_step')
_dir = os.path.split(os.getcwd())[0]
df.to_csv(os.path.join(_dir, 'data', f'observations.csv'))
print(f"*** Saved: data/observations.csv")

springs = None
flag = False
for name in glob.glob('../data/simulations/springs_*.csv'):
    print(f'*** Reading {name}')
    if not flag:
        springs = pd.read_csv(name)
        flag = True
    else:
        springs = springs.append(pd.read_csv(name))
    print(f'*** Deleting {name}')
    os.remove(name)
print(f"*** Saving: springs {springs.shape}")
df = pd.DataFrame(springs).set_index('trajectory_step')
_dir = os.path.split(os.getcwd())[0]
df.to_csv(os.path.join(_dir, 'data', f'springs.csv'))
print(f"*** Saved: data/springs.csv")
