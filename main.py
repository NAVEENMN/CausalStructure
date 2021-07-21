#!/usr/bin/env python
# -*- coding: utf-8 -*-
from utils import Utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

observations = pd.read_csv('data/observations.csv')
springs = pd.read_csv('data/springs.csv')

#Utils.plots_trajectory()
#Utils.create_gif()

def plots():
    sns.histplot(springs.s_0_0)
    plt.show()
    sns.scatterplot(springs.s_0_1, observations.p_1_x_velocity)
    plt.show()
    sns.scatterplot(springs.s_0_1, observations.p_1_y_velocity)
    plt.show()
    sns.scatterplot(springs.s_1_2, observations.p_2_x_velocity)
    plt.show()
    sns.scatterplot(springs.s_1_2, observations.p_2_y_velocity)
    plt.show()
    sns.scatterplot(springs.s_0_1, observations.p_0_1_distance)
    plt.show()
    sns.scatterplot(springs.s_0_1, observations.p_0_2_distance)
    plt.show()
    sns.lineplot(observations.trajectory_step[:5000], observations.p_0_2_distance[:5000])
    plt.show()
    sns.lineplot(observations.trajectory_step[:5000], observations.p_0_1_distance[:5000])
    plt.show()

#plots()

datax = []
datay = []
for i in range(len(observations)):
    if i % 10000 == 0:
        datax.append(observations.p_0_x_position)
        datay.append(observations.p_0_y_position)
sns.scatterplot(datax, datay)
plt.show()