#!/usr/bin/env python
# -*- coding: utf-8 -*-
from utils import Utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
observations = pd.read_csv('data/observations.csv')
springs = pd.read_csv('data/springs.csv')

Utils.plots_trajectory()
Utils.create_gif()

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

def pcm_test():
    import numpy as np
    from tigramite import data_processing as pp
    np.random.seed(42)     # Fix random seed
    links_coeffs = {0: [((0, -1), 0.7), ((1, -1), -0.8)],
                    1: [((1, -1), 0.8), ((3, -1), 0.8)],
                    2: [((2, -1), 0.5), ((1, -2), 0.5), ((3, -3), 0.6)],
                    3: [((3, -1), 0.4)],
                    }
    T = 1000     # time series length
    data, true_parents_neighbors = pp.var_process(links_coeffs, T=T)
    print(data.shape)
    print(data)
    # Initialize dataframe object, specify time axis and variable names
    var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$']
    dataframe = pp.DataFrame(data,
                             datatime=np.arange(len(data)),
                             var_names=var_names)
    print(dataframe)

def pcmci():
    from tigramite.independence_tests import ParCorr
    from tigramite.pcmci import PCMCI
    from tigramite import plotting as tp
    import numpy as np

    # Extract variables of interests
    pd_data = observations[['p_0_x_position', 'p_1_x_position', 'p_2_x_position']]

    # Convert pandas dataframe to tigramite dataframe
    from tigramite import data_processing as pp
    # Initialize dataframe object, specify time axis and variable names
    var_names = pd_data.columns.values
    dataframe = pp.DataFrame(pd_data.values,
                             datatime=np.arange(len(pd_data)),
                             var_names=var_names)
    tp.plot_timeseries(dataframe); plt.show()
    pc_mci = PCMCI(dataframe=dataframe,
                   cond_ind_test=ParCorr(),
                   verbosity=2)

    results = pc_mci.run_pcmci(tau_max=100)
    print(results)


#plots()
#pcmci()