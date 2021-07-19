#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data/observations.csv')

sns.pairplot(data)
plt.show()
