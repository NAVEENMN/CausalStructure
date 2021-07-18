#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

class Utils(object):

    @classmethod
    def plots_trajectory(cls, loc='data/observations.csv'):
        import seaborn as sns
        # Pick a random subsequence from observations and plot trajectory
        observations = pd.read_csv(loc)
        particle_count = 0
        for col in observations.columns:
            if 'position' in col:
                particle_count += 1
        particle_count /= 2
        particle_count = int(particle_count)
        for time_step in range(0, observations.shape[0]):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].set_title('Position')
            axes[1].set_title('Spring')
            fig.suptitle(f'Time step {time_step}')
            entries = []
            for particle_id in range(0, particle_count):
                data = {'particle': particle_id,
                        'x_cordinate': observations.iloc[time_step][f'p_{particle_id}_x_position'],
                        'y_cordinate': observations.iloc[time_step][f'p_{particle_id}_y_position']}
                entries.append(data)
            pdframe = pd.DataFrame(entries)
            pl = sns.scatterplot(data=pdframe, x='x_cordinate', y='y_cordinate', hue='particle', ax=axes[0])
            #sns.heatmap(self.edges[time_step], vmin=0.0, vmax=1.0, ax=axes[1])
            pl.set_ylim(-5.0, 5.0)
            pl.set_xlim(-5.0, 5.0)
            plt.savefig(f"{os.getcwd()}/media/timestep_{time_step}.png")
            plt.clf()
            logging.info(f"plot saved to {os.getcwd()}/media/timestep_{time_step}.png")


    @classmethod
    def create_gif(cls, loc='particle_simulation'):
        """
        Read png files from a location and compose a gif
        :param
        """
        import os
        import glob
        from PIL import Image

        fcont = len(glob.glob(f"{os.getcwd()}/media/timestep_*.png"))
        # ref: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img, *imgs = [Image.open(f"{os.getcwd()}/media/timestep_{i}.png") for i in range(1, fcont)]
        img.save(fp=f"{os.getcwd()}/media/{loc}.gif",
                 format='GIF',
                 append_images=imgs,
                 save_all=True,
                 duration=10,
                 loop=0)

        # delete all png files.
        fp_in = f"{os.getcwd()}/media/timestep_*.png"
        for f in glob.glob(fp_in):
            os.remove(f)
        logging.info('trajectory gif stores in media')

    @classmethod
    def save_pair_plot(cls, observations):
        import os
        import seaborn as sns
        sns.pairplot(observations).savefig(f"{os.getcwd()}/media/relationship.png")

    @classmethod
    def save_graph(cls, causal_graph, testing_graph, predicted_graph, step, attr=None):
        import networkx as nx

        fig, axes = plt.subplots(1, 3, figsize=(16, 8))
        axes[0].set_title('Original Graph')
        nx.draw(causal_graph,
                nx.circular_layout(causal_graph),
                with_labels=True,
                node_size=500,
                ax=axes[0])

        axes[1].set_title(f'{attr[0]} = {attr[1]}')
        nx.draw(testing_graph,
                nx.circular_layout(testing_graph),
                with_labels=True,
                node_size=500,
                ax=axes[1])

        axes[2].set_title('Predicted Graph')
        nx.draw(predicted_graph,
                nx.circular_layout(predicted_graph),
                with_labels=True,
                node_size=500,
                ax=axes[2])

        # plt.show()
        fig.savefig(os.path.join(os.getcwd(), 'tmp', f'graph_{step}.png'))
        plt.clf()
        plt.close(fig)