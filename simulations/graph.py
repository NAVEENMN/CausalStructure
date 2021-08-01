#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np
import networkx as nx
import  matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

class Graph(object):
    def __init__(self, nx_graph):
        self.graph = nx_graph
        self.node_size = 800
        self.node_color = '#0D0D0D'
        self.font_color = '#D9D9D9'
        self.edge_color = '#262626'

    def load_graph(self, path="data/graph.edgelist"):
        self.graph = nx.read_edgelist(path,
                                      create_using=nx.DiGraph)

    def add_node_to_graph(self, node):
        logging.debug(f'*** Graph: Adding a node {node}')
        self.graph.add_node(node, value=np.random.randn())

    def add_an_edge_to_graph(self, node_a, node_b, weight=None):
        # edge is a linear function v(node_b) = v(node_a) * _w + _c
        _w = weight if weight else random.choice([1, -1]) * np.random.normal(2, 0.5, 1)
        _c = np.random.normal(0, 1, 1)
        logging.info(f'*** Graph: Adding an edge {node_a}-{node_b}:{weight}')
        self.graph.add_edge(node_a, node_b, color=self.edge_color,
                            weight=_w.item(0),
                            capacity=_c.item(0))

    def remove_an_edge_from_graph(self, node_a, node_b):
        logging.debug(f'*** Graph: Removing an edge {node_a}-{node_b}')
        if self.graph.has_edge(node_a, node_b):
            logging.debug(f'*** Graph: Removed an edge {node_a}-{node_b}')
            self.graph.remove_edge(node_a, node_b)
            return True
        logging.debug(f'*** Graph: No edge {node_a}-{node_b} found')
        return False

    def get_graph(self):
        return self.graph

    def get_total_nodes(self):
        return self.graph.number_of_nodes()

    def get_node_values(self):
        values = nx.get_node_attributes(self.graph, 'value').values()
        return values

    def get_edge_value(self, node_a, node_b):
        return self.graph.get_edge_data(node_a, node_b)

    def get_successors(self, node):
        return list(self.graph.successors(node))

    def get_node_value(self, node):
        _nodes = self.graph.nodes()
        return _nodes[node]['value']

    def set_node_value(self, node, value):
        attrs = {node: {'value': value}}
        nx.set_node_attributes(self.graph, attrs)

    def get_random_node(self):
        node = None
        if self.graph.number_of_nodes() != 0:
            node = random.sample(self.graph.nodes(), 1)[0]
        return node

    def reset_all_nodes(self):
        # TODO:reset all node values and edge values.
        for node in self.get_nodes():
            self.set_node_value(node, np.random.randn()/1000)

    def get_source_nodes(self):
        _nodes = []
        for node in self.graph.nodes():
            if self.graph.in_degree(node) == 0:
                _nodes.append(node)
        return _nodes

    def get_nodes(self):
        return self.graph.nodes()

    def get_all_parents(self):
        _parents = []
        for node in self.graph.nodes():
            if self.graph.out_degree(node) > 0:
                _parents.append(node)
        return _parents

    @classmethod
    def draw_graph(cls, _graph, axes):
        nx.draw(_graph, nx.circular_layout(_graph),
                with_labels=True,
                node_size=500,
                ax=axes)


class ParticleGraph(Graph):
    """
    Keeps track of particles and springs as a graph system.
    :param
    """
    def __init__(self):
        super().__init__(nx.DiGraph())
        self.particle_count = 0
        self.spring_count = 0

    def __repr__(self):
        return self.get_graph()

    def get_node_names(self):
        return self.get_nodes()

    def get_total_number_of_particles(self):
        return self.particle_count

    def get_total_number_of_springs(self):
        return self.spring_count

    def add_particle_node_to_graph(self, name=None):
        _node = name if name else f'p_{self.get_total_nodes()}'
        logging.debug(f'*** ParticleGraph: Adding a node to graph {name}')
        self.add_node_to_graph(_node)
        self.particle_count += 1
        return _node

    def add_spring_to_graph(self, particle_a, particle_b, spring_constant):
        # Spring is a fork in causal graph terminology
        # X(particle) <- Y(Spring) -> Z(particle)
        x = particle_a
        z = particle_b
        y = f's{self.get_total_nodes()}'
        logging.debug(f'*** ParticleGraph: Adding a spring {particle_a}-{particle_b}:{spring_constant}')
        self.add_node_to_graph(y)
        self.spring_count += 1
        self.add_an_edge_to_graph(y, x, weight=spring_constant)
        self.add_an_edge_to_graph(y, z, weight=spring_constant)

    def remove_spring_from_graph(self, node_a, node_b):
        if self.remove_an_edge_from_graph(node_a=node_a, node_b=node_b):
            self.spring_count -= 1
            return True
        return False

    def add_springs_to_graph(self, spring_constant_matrix):
        # Since spring constant matrix is symmetric nullify lower half
        spring_constant_matrix = np.tril(spring_constant_matrix, k=0)
        for i in range(self.particle_count):
            for j in range(self.particle_count):
                if spring_constant_matrix[i][j] != 0:
                    logging.info(f'Adding spring between particle_{i} and particle_{j} with k={spring_constant_matrix[i][j]} ')
                    self.add_spring_to_graph(particle_a=f'p_{i}',
                                             particle_b=f'p_{j}',
                                             spring_constant=spring_constant_matrix[i][j])

    def show(self):
        colors = nx.get_edge_attributes(self.graph, 'color').values()
        weights = nx.get_edge_attributes(self.graph, 'weight').values()
        nx.draw(self.graph,
                pos=nx.circular_layout(self.graph),
                with_labels=True,
                edge_color=list(colors),
                width=list(weights),
                node_size=500)
        plt.show()

    def draw(self, axes):
        colors = nx.get_edge_attributes(self.graph, 'color').values()
        weights = nx.get_edge_attributes(self.graph, 'weight').values()
        nx.draw(self.graph,
                pos=nx.circular_layout(self.graph),
                with_labels=True,
                edge_color=list(colors),
                width=list(weights),
                node_size=500,
                ax=axes)