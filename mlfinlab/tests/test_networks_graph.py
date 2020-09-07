"""
Tests for Graph class in networks module
"""

import os
import unittest

import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot

from mlfinlab.networks.graph import MST


class TestGraph(unittest.TestCase):
    """
    Tests for Graph object and its functions in Networks module
    """

    def setUp(self):
        """
        Set up path to import test data
        """
        # Set project path to current directory.
        project_path = os.path.dirname(__file__)
        # Add new data path to match stock_prices.csv data.
        data_path = project_path + '/test_data/stock_prices_2.csv'
        log_return_dataframe = pd.read_csv(data_path, index_col=False)

        # Calculate correlation and distances
        correlation_matrix = log_return_dataframe.corr(method='pearson')
        self.distance_matrix = np.sqrt(2 * (1 - correlation_matrix))

        # Creates Graph class objects from correlation and distance matrices
        self.graph_distance = MST(self.distance_matrix, "distance")
        self.graph_correlation = MST(correlation_matrix, "correlation")

        self.industry = {"tech": ['Apple', 'Amazon', 'Facebook'], "utilities": ['Microsoft', 'Netflix', 'Tesla']}
        self.market_cap = [2000, 2500, 3000, 1000, 5000, 3500]

    def test_invalid_mst_algorithm(self):
        """
        Tests for invalid MST algorithm type which raises a ValueError.
        """
        self.assertRaises(ValueError, MST, self.distance_matrix, "distance", mst_algorithm="invalid algo")

    def test_matrix_to_mst(self):
        """
        Tests initialisation of NetworkX graphs when given
        distance or correlation matrices
        """
        mst_graph = self.graph_distance.get_graph()
        # Checking mst has the correct number of edges and nodes
        self.assertEqual(mst_graph.number_of_edges(), 5)
        self.assertEqual(mst_graph.number_of_nodes(), 6)

    def test_matrix_type(self):
        """
        Tests name of matrix type returns as set
        """
        self.assertEqual(self.graph_distance.get_matrix_type(), "distance")
        self.assertEqual(self.graph_correlation.get_matrix_type(), "correlation")

    def test_get_pos(self):
        """
        Tests get_pos returns a dictionary of node positions
        """
        pos = self.graph_distance.get_pos()
        self.assertEqual(len(pos), 6)
        self.assertIsInstance(pos, dict)
        nodes = ['Apple', 'Amazon', 'Facebook', 'Microsoft', 'Netflix', 'Tesla']
        for i, item in enumerate(pos):
            self.assertEqual(item, nodes[i])

    def test_get_graph(self):
        """
        Tests whether get_graph returns a nx.Graph object
        """
        self.assertIsInstance(self.graph_distance.get_graph(), nx.Graph)

    def test_get_graph_plot(self):
        """
        Tests get_graph returns axes
        """
        axes = self.graph_distance.get_graph_plot()
        self.assertIsInstance(axes, pyplot.Axes)

    def test_set_node_group(self):
        """
        Tests industry groups is set as attribute of class
        """
        self.graph_distance.set_node_groups(self.industry)
        self.assertEqual(self.graph_distance.node_groups, self.industry)
        self.get_node_colours()

    def test_set_node_size(self):
        """
        Tests node size (e.g. market cap) is set as attribute of class
        """
        self.graph_distance.set_node_size(self.market_cap)
        self.assertEqual(self.graph_distance.node_sizes, self.market_cap)
        self.get_node_sizes()

    def get_node_sizes(self):
        """
        Test for getter method of node sizes
        """
        node_sizes = self.graph_distance.get_node_sizes()
        self.assertEqual(node_sizes, self.market_cap)

    def get_node_colours(self):
        """
        Test for getter method of node colours
        """
        node_colours = self.graph_distance.get_node_colours()
        self.assertEqual(node_colours, self.industry)
