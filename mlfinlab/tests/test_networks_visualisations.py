"""
Tests for visualisations.py functions which help to easily deploy Dash servers.
"""

import os
import unittest

import dash
import numpy as np
import pandas as pd
from jupyter_dash import JupyterDash

from mlfinlab.networks.visualisations import generate_mst_server, create_input_matrix


class TestVisualisations(unittest.TestCase):
    """
    Tests for the different options in the visualisations.py methods.
    """

    def setUp(self):
        """
        Sets up the data input path.
        """
        # Set project path to current directory.
        project_path = os.path.dirname(__file__)
        # Add new data path to match stock_prices.csv data.
        data_path = project_path + '/test_data/stock_prices_2.csv'
        self.log_return_dataframe = pd.read_csv(data_path, index_col=False)

    def test_create_input_matrix(self):
        """
        Tests distance matrix sub type inputs.
        """
        input_matrix = create_input_matrix(self.log_return_dataframe, 'angular')
        self.check_angular_distance(input_matrix)
        # An incorrect sub type raises Value Error
        self.assertRaises(ValueError, create_input_matrix, self.log_return_dataframe, 'invalid matrix subtype')

    def check_matrix_structure(self, input_matrix):
        """
        Tests matrix structure is formed correctly.
        """
        stocks = ['Apple', 'Amazon', 'Facebook', 'Microsoft', 'Netflix', 'Tesla']
        # Check stock names are matrix indexes
        self.assertEqual(list(input_matrix.index.values), stocks)
        self.assertEqual(list(input_matrix.columns.values), stocks)
        # Make sure the diagonal matrix
        self.assertEqual(sum(np.diag(input_matrix)), 0)

    def check_angular_distance(self, input_matrix):
        """
        Tests angular distance correctly returned when 'angular' is passed as a parameter.
        """
        # Check structure of matrix is correct
        self.check_matrix_structure(input_matrix)

        # Check values of the matrix are correct
        self.assertEqual(input_matrix.iat[0, 1], 0.2574781544131463)
        self.assertEqual(input_matrix.iat[1, 2], 0.24858600121487132)
        self.assertEqual(input_matrix.iat[0, 4], 0.2966985001647295)
        self.assertEqual(input_matrix.iat[1, 4], 0.12513992168768526)
        self.assertEqual(input_matrix.iat[2, 3], 0.2708412819346416)

    def test_default_generate_mst_server(self):
        """
        Tests the default, minimal input of the method generate_mst_server.
        """
        default_server = generate_mst_server(self.log_return_dataframe)
        self.assertIsInstance(default_server, dash.Dash)
        for element in default_server.layout['cytoscape'].elements:
            if len(element) > 1:
                colour_group = element['data']['colour_group']
                self.assertEqual(colour_group, 'default')
                size = element['data']['size']
                self.assertEqual(size, 0)

    def test_jupyter_generate_mst_server(self):
        """
        Tests the Jupyter notebook option for the generator method.
        """
        jupyter_server = generate_mst_server(self.log_return_dataframe, jupyter=True)
        self.assertIsInstance(jupyter_server, JupyterDash)

    def test_colours_mst_server(self):
        """
        Tests the groups are added correctly, when they are passed using the colours parameter.
        """
        colours_input = {"tech": ['Apple', 'Amazon', 'Facebook', 'Microsoft', 'Netflix', 'Tesla']}
        server_colours = generate_mst_server(self.log_return_dataframe, colours=colours_input)
        for element in server_colours.layout['cytoscape'].elements:
            if len(element) > 1:
                colour_group = element['data']['colour_group']
                self.assertEqual(colour_group, 'tech')

    def test_sizes_mst_server(self):
        """
        Tests the sizes are added correctly when sizes are passed in as a parameter.
        """
        sizes_input = [100, 240, 60, 74, 22, 111]
        server_sizes = generate_mst_server(self.log_return_dataframe, sizes=sizes_input)
        sizes_output = []
        for element in server_sizes.layout['cytoscape'].elements:
            if len(element) > 1:
                size = element['data']['size']
                sizes_output.append(size)
        self.assertEqual(sizes_output, sizes_input)
