"""
These methods allows the user to easily deploy graph visualisations given an input file dataframe.
"""

from mlfinlab.networks.dash_graph import DashGraph
from mlfinlab.networks.graph import MST
from mlfinlab.codependence import get_distance_matrix


def generate_mst_server(log_returns_df, mst_algorithm='kruskal', distance_matrix_type='angular', jupyter=False,
                        colours=None, sizes=None):
    """
    This method returns a Dash server ready to be run.

    :param log_returns_df: (pd.Dataframe) An input dataframe of log returns
        with stock names as columns.
    :param mst_algorithm: (str) A valid MST type such as 'kruskal', 'prim', or 'boruvka'.
    :param distance_matrix_type: (str) A valid sub type of a distance matrix,
        namely 'angular', 'abs_angular', 'squared_angular'.
    :param jupyter: (bool) True if the user would like to run inside jupyter notebook. False otherwise.
    :param colours: (Dict) A dictionary of key string for category name and value of a list of indexes
        corresponding to the node indexes inputted in the initial dataframe.
    :param sizes: (List) A list of numbers, where the positions correspond to the node indexes inputted
        in the initial dataframe.
    :return: (Dash) Returns the Dash app object, which can be run using run_server.
        Returns a Jupyter Dash object if the parameter jupyter is set to True.
    """
    distance_matrix = create_input_matrix(log_returns_df, distance_matrix_type)

    # Create MST object
    graph = MST(distance_matrix, 'distance', mst_algorithm)

    # If colours are inputted, call set the node colours
    if colours:
        graph.set_node_groups(colours)

    # If sizes are inputted, set the node sizes
    if sizes:
        graph.set_node_size(sizes)

    # If Jupyter is true, create a Jupyter compatible DashGraph object
    if jupyter:
        dash_graph = DashGraph(graph, 'jupyter notebook')
    else:
        dash_graph = DashGraph(graph)

    # Retrieve the server
    server = dash_graph.get_server()

    return server


def create_input_matrix(log_returns_df, distance_matrix_type):
    """
    This method returns the distance matrix ready to be inputted into the Graph class.

    :param log_returns_df: (pd.Dataframe) An input dataframe of log returns
        with stock names as columns.
    :param distance_matrix_type: (str) A valid sub type of a distance matrix,
        namely 'angular', 'abs_angular', 'squared_angular'.
    :return: (pd.Dataframe) A dataframe of a distance matrix.
    """
    # Create correlation matrix
    correlation_matrix = log_returns_df.corr(method='pearson')

    # Valid distance matrix sub types
    valid_matrix_sub_types = ['angular', 'abs_angular', 'squared_angular']

    # If an invalid distance matrix type is used, raise an Error to notify the user
    if distance_matrix_type not in valid_matrix_sub_types:
        msg = "{} is not a valid choice distance matrix sub type. " \
              "Please select one shown in the docstring.".format(distance_matrix_type)
        raise ValueError(msg)

    # Create distance matrix
    distance_matrix = get_distance_matrix(correlation_matrix, distance_metric=distance_matrix_type)

    return distance_matrix
