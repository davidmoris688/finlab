"""
This class takes in a Graph object and creates interactive visualisations using Plotly's Dash.
The DashGraph class contains private functions used to generate the frontend components needed to create the UI.

Running run_server() will produce the warning "Warning: This is a development server. Do not use app.run_server
in production, use a production WSGI server like gunicorn instead.".
However, this is okay and the Dash server will run without a problem.
"""

import json
import random

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_cytoscape as cyto
import dash_html_components as html
from dash import Dash
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
from networkx import nx


class DashGraph:
    """
    This DashGraph class creates a server for Dash cytoscape visualisations.
    """

    def __init__(self, input_graph, app_display='default'):
        """
        Initialises the DashGraph object from the Graph class object.
        Dash creates a mini Flask server to visualise the graphs.

        :param app_display: (str) 'default' by default and 'jupyter notebook' for running Dash inside Jupyter Notebook.
        :param input_graph: (Graph) Graph class from graph.py.
        """
        # Dash app styling with Bootstrap
        if app_display == 'jupyter notebook':
            self.app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        else:
            self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        # Graph class object
        self.graph = input_graph
        # The dictionary of the nodes coordinates
        self.pos = self.graph.get_pos()

        # Colours of nodes
        self.colour_groups = {}
        # If colours have been assigned in Graph class, add styling
        if self.graph.get_node_colours():
            colour_map = self.graph.get_node_colours()
            self._assign_colours_to_groups(list(colour_map.keys()))

        # Create elements for Dash graph
        self.weights = []
        self.elements = []
        self._update_elements()

        # Load the different graph layouts
        cyto.load_extra_layouts()

        # Load default stylesheet
        self.stylesheet = self._get_default_stylesheet()

        # Append stylesheet for colour and size
        # If sizes have been set in the Graph class
        self._style_colours()
        if self.graph.get_node_sizes():
            self._assign_sizes()

        # Callback functions to hook frontend elements to functions
        self.app.callback(Output('cytoscape', 'layout'),
                          [Input('dropdown-layout', 'value')])(DashGraph._update_cytoscape_layout)
        self.app.callback(Output('json-output', 'children'),
                          [Input('dropdown-stat', 'value')])(self._update_stat_json)
        self.app.callback(Output('cytoscape', 'elements'),
                          [Input('rounding_decimals', 'value')])(self._round_decimals)

    def _generate_cyto_graph(self):
        """
        Generates the cytoscape graph element.

        :return: (Cytoscape) Returns the Cytoscpae graph object.
        """
        cyto_graph = cyto.Cytoscape(
            id='cytoscape',
            layout={
                'name': 'cose-bilkent'
            },
            style={
                'width': '100%',
                'height': '600px',
                'padding': '5px 3px 5px 3px'
            },
            elements=self.elements,
            stylesheet=self.stylesheet
        )
        return cyto_graph

    def _generate_layout(self):
        """
        Generates the layout for cytoscape.

        :return: (dbc.Container) Returns Dash Bootstrap Component Container containing the layout of UI.
        """

        layout_input = [
            html.H1("Minimum Spanning Tree from {} matrix".format(self.graph.get_matrix_type())),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(DashGraph._get_default_controls(), md=4),
                    dbc.Col(self._generate_cyto_graph(), md=8),
                ],
                align="center",
            )
        ]
        if self.colour_groups:
            layout_input.append(self._get_toast())

        layout = dbc.Container(
            layout_input,
            fluid=True,
        )
        return layout

    def _assign_colours_to_groups(self, groups):
        """
        Assigns the colours to industry or sector groups by creating a dictionary of group name to colour.

        :param groups: (List) List of industry groups as strings.
        """
        # List of colours selected to match with industry groups
        colours = ["#d0b7d5", "#a0b3dc", "#90e190", "#9bd8de",
                   "#eaa2a2", "#f6c384", "#dad4a2", '#ff52a8',
                   '#ffd1e8', '#bd66ff', '#6666ff', '#66ffff',
                   '#00e600', '#fff957', '#ffc966', '#ff8833',
                   '#ff6666', '#C0C0C0', '#008080']

        # Random colours are generated if industry groups added exceeds 19
        while len(groups) > len(colours):
            random_number = random.randint(0, 16777215)
            hex_number = str(hex(random_number))
            hex_number = '#' + hex_number[2:]
            colours.append(hex_number)

        # Create and add to the colour map
        colour_map = {}
        for i, item in enumerate(groups):
            colour_map[item] = colours[i].capitalize()
        self.colour_groups = colour_map

    def _style_colours(self):
        """
        Appends the colour styling to stylesheet for the different groups.
        """
        if self.colour_groups:
            keys = list(self.colour_groups.keys())
            for item in keys:
                new_colour = {
                    "selector": "node[colour_group=\"{}\"]".format(item),
                    "style": {
                        'background-color': '{}'.format(self.colour_groups[item]),
                    }
                }
                self.stylesheet.append(new_colour)

    def _get_node_group(self, node_name):
        """
        Returns the industry or sector name for a given node name.

        :param node_name: (str) Name of a given node in the graph.
        :return: (str) Name of industry that the node is in or "default" for nodes which haven't been assigned a group.
        """
        node_colour_map = self.graph.get_node_colours()
        for key, val in node_colour_map.items():
            if node_name in val:
                return key
        return "default"

    def _get_node_size(self, index):
        """
        Returns the node size for given node index if the node sizes have been set.

        :return: (float) Returns size of node set, 0 if it has not been set.
        """
        if self.graph.get_node_sizes():
            return self.graph.get_node_sizes()[index]
        return 0

    def _assign_sizes(self):
        """
        Assigns the node sizing by appending to the stylesheet.
        """
        sizes = self.graph.get_node_sizes()
        max_size = max(sizes)
        min_size = min(sizes)
        new_sizes = {
            'selector': 'node',
            'style': {
                "width": "mapData(size, {min}, {max}, 25, 250)".format(min=min_size, max=max_size),
                "height": "mapData(size, {min}, {max}, 25, 250)".format(min=min_size, max=max_size),
            }
        }
        self.stylesheet.append(new_sizes)

    def _update_elements(self, dps=4):
        """
        Updates the elements needed for the Dash Cytoscape Graph object.

        :param dps: (int) Decimal places to round the edge values.
        """
        self.elements = []
        self.weights = []
        i = 0
        for node in self.pos:
            self.elements.append({
                'data': {'id': node, 'label': node, 'colour_group': self._get_node_group(node),
                         'size': self._get_node_size(i)},
                'position': {
                    'x': self.pos[node][0] * 256,
                    'y': self.pos[node][1] * 256
                }
            })
            i += 1

        for node1, node2, weight in self.graph.get_graph().edges(data=True):
            self.weights.append(round(weight['weight'], dps))
            self.elements.append({'data': {'source': node1, 'target': node2, 'weight': round(weight['weight'], dps)}})

    def get_server(self):
        """
        Starts up a small Flask server.

        :return: (Dash) Returns the Dash app object, which can be run using run_server.
            Returns a Jupyter Dash object if DashGraph has been initialised for Jupyter Notebook.
        """
        self.app.layout = self._generate_layout()
        return self.app

    @staticmethod
    def _update_cytoscape_layout(layout):
        """
        Callback function for updating the cytoscape layout.
        The useful layouts for MST have been included as options (cola, cose-bilkent, spread).

        :return: (Dict) Dictionary of the key 'name' to the desired layout (e.g. cola, spread).
        """
        return {'name': layout}

    def _update_stat_json(self, stat_name):
        """
        Callback function for updating the statistic shown.

        :param stat_name: (str) Name of the statistic to display (e.g. graph_summary).
        :return: (json) Json of the graph information depending on chosen statistic.
        """
        switcher = {
            "graph_summary": self.get_graph_summary(),
            "average_degree_connectivity": nx.average_degree_connectivity(self.graph.get_graph()),
            "average_neighbor_degree": nx.average_neighbor_degree(self.graph.get_graph()),
            "betweenness_centrality": nx.betweenness_centrality(self.graph.get_graph()),
        }
        return json.dumps(switcher.get(stat_name), indent=2)

    def get_graph_summary(self):
        """
        Returns the Graph Summary statistics.
        The following statistics are included - the number of nodes and edges, smallest and largest edge,
        average node connectivity, normalised tree length and the average shortest path.

        :return: (Dict) Dictionary of graph summary statistics.
        """
        summary = {
            "nodes": len(self.pos),
            "edges": self.graph.get_graph().number_of_edges(),
            "smallest_edge": min(self.weights),
            "largest_edge": max(self.weights),
            "average_node_connectivity": nx.average_node_connectivity(self.graph.get_graph()),
            "normalised_tree_length": (sum(self.weights)/(len(self.weights))),
            "average_shortest_path": nx.average_shortest_path_length(self.graph.get_graph())
        }
        return summary

    def _round_decimals(self, dps):
        """
        Callback function for updating decimal places.
        Updates the elements to modify the rounding of edge values.

        :param dps: (int) Number of decimals places to round to.
        :return: (List) Returns the list of elements used to define graph.
        """
        if dps:
            self._update_elements(dps)
        return self.elements

    def _get_default_stylesheet(self):
        """
        Returns the default stylesheet for initialisation.

        :return: (List) A List of definitions used for Dash styling.
        """
        stylesheet = \
            [
                {
                    'selector': 'node',
                    'style': {
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'background-color': '#00BFFF',
                        'color': '',
                        'font-family': 'sans-serif',
                        'font-size': '12',
                        'font-weight': 'bold',
                        'border-width': 1.5,
                        'border-color': '#161615',
                    }
                },
                {
                    "selector": 'edge',
                    "style": {
                        'label': 'data(weight)',
                        "line-color": "#BDB76B",
                        'font-size': '8',
                    }
                },
                {
                    "selector": '[weight => 0]',
                    "style": {
                        "line-color": "#ADD8E6",
                        "width": "mapData(weight, 0, {}, 1, 8)".format(max(self.weights)),
                    }
                },
                {
                    "selector": '[weight < 0]',
                    "style": {
                        'line-color': '#FAEBD7',
                        "width": "mapData(weight, 0, {}, 1, 8)".format(min(self.weights)),
                    }
                }
            ]
        return stylesheet

    def _get_toast(self):
        """
        Toast is the floating colour legend to display when industry groups have been added.
        This method returns the toast component with the styled colour legend.

        :return: (html.Div) Returns Div containing colour legend.
        """
        list_elements = []
        for industry, colour in self.colour_groups.items():
            span_styling = \
                {
                    "border": "1px solid #ccc",
                    "background-color": colour,
                    "float": "left",
                    "width": "12px",
                    "height": "12px",
                    "margin-right": "5px"
                }
            children = [industry.title(), html.Span(style=span_styling)]
            list_elements.append(html.Li(children))

        toast = html.Div(
            [
                dbc.Toast(
                    html.Ul(list_elements, style={"list-style": "None", "padding-left": 0}),
                    id="positioned-toast",
                    header="Industry Groups",
                    dismissable=True,
                    # stuck on bottom right corner
                    style={"position": "fixed", "bottom": 36, "right": 10, "width": 350},
                ),
            ]
        )
        return toast

    @staticmethod
    def _get_default_controls():
        """
        Returns the default controls for initialisation.

        :return: (dbc.Card) Dash Bootstrap Component Card which defines the side panel.
        """
        controls = dbc.Card(
            [
                html.Div([
                    dbc.FormGroup(
                        [
                            dbc.Label("Graph Layout"),
                            dcc.Dropdown(
                                id="dropdown-layout",
                                options=[
                                    {"label": col, "value": col} for col in ['cose-bilkent', 'cola', 'spread']
                                ],
                                value="cose-bilkent",
                                clearable=False,
                            ),
                        ]
                    ),
                    dbc.FormGroup(
                        [
                            dbc.Label("Statistic Type"),
                            dcc.Dropdown(
                                id="dropdown-stat",
                                options=[
                                    {"label": col, "value": col} for col in
                                    ['graph_summary',
                                     'average_degree_connectivity', 'average_neighbor_degree',
                                     'betweenness_centrality'
                                     ]
                                ],
                                value="graph_summary",
                                clearable=False,
                            ),
                        ]
                    ),
                    html.Pre(
                        id='json-output',
                        style={'overflow-y': 'scroll',
                               'height': '100px',
                               'border': 'thin lightgrey solid'}
                    ),
                    dbc.FormGroup(
                        [
                            dbc.Label("Decimal Places"),
                            dbc.Input(id="rounding_decimals", type="number", value=4, min=1),
                        ]
                    ),
                ]),
                dbc.CardBody(html.Div(id="card-content", className="card-text")),
            ],
            body=True,
        )
        return controls
