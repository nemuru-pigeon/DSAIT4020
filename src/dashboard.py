from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd


class Dashboard:
    def __init__(self):
        self.app = Dash(__name__)
        self.df = pd.DataFrame([])

        self.init_sample_data()

    def init_sample_data(self):
        # Example data
        self.df = pd.DataFrame({
            "x": range(10),
            "y": [i ** 2 for i in range(10)],
            "category": ['A'] * 5 + ['B'] * 5
        })

    @staticmethod
    def create_dash_tab(title, df_graph_data, label_x, label_y, color=None, graph_type=None):
        return dcc.Tab(label=title, children=[
                dcc.Graph(figure=px.scatter(df_graph_data, x=label_x, y=label_y))
            ])

    def define_layout(self):
        # Define layout
        self.app.layout = html.Div([
            dcc.Tabs([
                Dashboard.create_dash_tab('Step 1', self.df, "x", "y"),
                Dashboard.create_dash_tab('Step 2', self.df, "x", "y"),
                # dcc.Tab(label='Step 2', children=[
                #     dcc.Graph(figure=px.bar(df, x="x", y="y", color="category"))
                # ]),
                dcc.Tab(label='Final Results', children=[
                    dcc.Graph(figure=px.line(self.df, x="x", y="y"))
                ])
            ])
        ])

