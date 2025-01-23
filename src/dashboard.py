from dash import Dash, dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np


class Dashboard:
    def __init__(self, results, learning_curves, validation_curves):
        self.app = Dash('Speeddating ML Dashboard')
        self.results = results
        self.learning_curves = learning_curves
        self.validation_curves = validation_curves
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
    def create_dash_tab(title, figures):
        return dcc.Tab(label=title, children=[
            html.Div([dcc.Graph(figure=fig) for fig in figures])
        ])

    def create_accuracy_bar_chart(self):
        df_acc = pd.DataFrame({
            "Model": [res['Model'] for res in self.results],
            "Accuracy": [res['Accuracy'] for res in self.results]
        })
        fig = px.bar(df_acc, x='Model', y='Accuracy', title='Model Accuracy Comparison')
        return fig

    def create_confusion_matrix(self, model_name, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        fig = go.Figure(data=go.Heatmap(z=cm, x=['Pred 0', 'Pred 1'], y=['Actual 0', 'Actual 1'], colorscale='Blues'))
        fig.update_layout(title=f'Confusion Matrix - {model_name}')
        return fig

    def create_roc_curve(self, model_name, y_test, y_scores):
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.2f}'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))
        fig.update_layout(title=f'ROC Curve - {model_name}', xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate')
        return fig

    def create_learning_curve(self, model_name, learning_curve_data):
        train_sizes = learning_curve_data['train_sizes']
        train_scores = np.mean(learning_curve_data['train_scores'], axis=1)
        test_scores = np.mean(learning_curve_data['test_scores'], axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_sizes, y=train_scores, mode='lines', name='Training Score'))
        fig.add_trace(go.Scatter(x=train_sizes, y=test_scores, mode='lines', name='Validation Score'))
        fig.update_layout(title=f'Learning Curve - {model_name}', xaxis_title='Training Size', yaxis_title='Score')
        return fig

    def create_validation_curve(self, model_name, validation_curve_data):
        param_range = validation_curve_data['param_range']
        train_scores = np.mean(validation_curve_data['train_scores'], axis=1)
        test_scores = np.mean(validation_curve_data['test_scores'], axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=param_range, y=train_scores, mode='lines', name='Training Score'))
        fig.add_trace(go.Scatter(x=param_range, y=test_scores, mode='lines', name='Validation Score'))
        fig.update_layout(title=f'Validation Curve - {model_name}', xaxis_title='Parameter', yaxis_title='Score')
        return fig

    def define_layout(self, y_test, model_preds):
        tabs = [
            Dashboard.create_dash_tab('Accuracy Comparison', [self.create_accuracy_bar_chart()])
        ]

        confusion_matrices = []
        roc_curves = []
        learning_curve_figures = []
        validation_curve_figures = []

        for res in self.results:
            model_name = res['Model']
            y_pred = model_preds[model_name]['y_pred']
            y_scores = model_preds[model_name]['y_scores']

            confusion_matrices.append(self.create_confusion_matrix(model_name, y_test, y_pred))
            roc_curves.append(self.create_roc_curve(model_name, y_test, y_scores))
            learning_curve_figures.append(self.create_learning_curve(model_name, self.learning_curves[model_name]))
            validation_curve_figures.append(self.create_validation_curve(model_name, self.validation_curves[model_name]))

        tabs.append(Dashboard.create_dash_tab('Confusion Matrices', confusion_matrices))
        tabs.append(Dashboard.create_dash_tab('ROC Curves', roc_curves))
        tabs.append(Dashboard.create_dash_tab('Learning Curves', learning_curve_figures))
        tabs.append(Dashboard.create_dash_tab('Validation Curves', validation_curve_figures))

        self.app.layout = html.Div([
            dcc.Tabs(tabs)
        ])

    def run(self):
        self.app.run_server(debug=True)