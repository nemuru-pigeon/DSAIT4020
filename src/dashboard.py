from dash import Dash, dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc


class Dashboard:
    def __init__(self, results):
        self.app = Dash('Speeddating ML Dashboard')
        self.results = results
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

    def define_layout(self, y_test, model_preds):
        tabs = [
            Dashboard.create_dash_tab('Accuracy Comparison', [self.create_accuracy_bar_chart()])
        ]

        confusion_matrices = []
        roc_curves = []

        for res in self.results:
            model_name = res['Model']
            y_pred = model_preds[model_name]['y_pred']
            y_scores = model_preds[model_name]['y_scores']

            confusion_matrices.append(self.create_confusion_matrix(model_name, y_test, y_pred))
            roc_curves.append(self.create_roc_curve(model_name, y_test, y_scores))

        tabs.append(Dashboard.create_dash_tab('Confusion Matrices', confusion_matrices))
        tabs.append(Dashboard.create_dash_tab('ROC Curves', roc_curves))

        self.app.layout = html.Div([
            dcc.Tabs(tabs)
        ])

    def run(self):
        self.app.run_server(debug=True)