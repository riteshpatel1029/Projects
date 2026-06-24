import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Load Dataset
df = pd.read_csv("healthcare_data.csv")

# Initialize Dash App
app = Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Healthcare Data Analytics Dashboard",
            style={'textAlign': 'center'}),

    dcc.Dropdown(
        id='disease-dropdown',
        options=[{'label': i, 'value': i}
                 for i in df['Disease'].unique()],
        value='Diabetes'
    ),

    dcc.Graph(id='cost-chart'),
    dcc.Graph(id='age-chart')
])

# Callback
@app.callback(
    [Output('cost-chart', 'figure'),
     Output('age-chart', 'figure')],
    [Input('disease-dropdown', 'value')]
)
def update_graph(selected_disease):

    filtered_df = df[df['Disease'] == selected_disease]

    cost_fig = px.bar(
        filtered_df,
        x='Patient_ID',
        y='Cost',
        title=f'Patient Cost Analysis - {selected_disease}',
        color='Gender'
    )

    age_fig = px.histogram(
        filtered_df,
        x='Age',
        title=f'Age Distribution - {selected_disease}',
        nbins=10
    )

    return cost_fig, age_fig

if __name__ == '__main__':
    app.run(debug=True)