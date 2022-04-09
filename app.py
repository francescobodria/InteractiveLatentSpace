#!/usr/bin/env python
# coding: utf-8

import os
import re
import numpy as np
import pandas as pd

from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware

# Data Loading

# https://www.kaggle.com/c/titanic/data
tit_sub = pd.read_csv('./data/Titanic/gender_submission.csv')
tit_train = pd.read_csv('./data/Titanic/train.csv')
tit_test = pd.read_csv('./data/Titanic/test.csv')
df_train_final = pd.read_pickle("./data/Titanic/df_train_final")
df_test_final = pd.read_pickle("./data/Titanic/df_test_final")

from sklearn.preprocessing import StandardScaler

scaler_cols = ['Age', 'Fare', 'Name_Length', 'Family_Size', 'Ticket_Frequency', 'Fare_Family_Size', 'Fare_Cat_Pclass']
std = StandardScaler()
std.fit(df_train_final[scaler_cols])
df_train_final.loc[:, scaler_cols] = std.transform(df_train_final[scaler_cols])
df_test_final.loc[:, scaler_cols] = std.transform(df_test_final[scaler_cols])

features = ['Pclass', 'Sex', 'Age', 'Fare', 'Title', 'Name_Length', 'Emb_C',
            'Emb_Q', 'Emb_S', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs',
            'Title_Other', 'Title_Royal', 'Family_Size',
            'Family_Friends_Surv_Rate', 'Cabin_Clean',
            'Ticket_Frequency', 'Tkt_AS', 'Tkt_C', 'Tkt_CA',
            'Tkt_CASOTON', 'Tkt_FC', 'Tkt_FCC', 'Tkt_Fa', 'Tkt_LINE',
            'Tkt_NUM', 'Tkt_PC', 'Tkt_PP', 'Tkt_PPP', 'Tkt_SC', 'Tkt_SCA',
            'Tkt_SCAH', 'Tkt_SCAHBasle', 'Tkt_SCOW', 'Tkt_SCPARIS', 'Tkt_SCParis',
            'Tkt_SOC', 'Tkt_SOP', 'Tkt_SOPP', 'Tkt_SOTONO', 'Tkt_SOTONOQ', 'Tkt_SP',
            'Tkt_STONO', 'Tkt_SWPP', 'Tkt_WC', 'Tkt_WEP', 'Fare_Cat', 'Child', 'Senior']

features_train = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Title', 'Name_Length', 'Emb_C',
                  'Emb_Q', 'Emb_S', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs',
                  'Title_Other', 'Title_Royal', 'Family_Size',
                  'Family_Friends_Surv_Rate', 'Cabin_Clean',
                  'Ticket_Frequency', 'Tkt_AS', 'Tkt_C', 'Tkt_CA',
                  'Tkt_CASOTON', 'Tkt_FC', 'Tkt_FCC', 'Tkt_Fa', 'Tkt_LINE',
                  'Tkt_NUM', 'Tkt_PC', 'Tkt_PP', 'Tkt_PPP', 'Tkt_SC', 'Tkt_SCA',
                  'Tkt_SCAH', 'Tkt_SCAHBasle', 'Tkt_SCOW', 'Tkt_SCPARIS', 'Tkt_SCParis',
                  'Tkt_SOC', 'Tkt_SOP', 'Tkt_SOPP', 'Tkt_SOTONO', 'Tkt_SOTONOQ', 'Tkt_SP',
                  'Tkt_STONO', 'Tkt_SWPP', 'Tkt_WC', 'Tkt_WEP', 'Fare_Cat', 'Child', 'Senior']

df_train_final = df_train_final[features_train]
df_test_final = df_test_final[features]

# train/test/val split
features = df_test_final.columns.to_list()
X_train_df = df_train_final[features]
Y_train = df_train_final['Survived']
X_test_df = df_test_final.reset_index(drop=True)

c = pd.read_csv('./data/Titanic/titanic_test_labels.csv')
test_data_with_labels = c.copy()
for i, name in enumerate(test_data_with_labels['name']):
    if '"' in name:
        # test_data_with_labels['name'][i] = re.sub('"', '', name)
        test_data_with_labels.loc[i, 'name'] = re.sub('"', '', name)
for i, name in enumerate(tit_test['Name']):
    if '"' in name:
        # tit_test['Name'][i] = re.sub('"', '', name)
        tit_test.loc[i, 'Name'] = re.sub('"', '', name)
survived = []
for name in tit_test['Name']:
    survived.append(int(test_data_with_labels.loc[test_data_with_labels['name'] == name]['survived'].values[-1]))
Y_test = pd.Series(survived, index=X_test_df.index)

Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

# Black Box Loading

import pickle

bst = pickle.load(open('./models/XGBoost_Titanic.p', 'rb'))

import xgboost as xgb

dtrain = xgb.DMatrix(X_train_df.values, Y_train)
dtest = xgb.DMatrix(X_test_df.values)
y_train_pred = bst.predict(dtrain)
y_test_pred = bst.predict(dtest)

# Concatenate the label to the data in order to work with the Conditional VAE

X_train = np.hstack((X_train_df.values, y_train_pred.reshape(-1, 1)))
X_test = np.hstack((X_test_df.values, y_test_pred.reshape(-1, 1)))

# Autoencoder Loading

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNN_CVAE(nn.Module):
    def __init__(self, input_shape, hidden, latent_dim=2):
        super(FFNN_CVAE, self).__init__()

        # encoding components
        self.fc1 = nn.Linear(input_shape, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(hidden[1], latent_dim)
        self.fc3_logvar = nn.Linear(hidden[1], latent_dim)

        # Sampling vector
        self.fc4 = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden[1]),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.fc5 = nn.Sequential(
            nn.Linear(hidden[1], hidden[0]),
            nn.ReLU(inplace=True)
        )

        self.fc6 = nn.Linear(hidden[0], input_shape)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, y):
        x = torch.cat((z, y), dim=1)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


# hyperparameters
latent_dim = 2  # latent dim extracted
hidden = [26, 13]  # hidden layers

# Create autoencoder
FFNN_CVAE_model = FFNN_CVAE(X_train.shape[1], hidden=hidden, latent_dim=latent_dim)

# Load the weights
FFNN_CVAE_model.load_state_dict(torch.load('./models/CVAE_Titanic.pt'))

# Compute the latent space
with torch.no_grad():
    FFNN_CVAE_model.eval()
    z_train, mu, logvar = FFNN_CVAE_model(torch.tensor(X_train, dtype=torch.float32))
    z_test, mu, logvar = FFNN_CVAE_model(torch.tensor(X_test, dtype=torch.float32))

# Compute the max and the mins for the plots
x_min, x_max = z_test[:, 0].min(), z_test[:, 0].max()
y_min, y_max = z_test[:, 1].min(), z_test[:, 1].max()

############ DASH APP ###############
import plotly.graph_objects as go
import plotly.express as px

# create the figure needed:
shap_fig = go.Figure()  # main plot
clustering_fig = go.Figure()  # for the clustering select
violin_plot = go.Figure()  # violin plot for the distribution selected

colorscale = px.colors.diverging.RdBu[::-1]
# [[0, '#3b4cc0'],[1, '#b40426']]

# Add the points tot he main plot
shap_fig.add_trace(go.Scatter(x=z_train[:, 0].numpy(),
                              y=z_train[:, 1].numpy(),
                              mode='markers',
                              marker=dict(
                                  size=10,
                                  symbol='circle',
                                  color=y_train_pred,
                                  opacity=0.3,
                                  cmid=0.5,
                                  colorscale=colorscale,
                                  colorbar=dict(
                                      title="% Survival")
                              )))

# Add the black cross and the expected value
shap_fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', name='pointer', showlegend=False,
                              marker=dict(
                                  size=30,
                                  symbol='circle-cross',
                                  color='rgba(255,255,255,0.3)',
                                  line=dict(width=1.5,
                                            color='black')
                              )))



# Create the annotation vectors with 0 that will be updated in the code later
names = ['vector_0', 'vector_1', 'vector_2', 'vector_3', 'vector_4', 'vector_5', 'vector_6', 'vector_7', 'vector_8',
         'vector_9']

annotations = [
                  go.layout.Annotation(
                      # start
                      x=0,
                      y=0,
                      xref="x",
                      yref="y",
                      # end
                      ax=0,
                      ay=0,
                      axref="x",
                      ayref="y",
                      showarrow=True,
                      arrowside='start',
                      arrowhead=2,  # type od head [0,8]
                      arrowsize=1,  # head dimension
                      arrowwidth=1.5,  # arrow dimension
                      arrowcolor="#000000",
                      name=name,
                      text='  ',
                      hovertext='baseline',
                      font=dict(
                          family="Courier New, monospace",
                          size=1,
                          color="#ffffff"
                      ),
                      bgcolor=None
                  ) for name in names] + [
                  go.layout.Annotation(
                      # start
                      x=0,
                      y=0,
                      xref="x",
                      yref="y",
                      # end
                      ax=0,
                      ay=0,
                      axref="x",
                      ayref="y",
                      showarrow=True,
                      arrowside='start',
                      arrowhead=2,  # type od head [0,8]
                      arrowsize=1,  # head dimension
                      arrowwidth=2,  # arrow dimension
                      arrowcolor="#000000",
                      name='others_contrib',
                      text='  ',
                      hovertext='other contributions',
                      font=dict(
                          family="Courier New, monospace",
                          size=1,
                          color="white"
                      ),
                      bgcolor='#ff7f0e'
                  )
              ]

# setting for the main fig
shap_fig.update_layout(
    showlegend=False,
    annotations=annotations,
    autosize=True,
    template='plotly_white',
    margin=dict(
        l=20,
        r=20,
        b=20,
        t=20,
        pad=0
    ))

shap_fig.update_xaxes(
    range=[x_min, x_max],
    scaleanchor="y",
    scaleratio=1,
    # zeroline=True, zerolinewidth=0.5, zerolinecolor='black'
)

shap_fig.update_yaxes(
    range=[y_min, y_max],
    # zeroline=True, zerolinewidth=0.5, zerolinecolor='black'
);

# add the points to the clustering plots
clustering_fig.add_trace(go.Scatter(x=z_train[:, 0].numpy(),
                                    y=z_train[:, 1].numpy(),
                                    mode='markers',
                                    marker=dict(
                                        size=10,
                                        symbol='circle',
                                        color=y_train_pred,
                                        cmid=0.5,
                                        opacity=0.5,
                                        colorscale=colorscale,
                                        colorbar=dict(title="% Survival")
                                    )))

# setting for the clustering fig
clustering_fig.update_layout(
    dragmode='lasso',
    showlegend=False,
    autosize=True,
    template='plotly_white',
    margin=dict(
        l=10,
        r=10,
        b=10,
        t=10,
        pad=0
    ))

clustering_fig.update_xaxes(
    range=[x_min, x_max],
)

clustering_fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
)

# add the data to the violin plot
violin_plot.add_trace(go.Violin(name='left',
                                side='negative',
                                line_color='purple',
                                line_width=0.5,
                                bandwidth=0.25,
                                points=False)
                      )
violin_plot.add_trace(go.Violin(name='right',
                                side='positive',
                                line_color='indigo',
                                line_width=0.5,
                                bandwidth=0.25,
                                points=False)
                      )
violin_plot.update_traces(meanline_visible=True)

violin_plot.update_layout(violingap=0,
                          violinmode='overlay',
                          template='plotly_white',
                          margin=dict(
                              l=20,
                              r=20,
                              t=20,
                              b=20,
                              pad=20,
                          ), legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
                          )

# Set the mini plot near the sliders

slider_names = ['slider_name_0', 'slider_name_1', 'slider_name_2', 'slider_name_3', 'slider_name_4',
                'slider_name_5', 'slider_name_6', 'slider_name_7', 'slider_name_8', 'slider_name_9']

slider_figs = [go.Figure() for name in slider_names]

slider_annotations = [[
    go.layout.Annotation(
        # start
        x=0,
        y=0,
        xref="x",
        yref="y",
        # end
        ax=1,
        ay=1,
        axref="x",
        ayref="y",
        showarrow=True,
        arrowside='start',
        arrowhead=2,  # type od head [0,8]
        arrowsize=1,  # head dimension
        arrowwidth=2,  # arrow dimension
        arrowcolor="#b07495",
        name=name + '_top',
    ),
    go.layout.Annotation(
        # start
        x=0,
        y=0,
        xref="x",
        yref="y",
        # end
        ax=-1,
        ay=-1,
        axref="x",
        ayref="y",
        showarrow=True,
        arrowside='start',
        arrowhead=2,  # type od head [0,8]
        arrowsize=1,  # head dimension
        arrowwidth=2,  # arrow dimension
        arrowcolor="#59872a",
        name=name + '_bottom',
    )
] for name in slider_names]

for i in range(len(slider_names)):
    slider_figs[i].update_layout(
        height=50,
        width=50,
        annotations=slider_annotations[i],
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=0  # top margin
        ),
        plot_bgcolor='#f2f2f2'
    )

    slider_figs[i].update_xaxes(
        showgrid=False,
        zeroline=False,
        visible=False,
    )

    slider_figs[i].update_yaxes(
        showgrid=False,
        zeroline=False,
        visible=False,
    )

    slider_figs[i].update_xaxes(
        range=[-1, 1],
    )

    slider_figs[i].update_yaxes(
        range=[-1, 1],
        scaleanchor="x",
        scaleratio=1
    )



# SHAP values computation part
import shap


def wrapper(X):
    FFNN_CVAE_model.eval()
    z, mu, logvar = FFNN_CVAE_model(torch.tensor(X).float())
    return z.detach().numpy()


# use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(wrapper, shap.sample(X_train, 100), link="identity")

#    data : numpy.array or pandas.DataFrame or shap.common.DenseData or any scipy.sparse matrix
#        The background dataset to use for integrating out features. To determine the impact
#        of a feature, that feature is set to "missing" and the change in the model output
#        is observed. Since most models aren't designed to handle arbitrary missing data at test
#        time, we simulate "missing" by replacing the feature with the values it takes in the
#        background dataset. So if the background dataset is a simple sample of all zeros, then
#        we would approximate a feature being missing by setting it to zero. For small problems
#        this background dataset can be the whole training set, but for larger problems consider
#        using a single reference value or using the kmeans function to summarize the dataset.
#        Note: for sparse case we accept any sparse matrix but convert to lil format for
#        performance.
#    link : "identity" or "logit"
#        A generalized linear model link to connect the feature importance values to the model
#        output. Since the feature importance values, phi, sum up to the model output, it often makes
#        sense to connect them to the output with a link function where link(output) = sum(phi).
#        If the model output is a probability then the LogitLink link function makes the feature
#        importance values have log-odds units.

# Round the mins to two value for diplay
mins = np.min(X_train, axis=0)
mins = np.round(mins, 2)
maxs = np.max(X_train, axis=0)
maxs = np.round(maxs, 2)

# index of the selected point
selected_point = 0
# query point to use for analysis
query = X_train[selected_point, :]
# array of the expected value of the features
exp_input_values = X_train.mean(axis=0)


# main function to output the xai scores
def compute_XAI_values(query):
    shap_values = explainer.shap_values(query, nsamples='auto')
    return explainer.expected_value[0], shap_values[0], explainer.expected_value[1], shap_values[1]


# compute expected values and shap values
exp_x, xai_x, exp_y, xai_y = compute_XAI_values(query)

# update the trace of violet triangle with the real values
shap_fig.for_each_trace(
    lambda trace: trace.update(x=[exp_x], y=[exp_y]) if trace.name == "expected_value" else (),
)

s = [exp_x, exp_y]

# select only the 10 most important features
indices = np.argsort(np.sqrt(xai_x[:-1] ** 2 + xai_y[:-1] ** 2))[-10:][::-1]

# update the annotations to the computed shap values
for i in range(len(slider_names)):
    vec = np.array([xai_x[indices][i], xai_y[indices][i]])
    vec /= np.linalg.norm(vec)
    if query[indices][i] > exp_input_values[indices][i]:
        slider_annotations[i][0]['ax'] = vec[0]
        slider_annotations[i][0]['ay'] = vec[1]
        slider_annotations[i][1]['ax'] = -vec[0]
        slider_annotations[i][1]['ay'] = -vec[1]
    else:
        slider_annotations[i][0]['ax'] = -vec[0]
        slider_annotations[i][0]['ay'] = -vec[1]
        slider_annotations[i][1]['ax'] = vec[0]
        slider_annotations[i][1]['ay'] = vec[1]
    slider_figs[i].update_layout(annotations=slider_annotations[i])

# list of the increments of the sliders
steps = [1, 1, 0.01, 0.01, 1, 0.01, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.01, 0.01, 1, 0.01, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1]

# list of the column names
columns = list(X_train_df.columns) + ['bb_proba']

################# HTML #############
import json
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


# function to format the ticks of the sliders
def select_ticks(m, M, e):
    # round values
    if m % 1:
        m = np.round(m, 2)
    else:
        m = int(m)
    if M % 1:
        M = np.round(M, 2)
    else:
        M = int(M)
    if e % 1:
        e = np.round(e, 2)
        if np.abs(e) < 0.01:
            e = 0
    else:
        e = int(e)

    ticks = dict(sorted({m: {'label': str(m), 'style': {'color': '#59872a', 'font-size': '1rem'}},
                         e: {'label': 'e', 'style': {'color': '#800081', 'font-size': '1rem'}},
                         M: {'label': str(M), 'style': {'color': '#b07495', 'font-size': '1rem'}}
                         }.items()))
    return ticks


app = dash.Dash(__name__, url_base_pathname='/LSE/', meta_tags=[{"name": "viewport", "content": "width=device-width"}],
                external_stylesheets=[dbc.themes.BOOTSTRAP])

jumbotron = html.Div(
    [
        dbc.Row(
            [
                dbc.Col([
                    html.H2("Explaining Black Box with visual exploration of Latent Space"),
                    # html.H3("Qui ci starebbe proprio bene un sottotitolo"),
                    html.Hr(className="mt-2 mb-4"),
                    html.P(
                        '''Autoencoders are a powerful yet opaque feature reduction technique, on top of which we propose a novel way for the joint visual exploration of both latent and real space.
By interactively exploiting the mapping between latent and real features, it is possible to unveil the meaning of latent features while providing deeper insight into the original variables.
To achieve this goal, we exploit and re-adapt existing approaches from eXplainable Artificial Intelligence (XAI) to understand the relationships between the input and latent features.
The uncovered relationships between input features and latent ones allow the user to understand the data structure concerning external variables such as the predictions of a classification model.
We developed an interactive framework that visually explores the latent space and allows the user to understand the relationships of the input features with model prediction.''',
                        className='two-columns lead')], md=12),
            ],
            className="p-3"
        )
    ])

header = dbc.Container(
    [
        # dbc.Row(
        #     dbc.Col(html.Div(
        #         [html.P(
        #             'Authors: Francesco Bodria, Fosca Giannotti, Salvatore Rinzivillo, Riccardo Guidotti, Daniele Fadda, Dino Pedreschi.'),
        #             html.Hr()]
        #     ))
        # ),
        dbc.Row(
            dbc.Col(
                jumbotron,
                md=12,
                className="my-3  py-2 px-3 bg-light border rounded-3"
            ),
            className="align-items-md-stretch",
        )
    ]
)

neighborhood_title = dbc.Container(
    dbc.Row(
        dbc.Col(
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col([
                                html.H3("Neighborhood analysis"),
                                # html.H3("Qui ci starebbe proprio bene un sottotitolo"),
                                html.Hr(className="mb-4")
                            ], md=12),
                            dbc.Col([
                                html.P(
                                    [
                                        html.Span('On the Left - ', style={'font-style': 'italic'}),
                                        'Sliders of the 10 most important features for the '
                                        'selected point. Values ranges from minimum to maximum through the expected '
                                        'value. The expected value ',
                                        html.Span('e ', style={'font-style': 'italic', 'font-weight': '600'}),
                                        'is the value for which the magnitude of the '
                                        'vector for that specific feature is zero. Moving above this value, '
                                        'the dimension of the vector follow the pink direction, if the value is '
                                        'below the green one. '
                                    ]
                                )], md=6),
                            dbc.Col([
                                html.P(
                                    [
                                        html.Span('On the Right - ', style={'font-style': 'italic'}),
                                        'The contributions of the input feature are represented by the black '
                                        'vectors. the bigger the contribution the longer the vector. The purple '
                                        'triangle is the value for which the contribution of all the feature is set '
                                        'to zero. Starting from this value and adding the contribution of each '
                                        'feature it is possible to move to the final point represented by the '
                                        'grey cross.'
                                    ]
                                )], md=6),
                        ],
                        className="p-3"
                    )
                ]),
            md=12,
            # className="my-3 p-3 bg-light border rounded-3"
        ),
        className="align-items-md-stretch",
    )
)

clustering_title = dbc.Container(
    dbc.Row(
        dbc.Col(
            html.Div(
                [
                    dbc.Row(
                        dbc.Col([
                            html.H3("Clustering analysis"),
                            html.Hr(className="mt-2 mb-4"),
                            html.P(
                                '''You can select with the lasso tool on the left and on the rigth the points of 
                                interest. In the violin plots at the bottom the left distribution is the distribution 
                                 of the points belonging to the cluster highlighted in the left figure, 
                                 while the right one are the distributions of the points selected on the right 
                                 scatter plot. The features are sorted in descending order from the most separated 
                                 distributions on the right to the least separated one to the left.''',
                                className='two-columns')],
                            md=12)
                    )
                ]),
            md=12,
            # className="my-3 p-3 bg-light border rounded-3"
        ),
        className="align-items-md-stretch",
    )
)

app.layout = html.Div([
    header,
    neighborhood_title,
    dbc.Container([
        dbc.Row([
            html.Div([
                dbc.Col(
                    html.Div(
                        # title above the sliders
                        [html.Div([
                            html.Div([
                                html.H5('Features ordered by relevance', className="table-header")
                            ],
                                style={'width': '35%'}),
                            html.Div([
                                html.H5('')
                            ],
                                style={'width': '45%'}),
                            html.Div([
                                html.H5('Value', className="table-header")
                            ],
                                style={'width': '10%', 'padding-left': '5px'}),
                            html.Div([
                                html.H5('Vector', className="table-header")
                            ],
                                style={'width': '10%', 'padding-left': '5px'})
                        ],
                            id='index',
                            className='index')] + [
                            # cycle for for the 10 sliders
                            html.Div([
                                html.Div([
                                    html.H5(str(columns[indices[i]]) + ' : ', id='slider_variable_name_' + str(i))
                                ],
                                    className='variable_name'),
                                html.Div([
                                    dcc.Slider(
                                        min=mins[indices[i]],
                                        max=maxs[indices[i]],
                                        value=query[indices[i]],
                                        step=steps[indices[i]],
                                        marks=select_ticks(mins[indices[i]], maxs[indices[i]],
                                                           exp_input_values[indices[i]]),
                                        id='slider_' + str(i)
                                    )], className='slider'),
                                html.Div(id='slider_' + str(i) + '_value', className='slider_value'),
                                html.Div([
                                    dcc.Graph(
                                        id='slider_graph_' + str(i),
                                        figure=slider_figs[i],
                                        config={'displayModeBar': False})],
                                    id='slider_' + str(i) + '_vector',
                                    className='slider_value'),
                            ],
                                id='variable_' + str(i),
                                className='variable') for i in range(10)],
                        id='inputs',
                        className='inputs'),
                    md=6
                ),
                # main chart plotly
                dbc.Col(
                    [
                        html.Div([
                            dcc.Graph(
                                id='space-plot',
                                # style={'width': '40vw', 'height': '40vw'},
                                figure=shap_fig,
                                config={"modeBarButtonsToRemove": ['select2d', 'lasso2d'],
                                        'displaylogo': False}
                            ),
                            html.P([
                                html.Span('NB: ', style={'font-style': 'italic'}),
                                'it is possible to select a new point just by clicking on it in the ',
                                'scatter plot.'
                            ], style={'font-style': 'italic', 'margin-top': '20px', 'text-align': 'center'})
                        ],
                            id='graph',
                            className='graph'),

                    ],
                    md=6),
            ],
                id='shap_page',
                className='shap_page'),

            # new Text row here!
            clustering_title,
            dbc.Row(
                [dbc.Col(
                    html.Div([
                        dcc.Graph(
                            id='left_plot',
                            figure=clustering_fig,
                            config={"modeBarButtonsToRemove": ['select2d'],
                                    'displaylogo': False})
                    ])
                ),
                    dbc.Col(
                        html.Div([
                            dcc.Graph(
                                id='right_plot',
                                figure=clustering_fig,
                                config={"modeBarButtonsToRemove": ['select2d'],
                                        'displaylogo': False})
                        ])
                    )],
                id='graphs'
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        dcc.Graph(
                            id='violin_plot',
                            figure=violin_plot),
                        className='violin-graph'),
                    md=12
                ),
                id='clustering_page',
                className='clustering_page'
            )
        ]
        )
    ], fluid=False)
],
    id='main_body',
    className='main_body')


############ callback functions #############

# function to update the plot and the values to the 10 most important features
@app.callback(
    # Output 1 value for each slider + the plot

    [dash.dependencies.Output('slider_' + str(i) + '_value', 'children') for i in range(10)] +
    [dash.dependencies.Output('space-plot', 'figure')],
    # As input we only have the 10 values from the sliders
    [dash.dependencies.Input('slider_' + str(i), 'value') for i in range(10)],
    # as state we need to know which point is selected
    dash.dependencies.State('space-plot', 'relayoutData'))
def update_output(value0, value1, value2, value3, value4, value5, value6, value7, value8, value9, fig_data):
    v = [value0, value1, value2, value3, value4, value5, value6, value7, value8, value9]
    # query is the value to explain
    # in this for loop change the value of the query according to the slider values
    global selected_point
    query = X_train[selected_point, :]

    # set the new values
    for i in range(len(v)):
        query[indices[i]] = v[i]

    # compute xai values
    exp_x, xai_x, exp_y, xai_y = compute_XAI_values(query)

    # compute new z position
    with torch.no_grad():
        z, _ = FFNN_CVAE_model.encode(torch.tensor(query).float())
        z = z.numpy()

    # update the scatter plot and the vector explanations as done before
    shap_fig.for_each_trace(
        lambda trace: trace.update(x=[z[0]], y=[z[1]]) if trace.name == "pointer" else (),
    )
    s = [exp_x, exp_y]
    s_zero = [exp_x, exp_y]

    shap_fig.add_shape(type="line",
                       x0=-20, y0=s_zero[1], x1=20, y1=s_zero[1],
                       line=dict(
                           color="black",
                           width=1,
                           dash="dashdot",
                       )
                       )

    shap_fig.add_shape(type="line",
                       x0=s_zero[0], y0=20, x1=s_zero[0], y1=-20,
                       line=dict(
                           color="black",
                           width=1,
                           dash="dashdot",
                       )
                       )


    for i in range(len(v)):
        # start
        annotations[i]['x'] = s[0]
        annotations[i]['y'] = s[1]
        # end
        s[0] += xai_x[indices[i]]
        s[1] += xai_y[indices[i]]
        annotations[i]['ax'] = s[0]
        annotations[i]['ay'] = s[1]
        annotations[i]['hovertext'] = str(columns[indices[i]])
    annotations[-1]['x'] = s[0]
    annotations[-1]['y'] = s[1]
    annotations[-1]['ax'] = z[0]
    annotations[-1]['ay'] = z[1]

    # update with zoom level as selected by the user
    try:
        shap_fig.update_layout({'xaxis': {'range': [fig_data['xaxis.range[0]'], fig_data['xaxis.range[1]']]},
                                'yaxis': {'range': [fig_data['yaxis.range[0]'], fig_data['yaxis.range[1]']]}
                                },
                               annotations=annotations)
    except:
        shap_fig.update_layout(annotations=annotations)

    out = v
    # round the numbers
    for i in range(len(out)):
        if out[i] % 1:
            out[i] = np.round(out[i], 2)
        else:
            out[i] = int(out[i])
    return out + [shap_fig]


# select a new point
@app.callback(
    # as output we need the new values for each sliders
    sum([[dash.dependencies.Output('slider_variable_name_' + str(i), 'children'),
          dash.dependencies.Output('slider_' + str(i), 'min'),
          dash.dependencies.Output('slider_' + str(i), 'max'),
          dash.dependencies.Output('slider_' + str(i), 'value'),
          dash.dependencies.Output('slider_' + str(i), 'step'),
          dash.dependencies.Output('slider_' + str(i), 'marks'),
          ] for i in range(10)], []) +
    [dash.dependencies.Output('slider_graph_' + str(i), 'figure') for i in range(10)],
    # as input the click data
    dash.dependencies.Input('space-plot', 'clickData'), prevent_initial_call=True
)
def display_click_data(clickData):
    global selected_point
    # change the selected point
    selected_point = clickData['points'][0]['pointIndex']
    # redo the shap computation
    query = X_train[selected_point, :]
    exp_x, xai_x, exp_y, xai_y = compute_XAI_values(query)
    global shap_fig
    shap_fig.for_each_trace(
        lambda trace: trace.update(x=[exp_x], y=[exp_y]) if trace.name == "expected_value" else (),
    )
    s = [exp_x, exp_y]
    # select only the 10 most important features
    global indices
    indices = np.argsort(np.sqrt(xai_x[:-1] ** 2 + xai_y[:-1] ** 2))[-10:][::-1]
    # update annotation
    for i in range(len(slider_names)):
        vec = np.array([xai_x[indices][i], xai_y[indices][i]])
        vec /= np.linalg.norm(vec)
        if query[indices][i] > exp_input_values[indices][i]:
            slider_annotations[i][0]['ax'] = vec[0]
            slider_annotations[i][0]['ay'] = vec[1]
            slider_annotations[i][1]['ax'] = -vec[0]
            slider_annotations[i][1]['ay'] = -vec[1]
        else:
            slider_annotations[i][0]['ax'] = -vec[0]
            slider_annotations[i][0]['ay'] = -vec[1]
            slider_annotations[i][1]['ax'] = vec[0]
            slider_annotations[i][1]['ay'] = vec[1]
        slider_figs[i].update_layout(annotations=slider_annotations[i])
    return sum([[str(X_train_df.columns[indices[i]]) + ' : ',
                 mins[indices[i]],
                 maxs[indices[i]],
                 query[indices[i]],
                 steps[indices[i]],
                 select_ticks(mins[indices[i]],
                              maxs[indices[i]],
                              exp_input_values[indices[i]])
                 ] for i in range(10)], []) + slider_figs


# update the violin plot after selecting new data
def update_violin_plot(idx_l, idx_r):
    global X_train_df
    x_l = X_train_df.iloc[idx_l == 1, :].copy()
    x_r = X_train_df.iloc[idx_r == 1, :].copy()
    ma = np.mean(x_l.to_numpy(), axis=0)
    mb = np.mean(x_r.to_numpy(), axis=0)
    indices = np.argsort(np.abs(ma - mb))[-15:]
    x_l = x_l.iloc[:, indices].melt()
    x_r = x_r.iloc[:, indices].melt()
    global violin_plot
    violin_plot.for_each_trace(
        lambda trace: trace.update(x=x_l['variable'], y=x_l['value']) if trace.name == "left" else (
            trace.update(x=x_r['variable'], y=x_r['value'])),
    )
    return violin_plot


# extract data from the selection and update the violin plot
@app.callback(
    dash.dependencies.Output('violin_plot', 'figure'),
    [dash.dependencies.Input('left_plot', 'selectedData'),
     dash.dependencies.Input('right_plot', 'selectedData')]
)
def compute_violin(selectedData_left, selectedData_right):
    if selectedData_left:
        selected_points_left = [selectedData_left['points'][i]['pointIndex'] for i in
                                range(len(selectedData_left['points']))]
        if selectedData_right:
            selected_points_right = [selectedData_right['points'][i]['pointIndex'] for i in
                                     range(len(selectedData_right['points']))]
            idx_l = np.zeros(len(X_train_df))
            idx_r = np.zeros(len(X_train_df))
            idx_r[selected_points_right] = 1
            idx_l[selected_points_left] = 1
            return update_violin_plot(idx_l, idx_r)
        else:
            idx_l = np.zeros(len(X_train_df))
            idx_l[selected_points_left] = 1
            idx_r = np.ones(len(X_train_df))
            return update_violin_plot(idx_l, idx_r)
    else:
        if selectedData_right:
            selected_points_right = [selectedData_right['points'][i]['pointIndex'] for i in
                                     range(len(selectedData_right['points']))]
            idx_l = np.ones(len(X_train_df))
            idx_r = np.zeros(len(X_train_df))
            idx_r[selected_points_right] = 1
            return update_violin_plot(idx_l, idx_r)
        else:
            violin_plot.for_each_trace(
                lambda trace: trace.update(x=X_test_df.melt()['variable'],
                                           y=X_train_df.melt()['value']) if trace.name == "right" else
                (trace.update(x=X_test_df.melt()['variable'], y=X_train_df.melt()['value'])),
            )
            return violin_plot


# In[23]:


server = FastAPI()
server.mount("/", WSGIMiddleware(app.server))

# run the app

if __name__ == "__main__":
    app.run_server(port=8090,
                   dev_tools_ui=True,
                   debug=True,
                   dev_tools_hot_reload=True,
                   threaded=True,
                   # host='127.0.0.1:2595'
                   )
