#!/usr/bin/env python
# coding: utf-8

import sys, getopt

dataset = ''

try:
    opts, args = getopt.getopt(sys.argv[1:],"hd:r:",["dataset=","reduced="])
except getopt.GetoptError:
    print('app.py -d <titanic or adult>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('app.py -d <dataset name> -r <number of points to visualise> \n -d --dataset : name of the dataset to utilise, only "titanic" or "adult" are supported \n -r --reduced: option to reduce the number of points to visualise, if this option is not provided then all the test dataset is used for visualisation. For the titanic dataset a number less than 5000 is recommeded')
        sys.exit()
    elif opt in ("-d", "--dataset"):
        dataset = arg
    elif opt in ("-r", "--reduced"):
        points = arg
print(dataset,' selected')


import os
import re

import numpy as np
import pandas as pd

import plotly.graph_objects as go


# # Data Loading


if dataset == 'titanic':
    #https://www.kaggle.com/c/titanic/data
    tit_sub = pd.read_csv('./data/Titanic/gender_submission.csv')
    tit_train = pd.read_csv('./data/Titanic/train.csv')
    tit_test = pd.read_csv('./data/Titanic/test.csv')
    df_train_final = pd.read_pickle("./data/Titanic/df_train_final")
    df_test_final = pd.read_pickle("./data/Titanic/df_test_final")

    from sklearn.preprocessing import StandardScaler
    scaler_cols = ['Age', 'Fare', 'Name_Length', 'Family_Size', 'Name_Length', 'Ticket_Frequency', 'Fare_Family_Size', 'Fare_Cat_Pclass']
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
    X_train = df_train_final[features]
    Y_train = df_train_final['Survived']
    X_test = df_test_final.reset_index(drop=True)

    c=pd.read_csv('./data/Titanic/titanic_test_labels.csv')
    test_data_with_labels = c.copy()
    for i, name in enumerate(test_data_with_labels['name']):
        if '"' in name:
            test_data_with_labels['name'][i] = re.sub('"', '', name)
    for i, name in enumerate(tit_test['Name']):
        if '"' in name:
            tit_test['Name'][i] = re.sub('"', '', name)
    survived = []
    for name in tit_test['Name']:
        survived.append(int(test_data_with_labels.loc[test_data_with_labels['name'] == name]['survived'].values[-1]))
    Y_test = pd.Series(survived,index=X_test.index)

    Y_train = Y_train.to_numpy()
    Y_test = Y_test.to_numpy()
elif dataset == 'adult':
    train_df = pd.read_csv('./data/Adult/adult.data',
                           header=None,
                           names=['age','workclass','fnlwgt','education','education_num','marital_status','occupation',
                                  'relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','target'])
    test_df = pd.read_csv('./data/Adult/adult.test',
                           header=0,
                           names=['age','workclass','fnlwgt','education','education_num','marital_status','occupation',
                                  'relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','target'])

    train_df = train_df[np.sum(train_df.values==' ?',axis=1)==0]
    test_df = test_df[np.sum(test_df.values==' ?',axis=1)==0]

    train_df = train_df.drop(columns='education_num')
    test_df = test_df.drop(columns='education_num')
    
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

    categorical_columns = ['workclass','education','marital_status','occupation','relationship','race','sex','native_country']

    enc = OneHotEncoder(handle_unknown='ignore')
    train_df[enc.get_feature_names()] = enc.fit_transform(train_df.loc[:,categorical_columns]).astype(int).toarray()
    test_df[enc.get_feature_names()] = enc.transform(test_df.loc[:,categorical_columns]).astype(int).toarray()

    ordinal_columns = ['target']

    enc = OrdinalEncoder()
    train_df.loc[:,ordinal_columns] = enc.fit_transform(train_df.loc[:,ordinal_columns].values).astype(int)
    test_df.loc[:,ordinal_columns] = enc.transform(test_df.loc[:,ordinal_columns].values).astype(int)

    train_df = train_df.drop(columns=categorical_columns)
    test_df = test_df.drop(columns=categorical_columns)
    
    from sklearn.preprocessing import MinMaxScaler
    scaler_cols = ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']
    std = MinMaxScaler()
    std.fit(train_df.loc[:,scaler_cols])
    train_df.loc[:, scaler_cols] = std.transform(train_df.loc[:,scaler_cols])
    test_df.loc[:, scaler_cols] = std.transform(test_df.loc[:,scaler_cols])
    
    X_train = train_df.drop('target',axis=1)
    Y_train = train_df.loc[:,'target']
    X_test = test_df.drop('target',axis=1)
    Y_test = test_df.loc[:,'target']
    
    
else:
    raise Exception('dataset name not recognized, only adult or titanic are supported')


# # Model


import torch
import torch.nn as nn
import torch.nn.functional as F


if dataset == 'titanic':
    class FFNN_VAE(nn.Module):
        def __init__(self, input_shape, hidden, dropout_p=0.3, latent_dim=2):
            super(FFNN_VAE, self).__init__()

            # encoding components

            self.fc1 = nn.Linear(input_shape,hidden[0])
            self.drop1 = nn.Dropout(dropout_p)
            self.fc2 = nn.Linear(hidden[0],hidden[1])
            # Latent vectors mu and sigma
            self.fc3_mu = nn.Linear(hidden[1], latent_dim)      
            self.fc3_logvar = nn.Linear(hidden[1], latent_dim)  

            # Sampling vector
            self.fc4 = nn.Sequential(
                nn.Linear(latent_dim, hidden[1]),
                nn.Dropout(dropout_p),
                nn.ReLU(inplace=True)
            )
            # Decoder
            self.fc5 = nn.Sequential(
                nn.Linear(hidden[1], hidden[0]),
                nn.Dropout(dropout_p),
                nn.ReLU(inplace=True)
            )
            self.fc6 = nn.Linear(hidden[0], input_shape)

        def encode(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(self.drop1(x)))
            mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
            return mu, logvar

        def reparameterize(self, mu, logvar):
            if self.training:
                std = logvar.mul(0.5).exp_()
                eps = Variable(std.data.new(std.size()).normal_())
                return eps.mul(std).add_(mu)
            else:
                return mu

        def decode(self, z):
            x = self.fc4(z)
            x = self.fc5(x)
            x = self.fc6(x)
            return x

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_reconst = self.decode(z)
            return x_reconst, z, mu, logvar

    # hyperparameters
    latent_dim = 2    # latent dim extracted by 2D CNN
    dropout_p = 0       # dropout probability
    hidden = [24,12]

elif dataset == 'adult':
    class FFNN_VAE(nn.Module):
        def __init__(self, input_shape, hidden, latent_dim=2, dropout_p=0):
            super(FFNN_VAE, self).__init__()

            # encoding components

            self.fc1 = nn.Linear(input_shape, hidden[0])
            self.fc2 = nn.Linear(hidden[0], hidden[1])
            self.fc3 = nn.Linear(hidden[1], hidden[2])
            # Latent vectors mu and sigma
            self.fc4_mu = nn.Linear(hidden[2], latent_dim)      
            self.fc4_logvar = nn.Linear(hidden[2], latent_dim) 

            # Sampling vector
            self.fc5 = nn.Linear(latent_dim, hidden[2])
            # Decoder
            self.fc6 = nn.Linear(hidden[2], hidden[1])
            self.fc7 = nn.Linear(hidden[1], hidden[0])
            self.fc8 = nn.Linear(hidden[0], input_shape)

        def encode(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            mu, logvar = self.fc4_mu(x), self.fc4_logvar(x)
            return mu, logvar

        def reparameterize(self, mu, logvar):
            if self.training:
                std = logvar.mul(0.5).exp_()
                eps = Variable(std.data.new(std.size()).normal_())
                return eps.mul(std).add_(mu)
            else:
                return mu

        def decode(self, z):
            x = F.relu(self.fc5(z))
            x = F.relu(self.fc6(x))
            x = F.relu(self.fc7(x))
            x = self.fc8(x)
            return x

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_reconst = self.decode(z)
            return x_reconst, z, mu, logvar

    # hyperparameters
    latent_dim = 2    
    hidden = [50,25,12]
    dropout_p = 0
    
    
# Detect devices
use_cuda = torch.cuda.is_available()                   
device = torch.device("cuda:0" if use_cuda else "cpu")   

# Create Model
FFNN_VAE_model = FFNN_VAE(len(X_train.columns),hidden = hidden, dropout_p=dropout_p,latent_dim=latent_dim)
print(FFNN_VAE_model)

if dataset == 'titanic':
    FFNN_VAE_model.load_state_dict(torch.load('./models/VAE_Titanic.pt'))
elif dataset == 'adult':
    FFNN_VAE_model.load_state_dict(torch.load('./models/VAE_Adult.pt'))

with torch.no_grad():
    FFNN_VAE_model.eval()
    rec_x_train, z_train, mu, logvar = FFNN_VAE_model(torch.tensor(X_train.to_numpy(),dtype=torch.float32))
    rec_x_test, z_test, mu, logvar = FFNN_VAE_model(torch.tensor(X_test.to_numpy(),dtype=torch.float32))


if dataset == 'adult':
    reduced = np.random.permutation(range(15059))[:1000]
    X_test = X_test.iloc[reduced]
    X_test.reset_index(drop=True)
    Y_test = Y_test.iloc[reduced]
    Y_test.reset_index(drop=True)
    z_test = z_test[reduced]

# # APP

import plotly.express as px
import plotly.graph_objects as go

x_min, x_max = z_test[:, 0].min() - 0.1, z_test[:, 0].max() + 0.1
if dataset == 'titanic':
    y_min, y_max = z_test[:, 1].min() - 1, z_test[:, 1].max() + 1
elif dataset == 'adult':
    y_min, y_max = z_test[:, 1].min() - 0.1, z_test[:, 1].max() + 0.1

fig = go.Figure()
colorscale = [[0, '#3b4cc0'],[1, '#b40426']]

if dataset == 'titanic':
    target_name = 'dead'
elif dataset == 'adult':
    target_name = '<50K'

fig.add_trace(go.Scatter(x=z_test[:, 0].numpy(), y=z_test[:, 1].numpy(),mode='markers',name=target_name,
                         marker=dict(
                             size=10,
                             symbol='circle',
                             color=Y_test, 
                             opacity=0.5,
                             colorscale=colorscale,
                             line=dict(width=1,
                                       color='Black'))))

fig.add_trace(go.Scatter(x=[0], y=[0] ,mode='markers',name='pointer',showlegend=False,
                         marker=dict(
                             size=30,
                             symbol='x',
                             color='black')))

fig.add_trace(go.Scatter(x=[0], y=[0] ,mode='markers',name='expected_value',showlegend=False,
                         marker=dict(
                             size=15,
                             symbol='triangle-up',
                             color='purple')))

fig.update_yaxes(range=[y_min, y_max])
fig.update_xaxes(range=[x_min, x_max])
names = ['vector_0','vector_1','vector_2','vector_3','vector_4','vector_5','vector_6','vector_7','vector_8','vector_9']

annotations = [
    go.layout.Annotation(
            #start
            x=0, 
            y=0,
            xref="x",
            yref="y",
            #end
            ax=0, 
            ay=0,
            axref = "x", 
            ayref = "y",
            showarrow=True,
            arrowside='start',
            arrowhead=2, # type od head [0,8]
            arrowsize=1, # head dimension 
            arrowwidth=2, # arrow dimension
            arrowcolor="#000000",
            name=name,
            text='  ',
            hovertext='baseline',
            font=dict(
                family="Courier New, monospace",
                size=1,
                color="#ffffff"
            ),
            bgcolor='#ff7f0e'
            ) for name in names]+[
     go.layout.Annotation(
            #start
            x=0, 
            y=0,
            xref="x",
            yref="y",
            #end
            ax=0, 
            ay=0,
            axref = "x", 
            ayref = "y",
            showarrow=True,
            arrowside='start',
            arrowhead=2, # type od head [0,8]
            arrowsize=1, # head dimension 
            arrowwidth=2, # arrow dimension
            arrowcolor="#000000",
            name='others_contrib',
            text='  ',
            hovertext='other contributions',
            font=dict(
                family="Courier New, monospace",
                size=1,
                color="#ffffff"
            ),
            bgcolor='#ff7f0e'
            )
]

fig.update_layout(
    {'xaxis':{'range':[x_min,x_max]},
     'yaxis':{'range':[y_min,y_max]}},
    showlegend=True,
    autosize=True,
    annotations=annotations,
    #width=1000,
    #height=500,
    margin=dict(
    l=5,
    r=0,
    b=5,
    t=10,
    pad=0
));
#fig.show()

import shap

def wrapper(X):
    FFNN_VAE_model.eval()
    mu, log_var = FFNN_VAE_model.encode(torch.tensor(X).float())
    return mu.detach().numpy()


# use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(wrapper, shap.sample(X_train,100), link="identity")

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

mins = np.min(X_train.values,axis=0)
maxs = np.max(X_train.values,axis=0)
selected_point = 0
change_point = 0
query = X_test.values[selected_point,:]
exp_input_values = X_test.mean(axis=0).values

# main function to output the xai scores
def compute_XAI_values(query):
    
    #original_shap
    shap_values = explainer.shap_values(query, nsamples='auto')
    return explainer.expected_value[0], shap_values[0] ,explainer.expected_value[1], shap_values[1]
    

exp_x, xai_x, exp_y, xai_y = compute_XAI_values(query)
fig.for_each_trace(
        lambda trace: trace.update(x=[exp_x], y=[exp_y]) if trace.name == "expected_value" else (),
        )

s = [exp_x, exp_y]
#select only the 10 most important features
indices = np.argsort(np.sqrt(xai_x**2+xai_y**2))[-10:][::-1]
# list of the increments of the sliders
if dataset == 'titanic':
    steps = [1,1,0.001,0.001,1,0.001,1,1,1,1,1,1,1,1,1,0.001,0.001,1,0.001,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
elif dataset == 'adult':
    steps=[0.1]*5+[1]*98

import json
import dash
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# function to select the ticks of the sliders
def select_ticks(m, M, x, e):
    # round values
    if x % 1:
        x = np.round(x,3)
    else: x = int(x)
    if m % 1:
        m = np.round(m,3)
    else: m = int(m)
    if M % 1:
        M = np.round(M,3)
    else: M = int(M)
    if e%1:
        e = np.round(e,3)
    else: e = int(e)

    if x==m:
        ticks = dict(sorted({e:{'label':'e','style': {'color': '#800081'}},
                    x:{'label':'x','style': {'color': '#000000'}},
                    M:{'label':str(M),'style': {'color': '#ed4731'}}
                    }.items()))
    elif x==M:
        ticks = dict(sorted({m:{'label':str(m), 'style': {'color': '#77b0b1'}},
                    e:{'label':'e','style': {'color': '#800081'}},
                    x:{'label':str(x),'style': {'color': '#000000'}}
                    }.items()))
    else:
        ticks = dict(sorted({m:{'label':str(m), 'style': {'color': '#77b0b1'}},
                    e:{'label':'e','style': {'color': '#800081'}},
                    x:{'label':str(x),'style': {'color': '#000000'}},
                    M:{'label':str(M),'style': {'color': '#ed4731'}}
                    }.items()))
    return ticks

app = JupyterDash(__name__,meta_tags=[{"name": "viewport", "content": "width=device-width"}])

app.layout = html.Div([
    html.Div([
        html.H3('Latent Space Explanation'),
        ],
        id='title',
        className='title'),
    html.Div([
        html.Div([
            html.Div(
                # title above the sliders
                [html.Div([
                        html.Div([
                            html.H6('Name')
                        ],
                        style={'width':'35%','text-decoration': 'underline'}),
                        html.Div([
                            html.H6('Slider')
                        ],
                        style={'width':'45%','text-decoration': 'underline'}),
                        html.Div([
                            html.H6('Value')
                        ],
                        style={'width':'10%','text-decoration': 'underline'}),
                        html.Div([
                            html.H6('Sx')
                        ],
                        style={'width':'10%','text-decoration': 'underline'}),
                        html.Div([
                            html.H6('Sy')
                        ],
                        style={'width':'10%','text-decoration': 'underline'})
                    ],
                id = 'index',
                className = 'index')]+[
                # cycle for for the 10 sliders
                 html.Div([
                    html.Div([
                        html.H6(str(X_train.columns[indices[i]])+' : ', id = 'slider_variable_name_'+str(i))
                        ],
                        className='variable_name'),
                    dcc.Slider(
                        min = mins[indices[i]],
                        max = maxs[indices[i]],
                        value = query[indices[i]],
                        step = steps[indices[i]],
                        marks = select_ticks(mins[indices[i]], maxs[indices[i]], query[indices[i]], exp_input_values[indices[i]]),
                        id = 'slider_'+str(i),
                        className = 'slider'),
                    html.Div(id = 'slider_'+str(i)+'_value', className = 'slider_value'),
                    html.Div(id = 'slider_'+str(i)+'_xai_x', className = 'slider_value'),
                    html.Div(id = 'slider_'+str(i)+'_xai_y', className = 'slider_value')
                    ],
                    id = 'variable_'+str(i),
                    className = 'variable') for i in range(10)],
                id='inputs',
                className='inputs'),
            html.Div([
                dcc.Graph(
                    id='space-plot',
                    figure=fig)
                ],
                id='graph',
                className='graph'),
            ],
            id='main_page',
            className='main_page')
        ],
        id='main_body',
        className='main_body'
        )
    ])

#function to update the plot and the values to the 10 most important features
@app.callback(
    #Output 3 value for slider + the plot
     sum([[dash.dependencies.Output('slider_'+str(i)+'_value', 'children'),
      dash.dependencies.Output('slider_'+str(i)+'_xai_x', 'children'),
      dash.dependencies.Output('slider_'+str(i)+'_xai_y', 'children'),
     ] for i in range(10)],[])+[dash.dependencies.Output('space-plot', 'figure')],
    # As input we only have the 10 value from the sliders
    [dash.dependencies.Input('slider_'+str(i), 'value') for i in range(10)],
    [dash.dependencies.State('space-plot', 'relayoutData')])
def update_output(value0,value1,value2,value3,value4,value5,value6,value7,value8,value9,fig_data):
    v = [value0,value1,value2,value3,value4,value5,value6,value7,value8,value9]
    # query is the value to explain 
    # in this for loop change the value of the query according to the slider values
    query = X_test.values[selected_point,:]
    # Problem when selecting a new point: the slider values changes one at a time and this function is triggered 10 times!!! one for every changes with the older values
    global change_point
    if change_point:
        if dash.callback_context.triggered[0]['prop_id'] != 'slider_9.value':
            raise PreventUpdate
        else: change_point = 0
    for i in range(len(v)):
        query[indices[i]] = v[i]
    # compute xai values
    exp_x, xai_x, exp_y, xai_y = compute_XAI_values(query)
    #compute new z position
    with torch.no_grad():
        z , _ = FFNN_VAE_model.encode(torch.tensor(query).float())
        z = z.numpy()
    #update the scatter plot and the vector explanations
    fig.for_each_trace(
        lambda trace: trace.update(x=[z[0]], y=[z[1]]) if trace.name == "pointer" else (),
        )
    s = [exp_x, exp_y]
    for i in range(len(v)):
        #start
        annotations[i]['x'] = s[0]
        annotations[i]['y'] = s[1]
        #end
        s[0] += xai_x[indices[i]]
        s[1] += xai_y[indices[i]]
        annotations[i]['ax'] = s[0]
        annotations[i]['ay'] = s[1]
        annotations[i]['hovertext'] = str(X_train.columns[indices[i]])
    annotations[-1]['x'] = s[0]
    annotations[-1]['y'] = s[1]
    annotations[-1]['ax'] = z[0]
    annotations[-1]['ay'] = z[1]
    # update with zoom level as selected by the user
    try:
        fig.update_layout({'xaxis':{'range':[fig_data['xaxis.range[0]'],fig_data['xaxis.range[1]']]},
                           'yaxis':{'range':[fig_data['yaxis.range[0]'],fig_data['yaxis.range[1]']]},
                          },
                          annotations = annotations)
    except: fig.update_layout(annotations = annotations)
    # the output must be a list [slider_value,xai_x,xai_y]*10+[fig]
    v = np.array(v)
    xai_x = xai_x[indices]
    xai_y = xai_y[indices]
    out = list(np.vstack([v,xai_x,xai_y]).transpose().ravel())
    # round the numbers
    for i in range(len(out)):
        if out[i]%1:
            out[i] = np.round(out[i],3)
        else: out[i] = int(out[i])
    return out + [fig]
    
# select the point
@app.callback(
    sum([[dash.dependencies.Output('slider_variable_name_'+str(i), 'children'),
      dash.dependencies.Output('slider_'+str(i), 'min'),
      dash.dependencies.Output('slider_'+str(i), 'max'),
      dash.dependencies.Output('slider_'+str(i), 'value'),
      dash.dependencies.Output('slider_'+str(i), 'step'),
      dash.dependencies.Output('slider_'+str(i), 'marks'),
     ] for i in range(10)],[]),
    dash.dependencies.Input('space-plot', 'clickData')
    )
def display_click_data(clickData):
    try:
        global selected_point
        selected_point = clickData['points'][0]['pointIndex']
        global change_point
        change_point = 1
        query = X_test.values[selected_point,:]
        exp_x, xai_x, exp_y, xai_y = compute_XAI_values(query)
        global fig
        fig.for_each_trace(
            lambda trace: trace.update(x=[exp_x], y=[exp_y]) if trace.name == "expected_value" else (),
        )
        s = [exp_x, exp_y]
        #select only the 10 most important features
        global indices
        indices = np.argsort(np.sqrt(xai_x**2+xai_y**2))[-10:][::-1]
        return sum([[str(X_train.columns[indices[i]])+' : ',mins[indices[i]],maxs[indices[i]],query[indices[i]],steps[indices[i]],select_ticks(mins[indices[i]], maxs[indices[i]], query[indices[i]], exp_input_values[indices[i]])] for i in range(10)],[])
    except:
        raise PreventUpdate
    
# run the app
app.run_server(mode='external',
               port=8090, 
               dev_tools_ui=True, 
               #debug=True,
               dev_tools_hot_reload=False,
               threaded=True,
              )





