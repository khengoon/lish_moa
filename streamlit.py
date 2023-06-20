import streamlit as st
import numpy as np
import pandas as pd
import random 

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import log_loss

import category_encoders as ce

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F


from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import lottie_drug
# from streamlit_lottie import st_lottie
from model import *


# st.set_page_config(layout='wide')

################################################################################

# st_lottie(lottie_drug, height=200)

st.title('Drug Mechanisms of Action')

st.markdown('Disclaimer: This is a project by Low Kheng Oon. This application is not production ready. Use at your own discretion')

st.markdown('The Connectivity Map, a project within the Broad Institute of MIT and Harvard, the Laboratory for Innovation Science at Harvard (LISH), and the NIH Common Funds Library of Integrated Network-Based Cellular Signatures (LINCS), present this challenge with the goal of advancing drug development through improvements to MoA prediction algorithms.')

st.markdown('What is the Mechanism of Action (MoA) of a drug? And why is it important?')

st.markdown('In the past, scientists derived drugs from natural products or were inspired by traditional remedies. Very common drugs, such as paracetamol, known in the US as acetaminophen, were put into clinical use decades before the biological mechanisms driving their pharmacological activities were understood. Today, with the advent of more powerful technologies, drug discovery has changed from the serendipitous approaches of the past to a more targeted model based on an understanding of the underlying biological mechanism of a disease. In this new framework, scientists seek to identify a protein target associated with a disease and develop a molecule that can modulate that protein target. As a shorthand to describe the biological activity of a given molecule, scientists assign a label referred to as mechanism-of-action or MoA for short.')

st.markdown('How do we determine the MoAs of a new drug?')

st.markdown('One approach is to treat a sample of human cells with the drug and then analyze the cellular responses with algorithms that search for similarity to known patterns in large genomic databases, such as libraries of gene expression or cell viability patterns of drugs with known MoAs.')

st.markdown('In this competition, you will have access to a unique dataset that combines gene expression and cell viability data. The data is based on a new technology that measures simultaneously (within the same samples) human cells’ responses to drugs in a pool of 100 different cell types (thus solving the problem of identifying ex-ante, which cell types are better suited for a given drug). In addition, you will have access to MoA annotations for more than 5,000 drugs in this dataset.')

st.markdown('As is customary, the dataset has been split into testing and training subsets. Hence, your task is to use the training dataset to develop an algorithm that automatically labels each case in the test set as one or more MoA classes. Note that since drugs can have multiple MoA annotations, the task is formally a multi-label classification problem.')

st.markdown('How to evaluate the accuracy of a solution?')

st.markdown('Based on the MoA annotations, the accuracy of solutions will be evaluated on the average value of the logarithmic loss function applied to each drug-MoA annotation pair.')

st.markdown('If successful, you’ll help to develop an algorithm to predict a compound’s MoA given its cellular signature, thus helping scientists advance the drug discovery process.')

st.subheader('Basic data overview')

train = pd.read_csv('train_features.csv')
test = pd.read_csv('test_features.csv')

train['dataset'] = 'train'
test['dataset'] = 'test'

df = pd.concat([train, test]) 

st.markdown('Take a look into training and test sets.')

st.markdown('**train_features.csv** - Features for the training set. Features g- signify gene expression data, and c- signify cell viability data. cp_type indicates samples treated with a compound (cp_vehicle) or with a control perturbation (ctrl_vehicle); control perturbations have no MoAs; cp_time and cp_dose indicate treatment duration (24, 48, 72 hours) and dose (high or low).')

st.write(train.head())

st.markdown('**test_features.csv** - Features for the test data. You must predict the probability of each scored MoA for each row in the test data.')

st.write(test.head())

st.write('Number of rows in training set:', train.shape[0])
st.write('Number of columns in training set:', train.shape[1] - 1)
st.write('Number of rows in test set:', test.shape[0])
st.write('Number of columns in test set:', test.shape[1] - 1)

st.markdown('We can see that we have 872 float features 1 integer (cp_time) and 3 categorical (sig_id, cp_type and cp_dose).')

sample_submission = pd.read_csv('sample_submission.csv')
st.write(sample_submission.head())

st.markdown('Here we are going to check categorical features: cp_type, cp_time, cp_dose.')

st.subheader('Categories Visualization')

cp_width = 500
cp_height = 400
scatter_size = 600
WIDTH=800

ds = df.groupby(['cp_type', 'dataset'])['sig_id'].count().reset_index()

ds.columns = [
    'cp_type', 
    'dataset', 
    'count'
]

fig = px.bar(
    ds, 
    x='cp_type', 
    y="count", 
    color='dataset',
    barmode='group',
    orientation='v', 
    title='cp_type train/test counts', 
    width=cp_width,
    height=cp_height
)

st.plotly_chart(fig)

ds = df.groupby(['cp_time', 'dataset'])['sig_id'].count().reset_index()

ds.columns = [
    'cp_time', 
    'dataset', 
    'count'
]

fig = px.bar(
    ds, 
    x='cp_time', 
    y="count", 
    color='dataset',
    barmode='group',
    orientation='v', 
    title='cp_time train/test counts', 
    width=cp_width,
    height=cp_height
)

st.plotly_chart(fig)

ds = df.groupby(['cp_dose', 'dataset'])['sig_id'].count().reset_index()

ds.columns = [
    'cp_dose', 
    'dataset', 
    'count'
]

fig = px.bar(
    ds, 
    x='cp_dose', 
    y="count", 
    color='dataset',
    barmode='group',
    orientation='v', 
    title='cp_dose train/test counts', 
    width=cp_width,
    height=cp_height
)

st.plotly_chart(fig)

ds = df[df['dataset']=='train']
ds = ds.groupby(['cp_type', 'cp_time', 'cp_dose'])['sig_id'].count().reset_index()

ds.columns = [
    'cp_type', 
    'cp_time', 
    'cp_dose', 
    'count'
]

fig = px.sunburst(
    ds, 
    path=[
        'cp_type',
        'cp_time',
        'cp_dose' 
    ], 
    values='count', 
    title='Sunburst chart for all cp_type/cp_time/cp_dose',
    width=500,
    height=500
)

st.plotly_chart(fig)

st.subheader('Gene and cell features distribution')

st.markdown('Some distribution of randomly selected columns.')

train_columns = train.columns.to_list()
g_list = [i for i in train_columns if i.startswith('g-')]
c_list = [i for i in train_columns if i.startswith('c-')]

def plot_set_histograms(plot_list, title):
    fig = make_subplots(
        rows=4, 
        cols=3
    )
    
    traces = [
        go.Histogram(
            x=train[col], 
            nbinsx=100, 
            name=col
        ) for col in plot_list
    ]

    for i in range(len(traces)):
        fig.append_trace(
            traces[i], 
            (i // 3) + 1, 
            (i % 3) + 1
        )

    fig.update_layout(
        title_text=title,
        height=1000,
        width=WIDTH
    )
    st.plotly_chart(fig)

plot_list = [
    g_list[
        np.random.randint(0, len(g_list)-1)
    ] for i in range(50)
]

plot_list = list(set(plot_list))[:12]
plot_set_histograms(plot_list, 'Randomly selected gene expression features distributions')

plot_list = [
    c_list[
        np.random.randint(0, len(c_list)-1)
    ] for i in range(50)
]

plot_list = list(set(plot_list))[:12]
plot_set_histograms(plot_list, 'Randomly selected cell expression features distributions')

st.subheader('Training features correlation')

st.markdown("Let's see some correlation between randomly selected variables.")

columns = g_list + c_list
for_correlation = random.sample(columns, 50)
data = df[for_correlation]

f = plt.figure(
    figsize=(18, 18)
)

plt.matshow(
    data.corr(), 
    fignum=f.number
)

plt.xticks(
    range(data.shape[1]), 
    data.columns, 
    fontsize=14, 
    rotation=50
)

plt.yticks(
    range(data.shape[1]), 
    data.columns, 
    fontsize=14
)

cb = plt.colorbar()
cb.ax.tick_params(
    labelsize=13
)
st.pyplot(f)

st.markdown('Time to find pairs of features with high correlation.')

# cols = ['cp_time'] + columns
# all_columns = list()
# for i in range(0, len(cols)):
#     for j in range(i+1, len(cols)):
#         if abs(train[cols[i]].corr(train[cols[j]])) > 0.9:
#             all_columns = all_columns + [cols[i], cols[j]]

# all_columns = list(set(all_columns))

# st.write('Number of columns:', len(all_columns))
st.write('Number of columns:', 35)

st.markdown("In total we have 35 columns that have correlation with at least another 1 higher than 0.9. Let's visualize them.")
# st.markdown("Let's visualize them.")

# fig = make_subplots(
#     rows=12, 
#     cols=3
# )

# traces = [
#     go.Histogram(
#         x=train[col], 
#         nbinsx=100, 
#         name=col
#     ) for col in all_columns
# ]

# for i in range(len(traces)):
#     fig.append_trace(
#         traces[i], 
#         (i // 3) + 1, 
#         (i % 3) + 1
#     )

# fig.update_layout(
#     title_text='Highly correlated features',
#     height=1200
# )

# st.plotly_chart(fig)

st.subheader('Target analysis')

st.markdown("Let's check targets.")

train_target = pd.read_csv("train_targets_scored.csv")

st.write('Number of rows: ', train_target.shape[0])
st.write('Number of cols: ', train_target.shape[1])

st.write(train_target.head())

x = train_target.drop(['sig_id'], axis=1).sum(axis=0).sort_values().reset_index()

x.columns = [
    'column', 
    'nonzero_records'
]

x = x.tail(50)

fig = px.bar(
    x, 
    x='nonzero_records', 
    y='column', 
    orientation='h', 
    title='Columns with the higher number of positive samples (top 50)', 
    width=WIDTH,
    height=1000
)

st.plotly_chart(fig)

x = train_target.drop(['sig_id'], axis=1).sum(axis=0).sort_values(ascending=False).reset_index()

x.columns = [
    'column', 
    'nonzero_records'
]

x = x.tail(50)

fig = px.bar(
    x, 
    x='nonzero_records', 
    y='column', 
    orientation='h', 
    title='Columns with the lowest number of positive samples (top 50)', 
    width=WIDTH,
    height=1000 
)

st.plotly_chart(fig)

st.markdown("We can see that at least 50 target columns have number pf positive samples less than 20 (about 0.1%) !!!")

x = train_target.drop(['sig_id'], axis=1).sum(axis=0).sort_values(ascending=False).reset_index()

x.columns = [
    'column', 
    'count'
]

x['count'] = x['count'] * 100 / len(train_target)

fig = px.bar(
    x, 
    x='column', 
    y='count', 
    orientation='v', 
    title='Percent of positive records for every column in target', 
    width=1200,
    height=800 
)

st.plotly_chart(fig)

st.markdown("The biggest number of positive samples for 1 target column is 3.5%. So we deal here with highly imbalanced data.")

data = train_target.drop(['sig_id'], axis=1).astype(bool).sum(axis=1).reset_index()

data.columns = [
    'row', 
    'count'
]

data = data.groupby(['count'])['row'].count().reset_index()

fig = px.bar(
    data, 
    y=data['row'], 
    x="count", 
    title='Number of activations in targets for every sample', 
    width=WIDTH, 
    height=500
)

st.plotly_chart(fig)

data = train_target.drop(['sig_id'], axis=1).astype(bool).sum(axis=1).reset_index()

data.columns = [
    'row', 
    'count'
]

data = data.groupby(['count'])['row'].count().reset_index()

fig = px.pie(
    data, 
    values=100 * data['row'] / len(train_target), 
    names="count", 
    title='Number of activations in targets for every sample (Percent)', 
    width=WIDTH, 
    height=500
)

st.plotly_chart(fig)

st.markdown("We can see here that about 40% of sample have zeros in all columns and more than 50% have only one active target column.")

##################################################################################################################################

# https://www.kaggle.com/yasufuminakama/moa-pytorch-nn-starter/notebook

train_features = pd.read_csv(f'train_features.csv')
train_targets_scored = pd.read_csv(f'train_targets_scored.csv')
train_targets_nonscored = pd.read_csv(f'train_targets_nonscored.csv')

test_features = pd.read_csv(f'test_features.csv')
sample_submission = pd.read_csv(f'sample_submission.csv')

device = torch.device('cpu')

# ref: https://www.kaggle.com/c/lish-moa/discussion/180165
# check if labels for 'ctl_vehicle' are all 0.
train = train_features.merge(train_targets_scored, on='sig_id')
target_cols = [c for c in train_targets_scored.columns if c not in ['sig_id']]
cols = target_cols + ['cp_type']
train[cols].groupby('cp_type').sum().sum(1)

# constrcut train&test except 'cp_type'=='ctl_vehicle' data
st.write(f'Train features size: ', train_features.shape, f'Test features size: ', test_features.shape)
train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)
st.write(f'Train size: ', train.shape, f'Test size: ', test.shape)

st.subheader('CV Split')

cat_features = ['cp_time', 'cp_dose']
num_features = [c for c in train.columns if train.dtypes[c] != 'object']
num_features = [c for c in num_features if c not in cat_features]
num_features = [c for c in num_features if c not in target_cols]
target = train[target_cols].values

def cate2num(df):
    df['cp_time'] = df['cp_time'].map({24: 0, 48: 1, 72: 2})
    df['cp_dose'] = df['cp_dose'].map({'D1': 3, 'D2': 4})
    return df

train = cate2num(train)
test = cate2num(test)

folds = train.copy()
Fold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[target_cols])):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)
st.write(f'Folds size: ', folds.shape)

predictions = np.zeros((len(test), len(CFG.target_cols)))

if st.button('Run Prediction'):
    SEED = [0, 1, 2]
    for seed in SEED:
        _predictions = run_kfold_nn(CFG, test, num_features, cat_features, device='cpu', n_fold=5, seed=seed)
        
        predictions += _predictions / len(SEED)
    st.write('Prediction :', predictions)
