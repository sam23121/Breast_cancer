import pandas as pd
import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from causalnex.structure.notears import from_pandas
from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
sys.path.append(os.path.abspath(os.path.join('../scripts')))

df = pd.read_csv('../data/16_features.csv', index_col=[0])

le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

split = int(len(df) * 0.8)
train = df[:split]
test = df[split:]

sm = from_pandas(train, tabu_parent_nodes=['diagnosis'])
sm.remove_edges_below_threshold(0.8)
viz = plot_structure(
    sm,
    graph_attributes={"scale": "2.0", 'size':2.5},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename = "../assets/structure_model.png"
viz.draw(filename)
Image(filename)

x_frac = train.sample(frac=0.6)
sm1 = from_pandas(x_frac, tabu_parent_nodes=['diagnosis'])
sm1.remove_edges_below_threshold(0.8)
viz = plot_structure(
    sm1,
    graph_attributes={"scale": "2.0", 'size':2.5},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename = "../assets/structure_model_new.png"
viz.draw(filename)
Image(filename)

def jaccard_similarity(A, B):
    i = set(A).intersection(B)
    similarity = round(len(i)/(len(A)+len(B)-len(i)), 3)
    
    return similarity

similarity4 = jaccard_similarity(sm.edges, sm4.edges)
print(similarity4)

l = list(sm1.edges)

new_sm = sm1.copy()

for i in range(len(l)):
    if l[i][1] != 'diagnosis':
        new_sm.remove_edge(l[i][0], l[i][1])

l = list(new_sm.edges)

col = []
for i in range(len(l)):
    col.append(l[i][0])
col.append(l[1][1])

new_df = df[col]
split = int(len(new_df) * 0.8)
train2 = new_df[:split]
test2 = new_df[split:]

new_sm2 = from_pandas(train2, tabu_parent_nodes=['diagnosis'])
new_sm2.remove_edges_below_threshold(0.8)
viz = plot_structure(
    new_sm2,
    graph_attributes={"scale": "2.0", 'size':2.5},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename = "../assets/new_structure_model.png"
viz.draw(filename)
Image(filename)

for col in features:
    df[col] = tree_discretiser.transform(df[[col]])

split = int(len(df) * 0.9)
train = df[:split]
test = df[split:]

from causalnex.network import BayesianNetwork

bn = BayesianNetwork(sm2)
bn = bn.fit_node_states(df)

bn = bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")

bn.cpds["diagnosis"]

from causalnex.evaluation import classification_report
classification_report(bn, test, "diagnosis")

for col in features:
    df[col] = tree_discretiser.transform(df[[col]])

split = int(len(df) * 0.9)
train = df[:split]
test = df[split:]

from causalnex.network import BayesianNetwork

bn = BayesianNetwork(sm2)
bn = bn.fit_node_states(df)

bn = bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")

bn.cpds["diagnosis"]

from causalnex.evaluation import classification_report
classification_report(bn, test, "diagnosis")