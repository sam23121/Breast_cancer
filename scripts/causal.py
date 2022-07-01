import pandas as pd
import sys
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from causalnex.structure.notears import from_pandas
from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.discretiser.discretiser_strategy import DecisionTreeSupervisedDiscretiserMethod
from causalnex.network import BayesianNetwork
from causalnex.evaluation import classification_report
from log import Logger


sys.path.append(os.path.abspath(os.path.join('../scripts')))




class Causal:

    def __init__(self) -> None:
        """Initilize class."""
        try:
            pass
            self.logger = Logger("causal.log").get_app_logger()
            self.logger.info(
            'Successfully Instantiated causal Class Object')
        except Exception:
            self.logger.exception(
            'Failed to Instantiate causal Class Object')
            sys.exit(1)

    def split_data(self, df):
        le = LabelEncoder()
        df['diagnosis'] = le.fit_transform(df['diagnosis'])
        split = int(len(df) * 0.7)
        train = df[:split]
        test = df[split:]
        self.logger.info('Successfully created a train and test files')

        return train, test



    def plt_structure(self, train, frac=1, parent_node='diagnosis'):
        x_frac = train.sample(frac=frac)
        sm = from_pandas(x_frac, tabu_parent_nodes=[parent_node])
        sm.remove_edges_below_threshold(0.8)
        viz = plot_structure(
            sm,
            graph_attributes={"scale": "2.0", 'size':2.5},
            all_node_attributes=NODE_STYLE.WEAK,
            all_edge_attributes=EDGE_STYLE.WEAK)
        filename = "../assets/structure_model_new_other.png"
        self.logger.info('Successfully created a structure model')
        viz.draw(filename)
        Image(filename)

        return sm

    def jaccard_similarity(self, A, B):
        i = set(A).intersection(B)
        similarity = round(len(i)/(len(A)+len(B)-len(i)), 3)
        self.logger.info('Successfully caluated similarity')

        
        return similarity



    def remove_edges(self, sm):
        l = list(sm.edges)
        new_sm = sm.copy()
        for i in range(len(l)):
            if l[i][1] != 'diagnosis':
                new_sm.remove_edge(l[i][0], l[i][1])

        l = list(new_sm.edges)
        new_col = []
        for i in range(len(l)):
            new_col.append(l[i][0])
        new_col.append(l[1][1])
        self.logger.info('Successfully created a remove edges')


        return new_sm, new_col




    def discrete(self, df, parent_node='diagnosis'):
        le = LabelEncoder()
        df['diagnosis'] = le.fit_transform(df['diagnosis'])
        features = list(df.columns.difference([parent_node]))
        tree_discretiser = DecisionTreeSupervisedDiscretiserMethod(
            mode="single", 
            tree_params={"max_depth": 2, "random_state": 2021},
        )
        tree_discretiser.fit(
            feat_names=features, 
            dataframe=df, 
            target_continuous=True,
            target=parent_node,
        )

        for col in features:
            df[col] = tree_discretiser.transform(df[[col]])
        self.logger.info('Successfully discritzed the dataframe')


        return df


    def bayesian(self, sm, df, target='diagnosis'):
        bn = BayesianNetwork(sm)
        bn = bn.fit_node_states(df)
        train, test = self.split_data(df)
        bn = bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")
        table = bn.cpds[target]

        report = classification_report(bn, test, "diagnosis")
        self.logger.info('Successfully created a report and table')


        return table, report


    def compare(self, sm, sm2):
        similarity = self.jaccard_similarity(sm.edges, sm2.edges)
        self.logger.info('Successfully compare two structure model')

        return similarity
