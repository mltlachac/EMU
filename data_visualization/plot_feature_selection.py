#code authors: ML Tlachac & Joshua Lovering
#paper title: 'EMU: Early Mental Health Uncovering' 
#paper accessible at: 
#github: https://github.com/mltlachac/EMU

import argparse
import random
import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectFromModel

def parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--data",
        type=str,
        required= True,
        help="path to data csv."
    )
    arg_parser.add_argument(
        "--modality",
        type=str,
        required=True,
        help="type of modality, ex: 'audio_open' or 'audio_closed'."
    )
    arg_parser.add_argument(
        "--modality_name",
        type=str,
        required=True,
        help="type of modality, ex: 'Unstructured Audio' or 'Structured Audio'."
    )
    arg_parser.add_argument(
        "--feature_selection",
        type=str,
        default="pca",
        choices=["pca", "chi2", "etc"],
        help="feature selection method.",
    )
    arg_parser.add_argument(
        "--target_type",
        type=str,
        default="phq",
        choices=["phq", "gad", "q9"],
        help="type of target: PHQ-9 or GAD-7.",
    )
    # split by 10:
    #   0-9 = class:0
    #   10+ = class:1
    arg_parser.add_argument(
        "--split",
        type=int,
        default=10,
        choices=[*range(1, 28, 1)],
        help="value by which to split targetsinto classes.",
    )
    arg_parser.add_argument(
        "--target_to_classes",
        # action="store_false"
        type=bool,
        default=True,
        help="if True, targets will be split into classes based on --split value. if False, targets will be kept at original values.",
    )
    arg_parser.add_argument(
        "--run_pca_visualizations",
        type=bool,
        default=False,
        help="if True, pca visualizations will be generated.",
    )
    arg_parser.add_argument(
        "--feature_selection_before",
        action="store_true",
        help="if True, pca visualizations will be generated.",
    )
    return arg_parser
# plot line of variance between all pca features
def plot_pca_variance(var, num_f):
    x_ticks = list(range(1,num_f + 1))
    fig = plt.figure()
    plt.plot(x_ticks, var, linewidth=3)
    plt.title("Variance Covered by Each Principal Componant", fontsize = 15)
    plt.ylabel("Variance", fontsize = 12)
    plt.xlabel("Principal Component", fontsize = 12)
    plt.xticks(x_ticks)
    plt.savefig("machine_learning/experiments/feature_selection/" + args.modality + '_' + str(num_f) + "feature_" + args.target_type + "_var_PCA.png", bbox_inches = "tight")
    plt.close()
# plot scatter of two selected features
def plot_features(data, feature_name='n/a'):
    scatter_etc_df = data
    depressed = scatter_etc_df[scatter_etc_df.target == 1]
    not_depressed = scatter_etc_df[scatter_etc_df.target == 0]
    plt.figure(figsize = (5,4.5))

    if args.target_type == "phq":
        plt.scatter(depressed[0], depressed[1], alpha = 0.5, label = "Depressed", marker = '^', color = 'r')
        plt.scatter(not_depressed[0], not_depressed[1], alpha = 0.5, label = "Not Depressed", marker = 's', color = 'black')
    elif args.target_type == 'gad':
        plt.scatter(depressed[0], depressed[1], alpha = 0.5, label = "Anxious", marker = '^', color = 'r')
        plt.scatter(not_depressed[0], not_depressed[1], alpha = 0.5, label = "Not Anxious", marker = 's', color = 'black')
    if args.feature_selection == 'chi2':
        plt.title("Chi2 Selected Features for " + args.modality_name)
        plt.xlabel(feature_name[0])
        plt.ylabel(feature_name[1])
    elif args.feature_selection == 'etc':
        plt.title("ETC Selected Features for " + args.modality_name)
        plt.xlabel(feature_name[0])
        plt.ylabel(feature_name[1])
    elif args.feature_selection == 'pca':
        plt.title("PCA Selected Features for " + args.modality_name)
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
    plt.legend()
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig("machine_learning/experiments/feature_selection/" + args.modality + "_" + args.feature_selection + "_" + args.target_type + "_scatter.png", bbox_inches = "tight")
    plt.close()
# scale the data and run feature selection
def scale_feature_select(num_f, plot, cnames, random_state, data, label):
    min_max_scaler = preprocessing.MinMaxScaler().fit(data)
    data = min_max_scaler.transform(data)

    if args.feature_selection == 'pca':
        pca = PCA(n_components=num_f, random_state=random_state)
        pca.fit(data)
        X_pca = pca.transform(data)
        data = pd.DataFrame(X_pca).assign(target = label)

        pca_weights = pd.DataFrame(pca.components_, columns=cnames[1:-3]).transpose()
        pca_weights.to_csv('machine_learning/experiments/feature_selection/' + args.modality + '_' + str(num_f) + 'features' + "_" + args.target_type + '_pca_weights.csv')
        
        if plot == 'var':
            var = pca.explained_variance_ratio_
            plot_pca_variance(var, num_f)
        
    elif args.feature_selection == 'chi2':
        chi = SelectKBest(chi2, k=num_f)
        chi = chi.fit(data, label)
        data = chi.transform(data)
        data = pd.DataFrame(data).assign(target = label)

        feature_idx = chi.get_support()
        feature_name = cnames[1:-3][feature_idx].tolist()
        features_list.append(feature_name)

    elif args.feature_selection == 'etc':
        extra_tree_forest = ExtraTreesClassifier(random_state=random_state)
        extra_tree_forest = extra_tree_forest.fit(data, label)
        f_selected = SelectFromModel(extra_tree_forest, prefit=True, max_features=num_f)
        data = f_selected.transform(data)
        data = pd.DataFrame(data).assign(target = label)

        feature_idx = f_selected.get_support()
        feature_name = cnames[1:-3][feature_idx].tolist()
        features_list.append(feature_name)
    
    if args.feature_selection == 'chi2' or args.feature_selection == 'etc':
        with open('machine_learning/experiments/feature_selection/' + args.modality + "_" + args.feature_selection + '_' + str(num_f) + 'features_' + args.target_type + '.json', 'w') as jf:
            json.dump(features_list, jf, indent=2)
        if plot == 'scatter':
            plot_features(data, feature_name)
    else:
        if plot == 'scatter':
            plot_features(data)
# set random seed, input data, run functions
def main():
    random.seed(481998864)
    np.random.seed(481998864)
    random_state = 481998864

    df = pd.read_csv(args.data)
    cnames = df.columns

    df0 = pd.DataFrame()
    for c in cnames:
        df0[c] = df[c].fillna(0)

    data = df0

    #create index
    indices = data[data.columns[0]]
    # create feature subset
    # all feature data discluding col [0]: 'id', [-3]: 'q9', [-2]: 'phq', [-1]: 'gad'
    featureSubsetS = data[data.columns[1:-3]]

    #phq-9/gad-7
    if args.target_type == 'phq':
        target = pd.DataFrame(data = data[data.columns[-2]])
        target = target.rename(columns={'phq':'target'})
        target = target['target'].values.tolist()
    elif args.target_type == 'gad':
        target = pd.DataFrame(data = data[data.columns[-1]])
        target = target.rename(columns={'gad':'target'})
        target = target['target'].values.tolist()
    elif args.target_type == 'q9':
        target = pd.DataFrame(data = data[data.columns[-3]])
        target = target.rename(columns={'q9':'target'})
        target = target['target'].values.tolist()
    else: assert False, f"target_type value unexpected: {args.target_type}"

    # split target values into two classes
    if args.target_to_classes == True:
        for index, value in enumerate(target):
            if value >= args.split:
                target[index] = 1
            else: target[index] = 0

    num_features = 1
    scale_feature_select(num_features, "none", cnames, random_state, featureSubsetS, target)
    # num_features = 2
    # scale_feature_select(num_features, "scatter", cnames, random_state, featureSubsetS, target)
    # num_features = 10
    # scale_feature_select(num_features, "var", cnames, random_state, featureSubsetS, target)
# Initialization
if __name__ == "__main__":
    args = parser().parse_args()
    features_list = []
    main()