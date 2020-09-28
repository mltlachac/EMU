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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from statistics import mean 
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# parser for program arguments
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
        "--feature_selection",
        type=str,
        default="pca",
        choices=["pca", "chi2", "etc"],
        help="feature selection method.",
    )
    arg_parser.add_argument(
        "--cross_validation",
        type=str,
        default="loo",
        choices=["loo", "tts"],
        help="cross validation method: leave-one-out or train-test-split.",
    )
    arg_parser.add_argument(
        "--sampling",
        type=str,
        default="regular_oversampling",
        choices=["regular_oversampling", "regular_undersampling", "smote"],
        help="sampling method.",
    )
    arg_parser.add_argument(
        "--target_type",
        type=str,
        default="phq",
        choices=["phq", "gad", "q9"],
        help="type of target: PHQ-9, GAD-7, or PHQ-q9.",
    )
    arg_parser.add_argument(
        "--oppo_target",
        action="store_true",
        help="if True, the opposite target value will be included as a feature after feature selection.",
    )
    # split by 10:
    #   0-9 = class:0
    #   10+ = class:1
    arg_parser.add_argument(
        "--split",
        type=int,
        default=10,
        help="value by which to split targets into classes.",
    )
    arg_parser.add_argument(
        "--target_to_classes",
        action="store_false",
        help="if True, targets will be split into classes based on --split value. if False, targets will be kept at original values.",
    )
    return arg_parser
# Random Undersample uses imblearn library by randomly sampling data from the majority set 
def random_undersample(train_data):
    rus = RandomUnderSampler()
    featureSubset, featureSubset_labels = rus.fit_sample(
        train_data.drop('target', axis=1), 
        train_data['target'])
    return featureSubset, featureSubset_labels
# Random Upsample uses imblearn library by randomly sampling data from the minority set 
def random_upsample(train_data):
    ros = RandomOverSampler()
    featureSubset, featureSubset_labels = ros.fit_sample(
        train_data.drop('target', axis=1), 
        train_data['target'])
    return featureSubset, featureSubset_labels
# SMOTE Upsample uses SMOTE library by generating new data after analyzing the minority set
def SMOTE_upsample(train_data):
    oversample = SMOTE()
    featureSubset, featureSubset_labels = oversample.fit_sample(
        train_data.iloc[:,:-1], 
        train_data.iloc[:,-1])
    return featureSubset, featureSubset_labels
# Scale features & then run feature selection on train set and extract those features from test set
def scale_feature_select(train_data, train_label, test_data, test_label, num_f, cnames, fnames, random_state):
    min_max_scaler = preprocessing.MinMaxScaler().fit(train_data)
    train_data = min_max_scaler.transform(train_data)
    test_data = min_max_scaler.transform(test_data)
    if args.feature_selection == 'pca':
        # pca_columns = []
        # for num_f in nFeatureList:
        pca = PCA(n_components=num_f, random_state=random_state)
        pca = pca.fit(train_data)

        pca_weights = pd.DataFrame(pca.components_, columns=fnames).transpose()
        pca_weights.to_csv('machine_learning/results/pca_weights_' + args.modality + '.csv')

        train_data = pca.transform(train_data)
        train_data = pd.DataFrame(train_data).assign(target = train_label)
        test_data = pca.transform(test_data)
        test_data = pd.DataFrame(test_data).assign(target = test_label)
        # var = pca.explained_variance_ratio_
        # if args.run_pca_visualizations == True:
        #     run_pca_visualizations(pca, var, featureDF, pca_columns)
    elif args.feature_selection == 'chi2':
        chi = SelectKBest(chi2, k=num_f)
        chi = chi.fit(train_data, train_label)

        feature_idx = chi.get_support()
        feature_name = cnames[1:-3][feature_idx].tolist()
        features_list.append(feature_name)

        # booleans of columns list
        # mask = selector.get_support()
        train_data = chi.transform(train_data)
        train_data = pd.DataFrame(train_data).assign(target = train_label)
        # indices of chi2 columns
        # f_cols = train_data.columns[mask]
        test_data = chi.transform(test_data)
        test_data = pd.DataFrame(test_data).assign(target = test_label)
    elif args.feature_selection == 'etc':
        # feature_clf = ExtraTreesClassifier(n_estimators=50)
        extra_tree_forest = ExtraTreesClassifier(random_state=random_state)
        extra_tree_forest = extra_tree_forest.fit(train_data, train_label)
        f_selected = SelectFromModel(extra_tree_forest, prefit=True, max_features=num_f)

        feature_idx = f_selected.get_support()
        feature_name = cnames[1:-3][feature_idx].tolist()
        features_list.append(feature_name)

        train_data = f_selected.transform(train_data)
        train_data = pd.DataFrame(train_data).assign(target = train_label) 
        test_data = f_selected.transform(test_data)
        test_data = pd.DataFrame(test_data).assign(target = test_label) 
    return train_data, test_data
# Add opposite target as a feature
def oppo_target_as_feature(train_data, test_data, train_index, test_index, oppo_target, oppo_target_name):
    length = len(train_data.columns)
    train_oppo = oppo_target.loc[(train_index)].reset_index(drop=True)
    test_oppo = oppo_target.loc[(test_index)].reset_index(drop=True)

    train_data.insert(length-1, oppo_target_name, train_oppo, True)
    test_data.insert(length-1, oppo_target_name, test_oppo, True)
    return train_data, test_data
# finish preparing data and run through models
def run_models(train_data, train_label, test_data, test_label, train_index, test_index, num_f, cnames, fnames, random_state, oppo_target=None, oppo_target_name=None, temp_num=0):
    # run feature selection
    train_data, test_data = scale_feature_select(train_data, train_label, test_data, test_label, num_f, cnames, fnames, random_state)
    # add opposite target as feature
    if args.oppo_target:
        train_data, test_data = oppo_target_as_feature(train_data, test_data, train_index, test_index, oppo_target, oppo_target_name)
    # run SMOTE oversampling
    if args.sampling == "smote":
        featureSubset, featureSubset_labels = SMOTE_upsample(train_data)
    # run regular oversampling
    elif args.sampling == "regular_oversampling": 
        featureSubset, featureSubset_labels = random_upsample(train_data)
    # run regular undersampling
    elif args.sampling == 'regular_undersampling':
        featureSubset, featureSubset_labels = random_undersample(train_data)
    for modelType in modelTypelist:
        #select model
        if modelType == "SVC1":
            clf = svm.SVC(kernel='rbf', random_state=random_state)
        elif modelType == "SVC2":
            clf = svm.SVC(kernel='linear', random_state=random_state)
        elif modelType == "RF":
            #clf = RandomForestClassifier(criterion="gini", max_depth=3, random_state=r)
            clf = RandomForestClassifier(random_state=random_state)
        elif modelType == "kNN3":
            clf = KNeighborsClassifier(n_neighbors=3)
        elif modelType == "XG":
            clf = xgb.XGBClassifier(random_state=random_state)
        elif modelType == "LR":
            clf = LogisticRegression(max_iter=300, random_state=random_state)
        elif modelType == "NB":
            clf = GaussianNB()

        # convert balanced feature training set into dataframe
        train_dataFb = pd.DataFrame(featureSubset)
        # convert corresponding training labels into dataframe
        train_targetb = pd.DataFrame(featureSubset_labels)
        # remove label from test set
        test_dataF = test_data.iloc[:,:-1] #remove target 
        # convert test set to dataframe
        test_dataF_df = pd.DataFrame(test_dataF)

        #fit model and make predictions
        model = clf.fit(train_dataFb.to_numpy(), train_targetb.to_numpy()) #balanced training data
        result = model.predict(test_dataF_df.to_numpy()) #make predictions from testing data 

        #add to lists for df
        mlist.append(modelType)
        flist.append(num_f)
        pqtrain.append(train_index)
        pqtest.append(test_index)
        predictions.append(result.tolist())
        realvalues.append(test_label[0].to_list())
        
        #prepare evaluation results data
        if args.cross_validation == 'tts':
            if (modelType, num_f) not in results:
                results[(modelType, num_f)] = {"prediction": [], "true": [], "auc":[], "f1":[], "acc":[]}             
            else:
                results[(modelType, num_f)]["prediction"].append(result)
                results[(modelType, num_f)]["true"].append(test_label)

                results[(modelType, num_f)]["auc"].append(roc_auc_score(test_label, result))
                results[(modelType, num_f)]["f1"].append(f1_score(test_label, result))
                results[(modelType, num_f)]["acc"].append(accuracy_score(test_label, result))

        if args.cross_validation == 'loo':
            if (modelType, num_f) not in results:
                results[(modelType, num_f)] = {"prediction": [], "true": []}             
            
            results[(modelType, num_f)]["prediction"].append(result)
            results[(modelType, num_f)]["true"].append(test_data.target[0])
# calculate metrics based on true values and results
def calculate_metrics(true_values, prediction_values):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(prediction_values)): 
        if true_values[i]==prediction_values[i]==1:
           TP += 1
        if prediction_values[i]==1 and true_values[i]!=prediction_values[i]:
           FP += 1
        if true_values[i]==prediction_values[i]==0:
           TN += 1
        if prediction_values[i]==0 and true_values[i]!=prediction_values[i]:
           FN += 1
    return(TP, FP, TN, FN)
# import data and run program based on parameters
def main():
    randoms = [481998864, 689799321, 796360373, 325345551, 781053364, 425410490, 448592531, 477651899, 256556897, 950476446, 439161956, 617662138, 919221369, 372092462, 978558065, 915406186, 914758288, 270769509, 348581119, 620469471, 968622238, 493269528, 923889165, 187902000, 768516562, 656996274, 204570914, 478659400, 591118631, 455578751, 523453979, 904238395, 870935338, 65160836, 469733522, 301035177, 843432976, 931667506, 283989232, 77803117, 371210776, 231366353, 454473430, 335714437, 233937007, 131940380, 267081710, 208764677, 225578708, 684893704, 93911936, 333598779, 253843993, 390054067, 432395108, 730697387, 988951427, 963369310, 983748712, 206214635, 607442488, 783641874, 444298574, 799459448, 736269849, 222259535, 501043573, 914806112, 780691269, 993143254, 900823730, 946288505, 776711331, 393086523, 366784871, 181714875, 239540123, 101370413, 417433780, 288079126, 205915691, 73435964, 248074219, 582671864, 635043553, 338657949, 330517223, 804096498, 667716642, 995598949, 504427080, 778739823, 245211208, 96486247, 541502147, 5680657, 590309190, 5062322, 921199528, 188694207]
    random.seed(randoms[0])
    np.random.seed(randoms[0])
    # pd.random.seed(randoms[0])
    random_state = randoms[0]

    # "./open_smile_features_closed_with_phq_gad.csv"
    # "feature_extraction_audio/openSMILE/open_smile_features_closed_with_phq_gad.csv"

    df = pd.read_csv(args.data)
    cnames = df.columns
    fnames = cnames[1:-3]

    df0 = pd.DataFrame()
    for c in cnames:
        df0[c] = df[c].fillna(0)

    data = df0
    # featureDF = []

    
    if args.target_type == 'phq':
        oppo_target_name = 'gad-f'
        oppo_target = data[data.columns[-1]]
    elif args.target_type == 'gad':
        oppo_target_name = 'phq-f'
        oppo_target = data[data.columns[-2]]
    else:
        oppo_target_name = 'n/a'
        oppo_target = data[data.columns[-3]]

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

    # Feature Selection
    if args.feature_selection == "pca":
        #range of pca features to train and test
        nFeatureList = list(np.arange(1,11))
    elif args.feature_selection == 'etc':
        #range of etc features to train and test
        nFeatureList = list(np.arange(1,11))
    elif args.feature_selection == 'chi2':
        # range of chi2 features to train and test
        nFeatureList = list(np.arange(1,11))
    else: assert False, "feature selection argument unexpected"

    for index_f, num_f in enumerate(nFeatureList):
        #create feature array of this iteration's number of features    
        fDF = featureSubsetS
        fDF = fDF.to_numpy()    
        # cross validation
        # leave one out
        if args.cross_validation == "loo":
            loo = LeaveOneOut()
            loo.get_n_splits(fDF)
            temp_num=1
            for train_index, test_index in loo.split(fDF):
                # print("train:", train_index, "test:", test_index)
                train_data, test_data = fDF[train_index], fDF[test_index]

                train_index = train_index.tolist()
                test_index = test_index.tolist()
                train_data = pd.DataFrame(data = train_data)
                test_data = pd.DataFrame(data = test_data)
                train_label = pd.DataFrame(data = [target[i] for i in train_index]).reset_index(drop=True)
                test_label = pd.DataFrame(data = [target[i] for i in test_index]).reset_index(drop=True)
                run_models(train_data, train_label, test_data, test_label, train_index, test_index, num_f, cnames, fnames, random_state, oppo_target, oppo_target_name, temp_num)
                temp_num += 1
        # train test split
        elif args.cross_validation == 'tts':
            for r in randoms:
                random.seed(r)
                indices = np.arange(len(target))
                train_data, test_data, train_label, test_label, train_index, test_index = train_test_split(fDF, target, indices, test_size=0.3, shuffle = True, random_state = random_state)
                
                train_index = list(train_index)
                test_index = list(test_index)
                train_data = pd.DataFrame(data = train_data)
                test_data = pd.DataFrame(data = test_data)
                train_label = pd.DataFrame(data = train_label)
                test_label = pd.DataFrame(data = test_label)

                for model in modelTypelist:
                    seedlist.append(r)
                run_models(train_data, train_label, test_data, test_label, train_index, test_index, num_f, cnames, fnames, r, oppo_target, oppo_target_name)

    if args.feature_selection == 'chi2' or args.feature_selection == 'etc':
        with open('machine_learning/results/features_' + args.modality + '_' + args.feature_selection + '.json', 'w') as jf:
            json.dump(features_list, jf, indent=2)
    

    #make results summary df
    newDF2 = pd.DataFrame()
    newDF2["model"] = mlist
    newDF2["num_features"] = flist
    newDF2["train_index"] = pqtrain
    newDF2["test_index"] = pqtest
    newDF2["predicted_value"] = predictions
    newDF2["real_value"] = realvalues
    newDF2.insert(0, 'modality', args.modality)
    newDF2.insert(1, 'target_type', args.target_type)
    newDF2.insert(2, 'split', args.split)
    newDF2.insert(3, 'feature_selection', args.feature_selection)
    newDF2.insert(4, 'cross_validation', args.cross_validation)
    newDF2.insert(5, 'sampling', args.sampling)
    if args.cross_validation == 'tts':
        newDF2.insert(6, 'random_seed', seedlist)

    #make evaluation df
    if args.cross_validation == 'tts':
        eval_list = []
        for key, val_dict in results.items():
            model,f = key[0],key[1]
            auc = np.mean(np.array(val_dict["auc"]))
            f1 = np.mean(np.array(val_dict["f1"]))
            acc = np.mean(np.array(val_dict["acc"]))
            eval_list.append(
                {"modality": args.modality,
                "target_type":args.target_type,
                "split": args.split,
                "feature_selection": args.feature_selection,
                "cross_validation": args.cross_validation,
                "sampling": args.sampling,
                "model": model,
                "num_features": f,
                "auc": auc,
                "f1": f1,
                "accuracy": acc})
        evaluate_df = pd.DataFrame(eval_list)

    if args.cross_validation == 'loo':
        eval_list = []
        for key, val_dict in results.items():
            model,f = key[0],key[1]
            auc = roc_auc_score(val_dict["true"], val_dict["prediction"])
            f1 = f1_score(val_dict["true"], val_dict["prediction"])
            acc = accuracy_score(val_dict["true"], val_dict["prediction"])
            # print(val_dict["true"], val_dict["prediction"])
            TP, FP, TN, FN = calculate_metrics(val_dict['true'], val_dict['prediction'])
            if TP == 0:
                precision = 0
                sensitivity = 0
            else:
                precision = TP / (TP + FP)
                sensitivity = TP / (TP + FN) # recall
            if TN == 0:
                specificity = 0
            else:
                specificity = TN / (TN + FP)
            # print(TP, FP, TN, FN)
            eval_list.append(
                {"modality": args.modality,
                "target_type": args.target_type,
                "split": args.split,
                "feature_selection": args.feature_selection,
                "cross_validation": args.cross_validation,
                "sampling": args.sampling,
                "model": model,
                "num_features": f,
                "auc": auc,
                "f1": f1,
                "accuracy": acc,
                "precision": precision,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "true_positive": TP,
                "false_positive": FP,
                "true_negative": TN,
                "false_negative": FN})
        evaluate_df = pd.DataFrame(eval_list)

    #save dfs to csv
    newDF2.to_csv("machine_learning/results/" + args.modality + "_" + args.target_type + "_" + str(args.split) + "_" + args.feature_selection + "_" + args.cross_validation + "_" + args.sampling + ".csv", index = False)
    evaluate_df.to_csv("machine_learning/results/" + args.modality + "_" + args.target_type + "_" + str(args.split) + "_" + args.feature_selection + "_" + args.cross_validation + "_" + args.sampling + "_evaluate.csv", index = False)
# initialize arguments and constants
if __name__ == "__main__":
    args = parser().parse_args()

    modelTypelist = ["NB", "LR", "SVC1", "SVC2", "XG", "kNN3", "RF"]
    # modelTypelist = ["kNN3"]

    #parameters
    mlist = []
    flist = []
    #indexes in train and test sets
    pqtrain = []
    pqtest = []
    seedlist = []
    #results
    predictions = []
    realvalues = []
    results = {}
    #metrics
    f1List = []
    accList = []
    aucList = []
    #selected features
    features_list = []
    pca_weights = pd.DataFrame()

    main()