import pandas as pd
import os
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
import numpy as np
from catboost import CatBoostClassifier
import xgboost as xgb
import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.ensemble import GradientBoostingClassifier

logging.basicConfig(level=logging.INFO)

def read_csv(file_name):
    df = pd.read_csv(file_name, sep=",", header=None)
    return df


def read_csv_files(directory_path):
    dataframes = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):

            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path, header=None)
            df_name = os.path.splitext(filename)[0].rsplit('_', 1)[0]

            dataframes[df_name] = df

    return dataframes


def split_dataframe_and_label(df):

    mid_index = len(df) // 2
    upper_half = df.iloc[:mid_index].copy()
    lower_half = df.iloc[mid_index:].copy()

    upper_half[df.shape[1]] = 1
    lower_half[df.shape[1]] = 0

    return pd.concat([upper_half, lower_half])


def prepare_for_training(df):
    df_for_training = {}
    for key, value in df.items():
        df_split = split_dataframe_and_label(df[key])
        df_for_training[key] = df_split
    return df_for_training


def cross_validate_model(df, classifier, random_state=42):

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]


    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    fold_scores = {
        'ACC': [],
        'F1': [],
        'MCC': [],
        'AUROC': [],
        'AUPRC': [],
    }

    kde_plot = {
        'positive': [],
        'negative': [],
    }

    for fold_index, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)
        y_pred_proba = classifier.predict_proba(X_test)[:, 1]



        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        y_pred_proba_negative = classifier.predict_proba(X_test)[:, 0]

        if fold_index == 1:
            kde_plot['positive'] = y_pred_proba[y_test == 1]
            kde_plot['negative'] = y_pred_proba[y_test == 0]


        fold_scores['ACC'].append(round(accuracy_score(y_test, y_pred),4))
        fold_scores['F1'].append(round(f1_score(y_test, y_pred),4))
        fold_scores['MCC'].append(round(matthews_corrcoef(y_test, y_pred),4))
        fold_scores['AUROC'].append(round(roc_auc_score(y_test, y_pred_proba),4))

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        fold_scores['AUPRC'].append(round(auc(recall, precision),4))


    # print(fold_scores)

    return fold_scores

def training_test_data(df, clf):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    clf.fit(X_train, y_train)

    y_score = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_score)

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    curve_array = {'fpr':fpr ,'tpr':tpr,'precison': precision, 'recall':recall}

    return curve_array



Classifiers = {
'bioDGW-CMI': lgb.LGBMClassifier(n_estimators=53, max_depth=3, learning_rate=0.0403, verbosity=-1),
'JSNDCMI': GradientBoostingClassifier(n_estimators=35, learning_rate=0.0403, max_depth=2, random_state=42, verbose = 0),
'KS-CMI': CatBoostClassifier(iterations=45, learning_rate=1, depth=1, loss_function='Logloss', logging_level='Silent'),
'BCMCMI': xgb.XGBClassifier(n_estimators=60, learning_rate=0.0403, max_depth=2, use_label_encoder=False, eval_metric='mlogloss'),
'DeepCMI': xgb.XGBClassifier(n_estimators=55, learning_rate=0.0403, max_depth=2, use_label_encoder=False, eval_metric='mlogloss')
}


current_directory = os.path.dirname(os.path.realpath(__file__))
directory_9905 = os.path.join(current_directory, '9905')
directory_9589 = os.path.join(current_directory, '9589')

root_directory = os.path.dirname(current_directory)

df_9589 = read_csv_files(directory_9589)
df_9905 = read_csv_files(directory_9905)
df_prepared = prepare_for_training(df_9905)

results = {}
auc_and_aupr_curve = {}


selected_key =['JSNDCMI', 'KS-CMI', 'BCMCMI', 'DeepCMI', 'bioDGW-CMI']
for key, value in df_prepared.items():
    if key in selected_key:
        logging.info(f'processing {key}, the classifier: {Classifiers[key].__class__.__name__}')
        results[key] = cross_validate_model(df_prepared[key], Classifiers[key], random_state=42)
        mean_results = {k: round(np.mean(v),4) for k, v in results[key].items()}
        print(mean_results)
        auc_and_aupr_curve[key] = training_test_data(df_prepared[key], Classifiers[key])
# print(results)



benchmarking_boxplot_9589_path = os.path.join(root_directory, 'results', 'benchmarking_boxplot_9589.pkl')
benchmarking_curves_9589_path = os.path.join(root_directory, 'results','benchmarking_curves_9589.pkl')

benchmarking_boxplot_9905_path = os.path.join(root_directory, 'results','benchmarking_boxplot_9905.pkl')
benchmarking_curves_9905_path = os.path.join(root_directory, 'results','benchmarking_curves_9905.pkl')




def write_and_save_pickle(directory, data):
    try:
        with open(directory, 'wb') as file:
            pickle.dump(data, file)
    except Exception as e:
        print({e})

write_and_save_pickle(benchmarking_boxplot_9589_path, results)
write_and_save_pickle(benchmarking_curves_9589_path, auc_and_aupr_curve)

write_and_save_pickle(benchmarking_boxplot_9905_path, results)
write_and_save_pickle(benchmarking_curves_9905_path, auc_and_aupr_curve)




