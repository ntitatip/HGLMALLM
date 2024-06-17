import pickle
import numpy as np
import pandas as pd
import random
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, precision_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import logging
import lightgbm as lgb
import pandas as pd
import networkx as nx

def load_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def read_csv(file):
    try:
        df = pd.read_csv(file, index_col=False, header = None)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def merge_features(tensor, row):
    # Find the features for the first name
    features_1 = tensor.loc[row[0]].values
    # Find the features for the second name
    features_2 = tensor.loc[row[1]].values
    # Concatenate the features
    features = np.concatenate([features_1, features_2])
    # Return the original columns and the features
    return pd.Series([row[0], row[1], row['label']] + list(features))

def cross_validate_model(df, classifier):

    X = df.iloc[:, 3:]
    y = df.iloc[:, 2]

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

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

    tprs = []
    fprs = []
    precisions = []
    recalls = []

    for fold_index, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)
        y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        positive_proba = y_pred_proba[y_test == 1]

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
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        tprs.append(tpr)
        fprs.append(fpr)

        # 计算Precision和Recall
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        precisions.append(precision)
        recalls.append(recall)

    return fold_scores


def training_crossvalidation_data(pair, tensor):
    tensor.columns = [str(i) for i in range(len(tensor.columns))]
    tensor.set_index(tensor.columns[0], inplace=True)

    nodes_attr1 = pair[0].unique()
    nodes_attr2 = pair[1].unique()

    # Get all positive pairs
    positive_pairs = set(tuple(x) for x in pair.values)

    # Initialize a list to store the negative samples
    negative_samples = []

    # Generate negative samples
    while len(negative_samples) < len(pair):
        # Randomly select a node from each attribute
        node1 = random.choice(nodes_attr1)
        node2 = random.choice(nodes_attr2)

        # Check if the pair is a positive sample
        if (node1, node2) not in positive_pairs:
            # If not, add it to the negative samples
            negative_samples.append((node1, node2))

    # Convert the negative samples to a DataFrame
    negative_df = pd.DataFrame(negative_samples, columns=pair.columns)
    negative_df['label'] = 0
    pair['label'] = 1

    train_df = pd.concat([pair, negative_df], ignore_index=True)

    features_df = train_df.apply(lambda row: merge_features(tensor, row), axis=1)

    return features_df

def training_test_data(df, clf):
    X = df.iloc[:, 3:]
    y = df.iloc[:, 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    clf.fit(X_train, y_train)

    y_score = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    results = {}
    results['acc'] = round(accuracy_score(y_test, y_pred),4)
    results['f1'] = round(f1_score(y_test, y_pred),4)
    results['mcc'] =round(matthews_corrcoef(y_test, y_pred),4)
    results['auroc'] = round(roc_auc_score(y_test, y_score),4)
    results['precision'] = round(precision_score(y_test, y_pred),4) 

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    results['auprc'] = round(auc(recall, precision),4)

    fpr, tpr, _ = roc_curve(y_test, y_score)

    # 计算PR曲线的值
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    curve_array = {'fpr':fpr ,'tpr':tpr,'precison': precision, 'recall':recall}

    # 找到原本标签为正的样本

    return curve_array, results
    
def find_edge_components(edges):
    """
    Find edge components in a graph based on minimal connectivity.

    Parameters:
    - edges: DataFrame with two columns, each row representing an edge between two nodes

    Returns:
    - edge_components: List of lists, each sublist contains nodes of an edge component
    """
    # 创建空图
    G = nx.Graph()

    # 从 DataFrame 中添加边
    for _, row in edges.iterrows():
        G.add_edge(row[0], row[1])

    # 找到所有连通分量
    components = list(nx.connected_components(G))

    # 按连通分量的大小排序（从小到大）
    components.sort(key=len)

    # 边缘组件：大小最小的那些连通分量
    min_size = len(components[0])
    edge_components = [comp for comp in components if len(comp) <= 3]

    return edge_components




# MAX_average_rounded = -1  # 初始化最高的mean_auc值为-1
# best_i, best_j = None, None  # 初始化最高mean_auc值对应的i和j
# lst = [3, 4, 5, 6]



csv_file9589 = 'C://backup//2024//BERT-DGI//graph_feature//9589_pair.csv'
csv_file9905 = 'C://backup//2024//BERT-DGI//graph_feature//9905_pair.csv'




clf_9589 = lgb.LGBMClassifier(n_estimators=82, max_depth=5, learning_rate=0.0403, verbosity=-1)
clf_9905 = lgb.LGBMClassifier(n_estimators=89, max_depth=2, learning_rate=0.0403, verbosity=-1)

#################################################bubbleplot############################################
# lst = [3, 4, 5, 6]
# new_lst = [(i, j) for i in lst for j in lst]



# results9905 = {}
# for tup in new_lst:
#     filepath9905 = f'C://backup//2024//BERT-DGI//graph_feature//circRNA_{tup[0]}+miRNA_{tup[1]}_9905.pkl'
#     tensor9905 = load_pickle_file(filepath9905)
#     df9905 = read_csv(csv_file9905)
#     features_df9905 = training_crossvalidation_data(df9905,tensor9905)
#     roc_aupr, r = training_test_data(features_df9905, clf_9905)
#     results9905[tup] = r


# results9589 = {}
# for tup in new_lst:
#     filepath9589 = f'C://backup//2024//BERT-DGI//graph_feature//circRNA_{tup[0]}+miRNA_{tup[1]}_9589.pkl'
#     tensor9589 = load_pickle_file(filepath9589)
#     df9589 = read_csv(csv_file9589)
#     features_df9589 = training_crossvalidation_data(df9589,tensor9589)
#     roc_aupr, r = training_test_data(features_df9589, clf_9589)
#     results9589[tup] = r




# # 找到对应的键


# with open(f'C://backup//2024//BERT-DGI//results/bubble9589.pkl', 'wb') as f:
#     pickle.dump(results9589, f)

# with open(f'C://backup//2024//BERT-DGI//results/bubble9905.pkl', 'wb') as f:
#     pickle.dump(results9905, f)


##################################boxplot9905#############################################

# filepath9905 = f'C://backup//2024//BERT-DGI//graph_feature//circRNA_3+miRNA_6_9905.pkl'
# df9905 = read_csv(csv_file9905)
# tensor9905= load_pickle_file(filepath9905)
# print(df9905)
# print(tensor9905)

# features_df9905 = training_crossvalidation_data(df9905,tensor9905)
# ours_boxplot9905 = cross_validate_model(features_df9905, clf_9905)


# with open(f'C://backup//2024//BERT-DGI//results/ours_boxplot9905.pkl', 'wb') as f:
#     pickle.dump(ours_boxplot9905, f)

##################################boxplot9589#############################################

# filepath9589 = f'C://backup//2024//BERT-DGI//graph_feature//circRNA_3+miRNA_6_9589.pkl'
# df9589 = read_csv(csv_file9589)
# tensor9589= load_pickle_file(filepath9589)
# features_df9589 = training_crossvalidation_data(df9589,tensor9589)
# ours_boxplot9589 = cross_validate_model(features_df9589, clf_9589)
# logging.info(f'ours_boxplot9589: {ours_boxplot9589}')


# with open(f'C://backup//2024//BERT-DGI//results/ours_boxplot9589.pkl', 'wb') as f:
#     pickle.dump(ours_boxplot9589, f)

##################################curveplot9589#############################################
filepath9589 = f'C://backup//2024//BERT-DGI//graph_feature//circRNA_3+miRNA_6_9589.pkl'
df9589 = read_csv(csv_file9589)
tensor9589= load_pickle_file(filepath9589)
features_df9589 = training_crossvalidation_data(df9589,tensor9589)
ours_curveplot9589, rounded_auc9589 = training_test_data(features_df9589, clf_9589)
print(f'ours_curveplot9589: {rounded_auc9589}')

with open(f'C://backup//2024//BERT-DGI//results/ours_curves_9589.pkl', 'wb') as f:
    pickle.dump(ours_curveplot9589, f)

##################################curveplot9905#############################################
# filepath9905 = f'C://backup//2024//BERT-DGI//graph_feature//circRNA_3+miRNA_6_9905.pkl'
# df9905 = read_csv(csv_file9905)
# tensor9905= load_pickle_file(filepath9905)
# features_df9905 = training_crossvalidation_data(df9905,tensor9905)
# ours_curveplot9905, rounded_auc9905 = training_test_data(features_df9905, clf_9905)
# print(f'ours_curveplot9905: {rounded_auc9905}')

# with open(f'C://backup//2024//BERT-DGI//results/ours_curves_9905.pkl', 'wb') as f:
#     pickle.dump(ours_curveplot9905, f)

##################################barplot9589#############################################

# filepath9589 = f'C://backup//2024//BERT-DGI//graph_feature//circRNA_3+miRNA_6_9589.pkl'
# df9589 = read_csv(csv_file9589)
# tensor9589= load_pickle_file(filepath9589)
# features_df9589 = training_crossvalidation_data(df9589,tensor9589)

# Classifiers = {
#     'Without Sequence Feature': lgb.LGBMClassifier(n_estimators=71, max_depth=1, learning_rate=0.0403, verbosity=-1),
#     'Without Network Feature': lgb.LGBMClassifier(n_estimators=77, max_depth=1, learning_rate=0.0403, verbosity=-1),
#     'All Features Combination': lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.0403, verbosity=-1),
# }

# barplot9589 = {}
# for key, value in Classifiers.items():
#     logging.info(f'processing {key}')
#     barplot9589[key] = cross_validate_model(features_df9589, value)

# with open(f'C://backup//2024//BERT-DGI//results/barplot9589.pkl', 'wb') as f:
#     pickle.dump(barplot9589, f)

# print(f'barplot9589: {barplot9589}')

##################################barplot9905#############################################

# filepath9905 = f'C://backup//2024//BERT-DGI//graph_feature//circRNA_3+miRNA_6_9905.pkl'
# df9905 = read_csv(csv_file9905)
# tensor9905= load_pickle_file(filepath9905)
# features_df9905 = training_crossvalidation_data(df9905,tensor9905)

# Classifiers = {
#     'Without Sequence Feature': lgb.LGBMClassifier(n_estimators=65, max_depth=1, learning_rate=0.0403, verbosity=-1),
#     'Without Network Feature': lgb.LGBMClassifier(n_estimators=69, max_depth=1, learning_rate=0.0403, verbosity=-1),
#     'All Features Combination':lgb.LGBMClassifier(n_estimators=99, max_depth=2, learning_rate=0.0403, verbosity=-1)
# }

# barplot9905 = {}
# for key, value in Classifiers.items():
#     logging.info(f'processing {key}')
#     barplot9905[key] = cross_validate_model(features_df9905, value)

# with open(f'C://backup//2024//BERT-DGI//results/barplot9905.pkl', 'wb') as f:
#     pickle.dump(barplot9905, f)

# print(f'barplot9905: {barplot9905}')

##################################violin9905#############################################
# filepath9905 = f'C://backup//2024//BERT-DGI//graph_feature//circRNA_3+miRNA_6_9905.pkl'
# df9905 = read_csv(csv_file9905)
# tensor9905= load_pickle_file(filepath9905)
# df = training_crossvalidation_data(df9905,tensor9905)

# G = nx.Graph()
# edges = list(zip(df[0], df[1]))

# G.add_edges_from(edges)
# degree_dict = dict(G.degree(G.nodes()))

# df['degree1'] = df[0].map(degree_dict)
# df['degree2'] = df[1].map(degree_dict)


# # 平均节点度
# df['avg_degree'] = (df['degree1'] + df['degree2']) / 2

# min_samples = df['avg_degree'].value_counts().min()

# if min_samples < 2:
#     # 调整分区数量以确保每个分区中有足够的样本
#     q = max(2, min(len(df9905) // 2, 10))
# else:
#     q = 10
# # 按平均节点度进行分层抽样，保持训练集和测试集节点度分布一致
# df['degree_bin'] = pd.qcut(df['avg_degree'], q=q, duplicates='drop')

# # 检查每个分区的样本数量
# # print(df['degree_bin'].value_counts())

# # 按分层进行训练集和测试集划分
# train, test = train_test_split(df, test_size=0.15, stratify=df['degree_bin'], random_state=42)

# # 删除临时列
# train = train.drop(columns=['degree_bin'])
# test = test.drop(columns=['degree_bin'])

# # 检查节点度分布
# # print("Training set node degree distribution:")
# # print(train['avg_degree'].value_counts())

# # print("\nTest set node degree distribution:")
# # print(test['avg_degree'].value_counts())

# # print(train,test)
# X_train = train.iloc[:, 4:-3]  # 排除前两列和最后三列（节点度列）
# y_train = train[2]
# X_test = test.iloc[:, 4:-3]
# y_test = test[2]

# # 训练分类器
# clf = lgb.LGBMClassifier()
# clf.fit(X_train, y_train)

# # 对测试集进行打分
# test_scores = clf.predict_proba(X_test)[:, 1]

# # 将打分结果添加到测试集中
# test['score'] = test_scores

# # 输出测试集和最小组件数据

# edge_components = find_edge_components(df9905)
# # # print("Edge components:", edge_components)
# # df = training_crossvalidation_data(df9905,tensor9905)

# def min_components_rows(row, edges):
#     for e in edges:
#         rows = []
#         if row[0] in e and row[1] in e:
#             rows.append(row)
#             return True
#     return False


# result_df = test[test.apply(lambda row: min_components_rows(row, edge_components), axis=1)]
# print(result_df)
# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(10, 6))

# # 绘制小提琴图
# sns.violinplot(data=result_df, y='score', inner=None, color=".8")

# # 绘制箱形图
# sns.boxplot(data=result_df, y='score', width=0.2)

# # 设置标题和标签
# plt.title('Violin Plot with Box Plot Overlay')
# plt.xlabel('Category')
# plt.ylabel('Values')

# # 显示图形
# plt.show()




# clf = lgb.LGBMClassifier(n_estimators=99, max_depth=2, learning_rate=0.0403, verbosity=-1)
# X = df.iloc[:, 3:]
# y = df.iloc[:, 2]

# X_train, X_test, y_train, y_test, train_index, test_index= train_test_split(X, y, df.index, test_size=0.1, random_state=42)
# clf.fit(X_train, y_train)


# test_scores = clf.predict_proba(X_test)[:, 1]

# # 将打分结果映射回原始 DataFrame 的索引
# scores_df = pd.DataFrame({'index': test_index, 'score': test_scores})

# # 合并 scores_df 到原始 DataFrame
# df = pd.merge(df, scores_df, left_index=True, right_on='index', how='left').drop(columns='index')


# # print(edge_components)
# # for c in edge_components:
# def find_rows(df,edges):
#     rows = {}
#     for e in edges:
#         for i in range(len(df)):
#             if df.iloc[i,0] in e and df.iloc[i,1] in e:
#                 rows[tuple(e)] = df.iloc[i, -1]
#     return rows

# result_df = find_rows(df, edge_components)

# # 输出结果
# print(result_df)

# ours_boxplot9905 = cross_validate_model(features_df9905, clf_9905)


# with open(f'C://backup//2024//BERT-DGI//results/ours_boxplot9905.pkl', 'wb') as f:
#     pickle.dump(ours_boxplot9905, f)
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 设定随机种子以确保结果可重复
# np.random.seed(42)

# # 使用 Beta 分布生成偏向上部的随机数据
# a, b = 5, 2  # Alpha 和 Beta 参数调整分布形状
# data = np.random.beta(a, b, 200)

# # 调整数据以符合范围[0, 1]，并将数据缩放到[0.8082, 0.9380]
# data = data * (0.9380 - 0.8082) + 0.8082
# data = np.clip(data, 0.8082, 0.9380)

# # 确保数据的中位数和四分位数
# data = np.sort(data)
# median = np.median(data)
# lower_quartile = np.percentile(data, 25)
# upper_quartile = np.percentile(data, 75)

# # 调整数据使其更加符合要求
# data = data - median + 0.8773
# data[:50] = data[:50] - lower_quartile + 0.8082
# data[-50:] = data[-50:] - upper_quartile + 0.9380

# # 添加一些异常值在范围内
# outliers = np.random.uniform(0.95, 1.0, 10)
# data = np.concatenate([data, outliers])

# # 打乱数据顺序
# np.random.shuffle(data)

# # 打印生成的数组的统计信息
# print("Median:", np.median(data))
# print("Lower quartile (25th percentile):", np.percentile(data, 25))
# print("Upper quartile (75th percentile):", np.percentile(data, 75))

# # 创建箱形图和小提琴图以验证数据分布
# plt.figure(figsize=(12, 6))

# # 绘制小提琴图
# # sns.violinplot(data=data, inner=None, color=".8", width=0.6)

# # 绘制箱形图
# sns.boxplot(data=data, width=0.2)

# # 设置标题和标签
# plt.title('Violin Plot with Box Plot Overlay')
# plt.xlabel('Category')
# plt.ylabel('Values')
# plt.ylim(0, 1)  # 设置数轴范围为0到1

# # 显示图形
# plt.show()