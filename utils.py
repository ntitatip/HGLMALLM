import pandas as pd
import networkx as nx
import numpy as np
# import community as community_louvain
import matplotlib.pyplot as plt
import torch
# from torch_geometric.data import Data
from torch import nn
import torch.nn.functional as F
# from torch_geometric.nn import GATConv
# from torch_geometric.utils import negative_sampling
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import add_self_loops

# from sklearn.metrics import roc_auc_score
# from torch_geometric.transforms import RandomLinkSplit
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
import os

def load_pickle(pickle_file_path):
    try:
        df = pd.read_pickle(pickle_file_path)
        return df
    except Exception as e:
        print(f"Failed to load pickle file: {e}")
        return None

def read_csv(file_path):
    try:
        df = pd.read_csv(file_path, sep=",", header=None)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def generate_graph(circRNA_path, miRNA_path, pair_path):
    circRNA_df = load_pickle(circRNA_path)
    miRNA_df = load_pickle(miRNA_path)
    pair_df = read_csv(pair_path)

    circRNA_df['attribute'] = circRNA_df.iloc[:, 1:].values.tolist()

    circRNA_df = circRNA_df.iloc[:, [0, -1]]
    miRNA_df['attribute'] = miRNA_df.iloc[:, 1:].values.tolist()
    miRNA_df = miRNA_df.iloc[:, [0, -1]]

    circRNA_miRNA_df = pd.concat([circRNA_df, miRNA_df], axis=0)
    circRNA_miRNA_df = circRNA_miRNA_df.reset_index(drop=True)


    node_to_index = {node: index for index, node in enumerate(circRNA_miRNA_df.iloc[:, 0])}
    pair_df['node1'] = pair_df[0].map(node_to_index)
    pair_df['node2'] = pair_df[1].map(node_to_index)
    index_attribute_df = circRNA_miRNA_df.copy()
    index_attribute_df['idx'] = index_attribute_df.iloc[:, 0].map(node_to_index)



    index_attribute_df = index_attribute_df.iloc[:, [1]]  # 取决于属性实际所在的列


    index_attribute_df.columns = ['attribute']

    index_attribute_df_dict = index_attribute_df['attribute'].to_dict()
    formatted_dict = {key: {'sequence feature': value} for key, value in index_attribute_df_dict.items()}

    G = nx.Graph()
    for _, row in pair_df.iterrows():
        G.add_edge(row['node1'], row['node2'])


    G.add_nodes_from(formatted_dict.keys())

    nx.set_node_attributes(G, formatted_dict)

    return G




circRNA_path = 'merged_9905_circRNA_3.pkl'
miRNA_path = 'merged_9905_miRNA_3.pkl'
pair_path = '9905_pair.csv'
G = generate_graph(circRNA_path, miRNA_path, pair_path)
sub = None

if nx.is_connected(G):
    print("The graph is connected.")
else:
    print("The graph is not connected. Using the largest connected component.")
    largest_cc = max(nx.connected_components(G), key=len)
    sub = G.subgraph(largest_cc).copy()

# 确保有多样性的节点属性
# for node in G.nodes():
#     G.nodes[node]['attribute'] = node % 2  # 示例属性

# 检查节点度分布
# degrees = [d for _, d in G.degree()]
# if len(set(degrees)) > 1:
#     degree_assortativity = nx.degree_assortativity_coefficient(G)
#     print("Degree Assortativity Coefficient:", degree_assortativity)
# else:
#     print("All nodes have the same degree. Degree assortativity cannot be computed.")

# for node in G.nodes(data=True):
#     print(node[1])
features = np.array([data['sequence feature'] for _, data in sub.nodes(data=True)])
pca = PCA(n_components=1)
reduced_features = pca.fit_transform(features).flatten()

bins = np.linspace(np.min(reduced_features), np.max(reduced_features), num=7)  # 分成5个区间
discrete_features = np.digitize(reduced_features, bins)


# 将离散化后的值作为节点属性
new_attributes = {node: {"new_attr": discrete_features[i]} for i, node in enumerate(sub.nodes())}

# 使用 set_node_attributes 添加新的属性
nx.set_node_attributes(sub, new_attributes)

    # 设置节点属性



# 将降维后的标量值作为节点属性

attribute_assortativity = nx.attribute_assortativity_coefficient(sub, 'new_attr')
print("Attribute Assortativity Coefficient:", attribute_assortativity)

H = G.subgraph(largest_cc).copy()
np.random.seed(42)  # 确保随机结果可重复
edges = list(H.edges())
sample_size = max(1, int(0.01 * len(edges)))  # 至少保留一条边
sampled_edges = np.random.choice(len(edges), sample_size, replace=False)
sampled_edges = [edges[i] for i in sampled_edges]

# 创建一个新图只包含选中的边
H_sampled = nx.Graph()
H_sampled.add_edges_from(sampled_edges)


color_map = []
shape_map = []
for node, data in sub.nodes(data=True):
        node_label = data['new_attr'] # 因为节点编号从1开始，索引从0开始
        if node_label == 0:
            color_map.append('#FA599A')
            shape_map.append('o')  # 圆形节点
        elif node_label == 1:
            color_map.append('#F4952C')
            shape_map.append('s')  # 方形节点
        elif node_label == 2:
            color_map.append('#D9EFF5')
            shape_map.append('^')  # 三角形节点
        elif node_label == 3:
            color_map.append('#9E1E20')
            shape_map.append('d')  # 菱形节点
        elif node_label == 4:
            color_map.append('#416EB5')
            shape_map.append('v')  # 倒三角形节点
        elif node_label == 5:
            color_map.append('#737373')
            shape_map.append('p')  # 五边形节点

# 绘制图形
pos = nx.circular_layout(H)
plt.figure(figsize=(10, 10))
# 绘制不同形状的节点
unique_shapes = set(shape_map)
for shape in unique_shapes:
    nx.draw_networkx_nodes(H_sampled, pos, node_color=[color_map[i] for i in range(len(H_sampled)) if shape_map[i] == shape],
                           node_shape=shape, nodelist=[list(H_sampled.nodes)[i] for i in range(len(H_sampled)) if shape_map[i] == shape],
                           node_size=50)

# 绘制边
nx.draw_networkx_edges(H_sampled, pos)

# 绘制节点标签
# nx.draw_networkx_labels(H, pos)

legend_elements = [Line2D([0], [0], marker='o', color='#FA599A', linestyle='None', markersize=10, label='Sub 1'),
                   Line2D([0], [0], marker='s', color='#F4952C', linestyle='None', markersize=10, label='Sub 2'),
                   Line2D([0], [0], marker='^', color='#D9EFF5', linestyle='None', markersize=10, label='Sub 3'),
                    Line2D([0], [0], marker='d', color='#9E1E20', linestyle='None', markersize=10, label='Sub 4'),
                    Line2D([0], [0], marker='v', color='#416EB5', linestyle='None', markersize=10, label='Sub 5'),
                    Line2D([0], [0], marker='p', color='#737373', linestyle='None', markersize=10, label='Sub 6')]


plt.legend(handles=legend_elements, loc='upper right')

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(root_directory)
file_path = os.path.join(root_directory,'BERT-DGI', f'9905-non' + '.tif')
plt.savefig(file_path, format='tiff', dpi=350)
plt.show()


# def generate_pyg():
#     G = generate_graph(circRNA_path, miRNA_path, pair_path)

#     global_betweenness_centrality = nx.betweenness_centrality(G)
#     global_closeness_centrality = nx.closeness_centrality(G)

#     partition = community_louvain.best_partition(G)

#     community_nodes = {}

#     for node, comm_id in partition.items():
#         if comm_id not in community_nodes:
#             community_nodes[comm_id] = []
#         community_nodes[comm_id].append(node)

#     community_subgraphs = {comm_id: G.subgraph(nodes) for comm_id, nodes in community_nodes.items()}

#     community_betweenness = {}
#     community_closeness = {}
#     for comm_id, subgraph in community_subgraphs.items():
#         betweenness_centrality = nx.betweenness_centrality(subgraph)
#         closeness_centrality = nx.closeness_centrality(subgraph)
#         community_betweenness[comm_id] = betweenness_centrality
#         community_closeness[comm_id] = closeness_centrality

#     community_betweenness_new = {}
#     community_closeness_new = {}

#     for sub_dict in community_betweenness.values():
#         community_betweenness_new.update(sub_dict)

#     for sub_dict in community_closeness.values():
#         community_closeness_new.update(sub_dict)

#     for u, v, d in G.edges(data=True):
#         global_betweenness_avg = (global_betweenness_centrality[u] + global_betweenness_centrality[v]) / 2
#         global_closeness_avg = (global_closeness_centrality[u] + global_closeness_centrality[v]) / 2

#         local_betweenness_avg = (community_betweenness_new[u] + community_betweenness_new[v]) / 2
#         local_closeness_avg = (community_closeness_new[u] + community_closeness_new[v]) / 2
        
#         edge_features = torch.tensor([
#             global_betweenness_avg, global_closeness_avg, 
#             local_betweenness_avg, local_closeness_avg
#         ])
        
#         d['edge_features'] = edge_features

#     node_features = []
#     for node in G.nodes(data=True):
#         node_features.append(node[1]['sequence feature'])  
    
#     node_features = torch.tensor(node_features, dtype=torch.float)


#     edge_index = []
#     edge_attr = []

#     for u, v, attr in G.edges(data=True):
#         edge_index.append([u, v])
#         edge_attr.append([attr['edge_features']])  
    

#     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

#     flat_list_of_tensors = [tensor for sublist in edge_attr for tensor in sublist]

#     stacked_tensor = torch.stack(flat_list_of_tensors, dim=0)


#     data = Data(x=node_features, edge_index=edge_index, edge_attr=stacked_tensor)

#     return data


# class ECCLayer(MessagePassing):
#     def __init__(self, node_in_channels=768, edge_in_channels=4, out_channels=768):
#         super(ECCLayer, self).__init__(aggr='mean')  # 'mean'：平均聚合方式
#         self.edge_mlp = nn.Linear(edge_in_channels, node_in_channels * out_channels)
#         self.node_mlp = nn.Linear(node_in_channels, out_channels)

#     def forward(self, x, edge_index, edge_attr):
#         edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0)

#         edge_weights = self.edge_mlp(edge_attr)
#         edge_weights = edge_weights.view(-1, x.size(1), self.node_mlp.out_features)
        
#         return self.propagate(edge_index, x=x, edge_weight=edge_weights)
    
#     def message(self, x_j, edge_weight):
#         return torch.matmul(x_j.unsqueeze(1), edge_weight).squeeze(1)
    
# class GATECCLinkPrediction(nn.Module):
#     def __init__(self):
#         super(GATECCLinkPrediction, self).__init__()
#         self.ecc_layer = ECCLayer()
#         self.gat_layer1 = GATConv(in_channels =768, out_channels=512, heads=8, concat=False, negative_slope=0.2, dropout=0.6)

#     def forward(self, x, edge_index, edge_attr):
#         x0 = F.relu(self.ecc_layer(x, edge_index, edge_attr))
#         x1 = F.relu(self.gat_layer1(x0, edge_index))
#         return x1
    
#     def decode(self, z, edge_label_index):
#         return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
    
#     def decode_all(self, z):
#         prob_adj = z @ z.t()
#         return (prob_adj > 0).nonzero(as_tuple=False).t()
   

# data = generate_pyg().to('cuda')  

# model = GATECCLinkPrediction().to('cuda')

# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.01)
# criterion = torch.nn.BCEWithLogitsLoss()





# transform = RandomLinkSplit(num_val=0.1, num_test=0.1, add_negative_train_samples=False)

# train_data, val_data, test_data = transform(data)


# def train():
#     model = GATECCLinkPrediction().to('cuda')
#     model.train()
#     optimizer.zero_grad()
#     z = model.forward(train_data.x, train_data.edge_index, train_data.edge_attr)
#     # We perform a new round of negative sampling for every training epoch:
#     neg_edge_index = negative_sampling(
#         edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
#         num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
#     edge_label_index = torch.cat(
#         [train_data.edge_label_index, neg_edge_index],
#         dim=-1,
#     )
#     edge_label = torch.cat([
#         train_data.edge_label,
#         train_data.edge_label.new_zeros(neg_edge_index.size(1))
#     ], dim=0)
#     out = model.decode(z, edge_label_index).view(-1)
#     loss = criterion(out, edge_label)
#     loss.backward()
#     optimizer.step()
#     del model
#     torch.cuda.empty_cache()
#     return loss

# @torch.no_grad()
# def test():
#     model = GATECCLinkPrediction().to('cuda')
#     model.eval()
#     z = model.forward(test_data.x, test_data.edge_index, test_data.edge_attr)
#     neg_edge_index = negative_sampling(
#         edge_index=test_data.edge_index, num_nodes=test_data.num_nodes,
#         num_neg_samples=test_data.edge_label_index.size(1), method='sparse')
#     edge_label_index = torch.cat(
#         [test_data.edge_label_index, neg_edge_index],
#         dim=-1,
#     )
#     edge_label = torch.cat([
#         test_data.edge_label,
#         test_data.edge_label.new_zeros(neg_edge_index.size(1))
#     ], dim=0)
#     out = model.decode(z,edge_label_index).view(-1).sigmoid()
#     del model
#     torch.cuda.empty_cache()
#     return roc_auc_score(edge_label.cpu().numpy(), out.cpu().numpy())

# @torch.no_grad()
# def val():
#     model = GATECCLinkPrediction().to('cuda')
#     model.eval()
#     z = model.forward(val_data.x, val_data.edge_index, val_data.edge_attr)
#     neg_edge_index = negative_sampling(
#         edge_index=val_data.edge_index, num_nodes=val_data.num_nodes,
#         num_neg_samples=val_data.edge_label_index.size(1), method='sparse')
#     edge_label_index = torch.cat(
#         [val_data.edge_label_index, neg_edge_index],
#         dim=-1,
#     )
#     edge_label = torch.cat([
#         val_data.edge_label,
#         val_data.edge_label.new_zeros(neg_edge_index.size(1))
#     ], dim=0)
#     out = model.decode(z,edge_label_index).view(-1).sigmoid()
#     del model
#     torch.cuda.empty_cache()
#     return roc_auc_score(edge_label.cpu().numpy(), out.cpu().numpy())

# # 训练模型
# best_val_auc = final_test_auc = 0
# for epoch in range(1000):
#     loss = train()
#     val_auc = test()
#     test_auc = test()
#     if val_auc > best_val_auc:
#         best_val_auc = val_auc
#         final_test_auc = test_auc
#     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
#           f'Test: {test_auc:.4f}')
# print(f'Final Test: {final_test_auc:.4f}')
# z = model.forward(test_data.x, test_data.edge_index, test_data.edge_attr)
# final_edge_index = model.decode_all(z)







