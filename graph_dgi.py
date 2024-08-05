import pandas as pd

import torch
from torch_geometric.data import Data

from torch_geometric.nn import GCNConv
from torch_geometric.nn import DeepGraphInfomax, GCNConv

from torch_geometric.utils import to_undirected
import torch.optim as optim
import itertools 
import os
import pickle
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
    circRNA_df.columns = [str(i) for i in range(len(circRNA_df.columns))]
    miRNA_df.columns = [str(i) for i in range(len(miRNA_df.columns))]
    index_df = pd.concat([circRNA_df['0'], miRNA_df['0']], ignore_index=True)
    circRNA_feature = circRNA_df.iloc[:, 1:]
    miRNA_feature = miRNA_df.iloc[:, 1:]

    circ_array = circRNA_feature.to_numpy()
    mi_array = miRNA_feature.to_numpy()


    circ_tensor = torch.tensor(circ_array)
    miRNA_tensor = torch.tensor(mi_array)
    feature_tensor = torch.cat((circ_tensor, miRNA_tensor), 0)

    pair_df = read_csv(pair_path)

    index_df = pd.concat([circRNA_df.iloc[:, 0], miRNA_df.iloc[:, 0]], axis=0)
    index_df = index_df.reset_index(drop=True)


    node_to_index = {node: index for index, node in enumerate(index_df.iloc[:])}
    pair_df['node1'] = pair_df[0].map(node_to_index)
    pair_df['node2'] = pair_df[1].map(node_to_index)


    pair_1 = pair_df['node1'].to_numpy()
    pair_2 = pair_df['node2'].to_numpy()

    tensor1 = torch.tensor(pair_1)
    tensor2 = torch.tensor(pair_2)

    stacked_tensor = torch.stack((tensor1, tensor2), dim=1)  # 沿第0维堆叠

    x_transposed = torch.transpose(stacked_tensor, 0, 1)
    edge_index_undirected = to_undirected(x_transposed)

    data = Data(x=feature_tensor, edge_index=edge_index_undirected)


    return index_df, data




circRNA_path = 'merged_9589_circRNA_3.pkl'
miRNA_path = 'merged_9589_miRNA_3.pkl'
pair_path = '9589_pair.csv'

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.prelu = torch.nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0), device=x.device)], edge_index


if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')







def train():
    model.train()
    optimizer.zero_grad()
    pos_z, neg_z, summary = model(data.x, data.edge_index)
    loss = model.loss(pos_z, neg_z, summary)
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    z, _, _ = model(data.x, data.edge_index)
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask], max_iter=150)
    return acc

def feature_extractor():
    model.eval()
    z, _, _ = model(data.x, data.edge_index)
    return z

# for epoch in range(1, 301):
#     loss = train()
#     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
# # acc = test()
# z = feature_extractor()
# print(f'z: {z.shape}')






base_dir = os.path.dirname(os.path.realpath(__file__))  # replace with your base directory
folder_list = ['9589', '9905']

training_file_list = []
for folder in folder_list:
    folder_path = os.path.join(base_dir, 'sequence_feature', f'{folder}_sequence_feature')
    lst = [3, 4, 5, 6]
    combinations = list(itertools.product(lst, repeat=2))
    for combination in combinations:
        var1, var2 = combination
        circRNA_path = os.path.join(folder_path, f'merged_{folder}_circRNA_{var1}'+'.pkl')
        miRNA_path = os.path.join(folder_path, f'merged_{folder}_miRNA_{var2}'+'.pkl')
        pair_path = os.path.join(base_dir, f'{folder}_pair'+'.csv')
        paths = [circRNA_path, miRNA_path, pair_path]
        training_file_list.append(paths)

for paths in training_file_list:
    index_df, data = generate_graph(*paths)
    for epoch in range(1, 301):
        model = DeepGraphInfomax(
        hidden_channels=512,
        encoder=Encoder(data.num_features, 512),
        summary=lambda z, *args, **kwargs: z.mean(dim=0).sigmoid(),
        corruption=corruption,
        ).to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = train()
    z = feature_extractor()
    rna_type_1 = os.path.basename(paths[0])
    rna_type_1 = os.path.splitext(rna_type_1)
    rna_type_1 = rna_type_1[0].split("_")[-2:]
    rna_type_1_ = "_".join(rna_type_1)
    rna_type_2 = os.path.basename(paths[1])
    rna_type_2 = os.path.splitext(rna_type_2)
    rna_type_2 = rna_type_2[0].split("_")[-2:]
    rna_type_2_ = "_".join(rna_type_2)
    pair_ = os.path.basename(paths[2])
    pair = pair_.split("_")[:1][0]
    results_path = os.path.join(base_dir, 'graph_feature', f'{rna_type_1_}+{rna_type_2_}' + f'_{pair}' + '.pkl')
    z_df = z.to('cpu')
    z_df = pd.DataFrame(z_df.detach().numpy())
    result = pd.concat([index_df, z_df], axis=1, ignore_index=True)
    print(result)
    with open(results_path, 'wb') as file:
        pickle.dump(result, file)
        print(f'Successfully saved {results_path}')



        





# with open(results_path, 'wb') as file:
#     pickle.dump(sequence_vector, file)




# Define the list
# lst = [3, 4, 5, 6]

# # Generate all combinations
# combinations = list(itertools.product(lst, repeat=2))

# # Now combinations is a list of tuples, where each tuple is a combination
# for combination in combinations:
#     var1, var2 = combination
#     print(var1, var2)



    #       f'Test: {test_auc:.4f}')
# print(f'Final Test: {final_test_auc:.4f}')








