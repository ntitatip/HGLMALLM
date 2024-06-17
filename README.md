# HGLMALLM
===========================================================================


[![license](https://img.shields.io/badge/python_-3.8.0_-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/torch_-1.12.0_-blue)](https://pytorch.org/)


Identifying circRNA-microRNA interactions (CMI) is a significant biomedical issue in recent years. This problem provides insights into using circRNA as biomarkers, developing cancer therapies, and creating cancer vaccines. Using computational methods for identification and prediction is a more time-efficient and cost-effective approach. In computational methods, using graphs to represent and explore the CMI is a mainstreamapproach. However, existing relevant methods do not achieve optimal results by utilizing both the semantic information extracted from sequences and the topological information extracted from graph structures. To address this issue, we propose HGLMALLM, a graph contrastive learning method that learns node representation crossing both the semantic domain generated via motif-aware pre-trained LLMs and the topological domain extracted from hierarchical graph structures. Our method effectively addresses the issue in existing Message Passing Neural Network (MPNN) method that edge components losing heterogeneity after multiple iterations. Moreover, this method utilize the heterogeneity of graph which is extended from the traditional bipartite graph to heterogeneous through the semantic domain. Two commonly used datasets were partitioned based on the distribution of node degrees. Then, we benchmarked our method against 
existing methods. In the independent test set evaluation, it achieved a 3% amd 1% improvement on two datasets. Our method demonstrated the best stability in ten-fold cross-validation on the training set. The edge components with fewer than four nodes were separated and conducted tests to validate that our method performs well. A datasets collected from real scenarios was used to demonstrate the strong predictive ability of our method for identifying unidentified CMI.


![Image text](Plot/framework.png)(Fig. 1.)

## Table of Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Contributing](#contributing)
- [Cite](#cite)
- [Contacts](#contacts)
- [License](#license)


## Installation

Our method is tested to work under:

```
* Python 3.8.0
* Torch 1.12.0
* Numpy 1.23.5
* Other basic python toolkits
```
### Installation of other dependencies
* Install [Networkx](https://networkx.github.io/) using ` $ pip install networkx `
* Install [PyG](https://pypi.org/project/torch-geometric/) using ` $ pip install torch-geometric `
* Install [lightGBM](https://lightgbm.readthedocs.io/en/stable/) run the following commands in cmd:
```
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build
cd build
cmake -A x64 ..
cmake --build . --target ALL_BUILD --config Release
```
* Install [scikit-learn]https://scikit-learn.org/stable/) using '$ pip install -U scikit-learn'

# Quick start
To reproduce our results:

## 1，Extract sequence features via RNA-Ernie
```
We used RNA-ERNIE as our semantic descriptor, which they have packaged into a Docker container for ease of use. Please find it at the following link:

[RNA-Ernie](https://www.nature.com/articles/s42256-024-00836-4)
```
**Arguments**:
| **Arguments** | **Detail** |
| --- | --- |
| **k_lst** | The required k-value to be selected |
| **windows_size** | sliding_windows size, default as 510 |
| **step** | sliding_windows step, default as 50 |


## 2，fine-tuning the models
```
cd ./k_difference
python3 fine_tune.py
```
**Arguments**:
| **Arguments** | **Detail** |
| --- | --- |
| **Early_stop** | Early_stop settings, default as 5 |
| **max_sequences** | max_sequences per batch which should be setting sue to the gpu memory, default as 600 |
| **gpu_counts** | gpu used for training, default as 2 |
| **models** | The list of pretrained model for fine-tuning, default as [3, 4, 5, 6] |

## 3，Using fine-tuned model to extract sequences and post-process the vector by multiple-pooling strategy
```
cd ./k_difference
python3 feature_extraction.py
```
**Arguments**:
| **Arguments** | **Detail** |
| --- | --- |
| **pooling strategy** | The strategy for pooling the circRNA vector where the choices are global average pooling, global Max pooling, default as global Max pooling |
| **gpu_counts** | gpu used for training, default as 2 |
| **models** | The list of pretrained model for fine-tuning, default as [3, 4, 5, 6] |

## 4，Embed using graph contrastive after pairwise matching of semantic features with network structures
```
cd ./graph_feature
python3 results.py
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **Early_stop** | Early stop, default as 300|
| **learning_rate** | default as 1E-5 |

## 5，Visualization of results:
There are mutliple choices for visualize the results.
```
python3  Plot.py -p boxplot
```
<img src="plot/final/boxplot9589.svg" alt="Fig. 2." width="500" height="300">

```
python3  Plot.py -p aucplot
```
<img src="plot/final/auc_curves9589.svg" alt="Fig.3." width="500" height="300">

```
python3  Plot.py -p aucplot
```
<img src="plot/final/aupr_curves9589.svg" alt="Fig.3." width="500" height="300">

```
python3  Plot.py -p barplot
```
<img src="plot/final/barplot9589.svg" alt="Fig.3." width="500" height="300">

```
python3  Plot.py -p bubbleplot
```
<img src="plot/final/9589bubble.svg" alt="Fig.3." width="500" height="300">






# Authors

Jiren Zhou, Boya Ji, Rui Niu, Zhuhong You, Xuequn Shang

# Contacts
If you have any questions or comments, please feel free to email: zhoujiren@nwpu.edu.cn.

# License

[MIT ? Richard McRichface.](../LICENSE)
