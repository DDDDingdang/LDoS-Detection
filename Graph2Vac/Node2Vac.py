import os
import json
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import pandas as pd

# 数据目录路径
data_directory = r"D:\CICIDS2017\PCAPs\myDataset\first20\graph2\LDoS\slowhttptest"
output_file = r"D:\CICIDS2017\PCAPs\myDataset\first20\graph2\graph-features\LDoS\slowhttptest.csv"

# 用于存储所有文件的特征向量
all_embeddings = []

# 遍历目录，处理每个JSON文件
for filename in os.listdir(data_directory):
    if filename.endswith(".json"):
        file_path = os.path.join(data_directory, filename)

        # 读取JSON文件
        with open(file_path, 'r') as f:
            graph_data = json.load(f)

        # 创建加权图
        G = nx.Graph()
        for edge in graph_data["edges"]:
            G.add_edge(edge[0], edge[1], weight=edge[2])

        # 将节点特征添加到图中
        for node, feature in graph_data["features"].items():
            G.nodes[int(node)]['feature'] = float(feature)

        # 应用Node2Vec算法
        node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=50, weight_key='weight', workers=1)
        model = node2vec.fit(window=5, min_count=1, batch_words=4)

        # 提取节点嵌入向量并求均值表示整个图的嵌入
        node_embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
        graph_embedding = np.mean(node_embeddings, axis=0)

        # 去掉文件名的.json后缀
        file_name_without_extension = filename.replace('.json', '')

        # 将文件名和嵌入向量合并，作为一行数据
        all_embeddings.append([filename] + graph_embedding.tolist())

# 将所有结果保存到CSV文件
embedding_df = pd.DataFrame(all_embeddings)
embedding_df.to_csv(output_file, index=False, header=False)
print(f"所有特征向量已保存到 {output_file}")

