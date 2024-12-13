# 生成图结构，带有权重边
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json


def load_data(path):
    # 加载数据，假设每个数据流占两行，第一行是包长度，第二行是时间间隔
    data = pd.read_csv(path, header=None)  # 无表头
    print(f"Number of rows loaded: {data.shape[0]}")
    return data


def build_graphs(data):
    graphs = []
    identifiers = []
    start = time.perf_counter()

    # 按照每两行（包长度和包间隔时间）处理一个流
    for i in range(0, data.shape[0], 2):
        identifier = data.iloc[i, 0]  # 获取五元组和时间戳作为标识符
        packet_sizes = data.iloc[i, 1:].to_numpy(dtype=float)  # 第一行是包长度
        time_intervals = data.iloc[i + 1, 1:].to_numpy(dtype=float)  # 第二行是时间间隔
        graph = build_graph(packet_sizes, time_intervals)  # 构建图
        if graph is not None:
            graphs.append(graph)
            identifiers.append(identifier)

    end = time.perf_counter()
    print(f"Time to build all graphs: {end - start} seconds")
    return graphs, identifiers


def build_graph(packet_sizes, time_intervals):
    # 移除包长度和时间间隔中的 NaN 或者空值
    packet_sizes = packet_sizes[~np.isnan(packet_sizes)]
    time_intervals = time_intervals[~np.isnan(time_intervals)]

    if len(packet_sizes) == 0 or len(time_intervals) == 0:
        print("Empty packet sizes or time intervals. Skipping.")
        return None

    graph = nx.Graph()

    # 添加节点
    add_node_attributes(graph, packet_sizes.astype(str))

    # 添加边，边权重为时间间隔
    add_edges_with_weights(graph, time_intervals)

    return graph


def add_node_attributes(graph, packet_sizes):
    # 将包长度作为节点属性
    for i, packet_size in enumerate(packet_sizes):
        graph.add_node(i, feature=packet_size)


def add_edges_with_weights(graph, time_intervals):
    # 将时间间隔作为边的权重
    for i in range(len(time_intervals) - 1):
        graph.add_edge(i, i + 1, weight=time_intervals[i])


def visualize_graph(graph):
    # 对边权重进行变换，并写回图的边属性中
    pos = nx.spring_layout(graph, k=10, iterations=50)
    node_labels = nx.get_node_attributes(graph, 'feature')
    edge_weights = nx.get_edge_attributes(graph, 'weight')

    # 修复：将节点标签先转换为浮点数，再基于它的值来选择颜色
    color_map = ['skyblue' if float(label) < 0 else 'lightgreen' for label in node_labels.values()]

    nx.draw(graph, pos, node_size=850, node_color=color_map, with_labels=False)
    nx.draw_networkx_labels(graph, pos, labels=node_labels)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights)
    plt.show()


def save_graph(graph, identifier, filename):
    data = {
        'edges': [[u, v, graph[u][v]['weight']] for u, v in graph.edges()],
        'features': {n: graph.nodes[n]['feature'] for n in graph.nodes()}
    }
    filename = f"{filename}_{identifier}.json"  # 格式化文件名包含五元组和时间戳
    with open(filename, 'w') as file:
        json.dump(data, file)


def main():
    path = r"D:\CICIDS2017\PCAPs\myDataset\first20\graph2\LDoS\slowhttptest-first20-lengths.csv"
    data = load_data(path)
    graphs, identifiers = build_graphs(data)
    for i, graph in enumerate(graphs):
        save_graph(graph, identifiers[i],
                   f'D:\\CICIDS2017\\PCAPs\\myDataset\\first20\\graph2\\LDoS\\slowhttptest\\graph_{i}')  # 保存JSON文件时使用五元组和时间戳命名
    visualize_graph(graphs[8])  # Visualizing one graph as an example


if __name__ == "__main__":
    main()
