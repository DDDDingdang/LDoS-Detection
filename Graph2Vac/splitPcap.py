# pcap文件分块
import os
import shutil

def distribute_pcap_files(base_dir, num_files_per_folder=20000):
    source_folder = os.path.join(base_dir, "Thursday-first10")
    files = [f for f in os.listdir(source_folder) if f.endswith('.pcap')]
    total_files = len(files)
    current_folder = 1
    file_count = 0

    for i, file in enumerate(files):
        # 检查是否需要创建新文件夹
        if i % num_files_per_folder == 0:
            if i + num_files_per_folder > total_files and current_folder == 18:
                # 如果是最后一个文件夹且文件数量不足20000，直接使用剩余的文件数量
                num_files_per_folder = total_files - i
            folder_name = f"Thursday_first10_part{current_folder}"
            current_folder_path = os.path.join(source_folder, folder_name)
            if not os.path.exists(current_folder_path):
                os.makedirs(current_folder_path)
            current_folder += 1
            file_count = 0

        # 移动文件
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(current_folder_path, file)
        shutil.move(source_path, destination_path)
        file_count += 1

    print(f"Files distributed in {current_folder - 1} folders.")

# 使用示例
base_directory = "D:\\CICIDS2017\\PCAPs\\Thursday"
distribute_pcap_files(base_directory)


# # 根据csv给数据流打标签
# import os
# import csv
# import datetime
# import pytz
# import shutil
#
# # Define the paths
# features_dir = 'D:/CICIDS2017/PCAPs/Friday/Friday-first20-features'
# labeling_file = 'D:/CICIDS2017/TrafficLabelling_/Friday-labels/myDataset/DDoS/DDoS.csv'
# target_dir = 'D:/CICIDS2017/PCAPs/myDataset/first20/CIC-features/DDoS'
#
# # Ensure the target directory exists
# os.makedirs(target_dir, exist_ok=True)
#
#
# # Function to convert timestamp to 'America/Halifax' timezone and format it manually to match the label file
# def convert_timestamp(timestamp):
#     utc_dt = datetime.datetime.utcfromtimestamp(int(timestamp))
#     target_tz = pytz.timezone('America/Halifax')
#     target_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(target_tz)
#     # Format the datetime object to remove leading zeros manually
#     formatted_date = target_dt.strftime('%d/%m/%Y %I:%M').lstrip("0").replace("/0", "/").replace(" 0", " ")
#     # formatted_date = target_dt.strftime('%d/%m/%Y %I:%M')
#     return formatted_date
#
# # Load the labels data
# labels_data = {}
# with open(labeling_file, mode='r', newline='') as file:
#     reader = csv.reader(file)
#     next(reader)  # Skip header
#     for row in reader:
#         flow_tuple = row[0]  # Assuming the five-tuple is the first column
#         datetime_label = row[6]  # Assuming the formatted datetime is in column G (index 6)
#         key = (flow_tuple, datetime_label)
#         labels_data[key] = 'DDoS'
#
# # Process each CSV file in the directory
# n = 0
# for filename in os.listdir(features_dir):
#     if filename.endswith('.csv'):
#         n = n+1
#         file_path = os.path.join(features_dir, filename)
#
#         # Extract the five-tuple and timestamp from filename
#         base_name = filename[:-14]  # Remove '.csv'
#         flow_tuple, pcap_timestamp = '-'.join(base_name.split('-')[:5]), base_name.split('-')[-1]
#         file_datetime = convert_timestamp(pcap_timestamp)
#         key = (flow_tuple, file_datetime)
#
#         # Check if the tuple and timestamp match the labels data
#         if key in labels_data:
#             # Copy the file to the new directory and modify the copy
#             new_file_path = os.path.join(target_dir, filename)
#             shutil.copy(file_path, new_file_path)
#
#             with open(new_file_path, 'r+', newline='') as file:
#                 reader = csv.reader(file)
#                 content = list(reader)
#                 if len(content) > 1:
#                     # Replace "No Label" with "LDOS" in the last column of the second line
#                     content[1][-1] = labels_data[key]
#
#                     # Write the modified content back to the new file
#                     file.seek(0)
#                     writer = csv.writer(file)
#                     writer.writerows(content)
#                     file.truncate()
#             print(f"{n} Labeled and copied file: {filename}")
#         else:
#             print(f"{n} No label data for {filename}")

#
# import os
#
# # Directory path
# dir_path = 'D:/CICIDS2017/PCAPs/Monday/Monday-first20-features'
#
# # Process each file in the directory
# for filename in os.listdir(dir_path):
#     if filename.endswith('.csv'):
#         file_path = os.path.join(dir_path, filename)
#         try:
#             with open(file_path, 'r') as file:
#                 lines = file.readlines()
#                 num_lines = len(lines)
#
#             # Now that the file is closed, check the number of lines
#             if num_lines == 1:
#                 os.remove(file_path)
#                 print(f"删除 {filename}，因为它只包含一行。")
#             elif num_lines == 2:
#                 print(f"保留 {filename}，因为它包含两行。")
#         except PermissionError:
#             print(f"无法处理文件 {filename}，因为它正在被其他程序使用。")
#         except Exception as e:
#             print(f"处理文件 {filename} 时遇到错误: {str(e)}")

# #
# #Kmeans处理数据集
# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
#
# # 加载CSV文件
# data = pd.read_csv("D:\\CICIDS2017\\TrafficLabelling_\\Friday-labels\\PortScan.csv")
#
# # 移除列名中的多余空格
# data.columns = data.columns.str.strip()
#
# # 选择从 "Flow Duration" 开始的特征列
# features = data.iloc[:, 7:]
#
# # 筛选出数值型特征
# numeric_features = features.select_dtypes(include=[np.number])
#
# # 检查并处理无限值
# if np.isinf(numeric_features.values).any():
#     numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
#
# # 处理缺失值，可以选择填充或删除
# numeric_features.fillna(numeric_features.mean(), inplace=True)  # 使用均值填充
#
# # 使用标准化缩放器进行特征标准化
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(numeric_features)
#
# # 使用K-means算法进行聚类，k值为2
# kmeans = KMeans(n_clusters=3, random_state=42)
# clusters = kmeans.fit_predict(features_scaled)
#
# # 将聚类结果添加到原始数据中
# data['Cluster'] = clusters
# data.to_csv("D:\\CICIDS2017\\TrafficLabelling_\\Friday-labels\\PortScan.csv", index=False)
#
# # 打印包含聚类结果的数据的前几行
# print(data[['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp', 'Cluster']].head())



# # 根据聚类类别随机选出csv
# import pandas as pd
#
# # 加载CSV文件
# data = pd.read_csv("D:\\CICIDS2017\\TrafficLabelling_\\Wednesday-labels\\slowhttptest.csv")
#
# # 移除列名中的多余空格
# data.columns = data.columns.str.strip()
#
# # 筛选聚类结果为2的行
# cluster_two_data = data[data['Cluster'] == 0]
#
# # 如果行数多于5000行，则随机抽取5000行，否则选择所有行
# sampled_data = cluster_two_data.sample(n=3000, replace=False, random_state=42) if len(cluster_two_data) > 3000 else cluster_two_data
#
# # 保存抽取的数据到新的CSV文件
# sampled_data.to_csv('D:\\CICIDS2017\\TrafficLabelling_\\Wednesday-labels\\myDataset\\LDoS\\slowhttptest.csv', index=False)
#
# print("数据已保存到指定路径。")


# import os
# import glob
#
# # 目录路径
# directory = 'D:\\CICIDS2017\\PCAPs\\myDataset\\first20\\CIC-features\\DDoS'
#
# # 用于存储五元组和最大时间戳的字典
# file_map = {}
#
# # 遍历目录下所有CSV文件
# for file_path in glob.glob(os.path.join(directory, '*.csv')):
#     # 获取文件名
#     filename = os.path.basename(file_path)
#
#     # 解析文件名中的五元组和时间戳
#     parts = filename.split('-')
#     if len(parts) >= 6:
#         five_tuple = '-'.join(parts[:5])  # 构造五元组
#         timestamp = int(parts[5].split('.')[0])  # 时间戳，分割点号取第一部分
#
#         # 判断五元组是否已存在，以及时间戳是否更大
#         if five_tuple in file_map:
#             if file_map[five_tuple]['timestamp'] < timestamp:
#                 # 如果当前文件时间戳较大，删除旧文件，更新记录
#                 os.remove(file_map[five_tuple]['file_path'])
#                 file_map[five_tuple] = {'timestamp': timestamp, 'file_path': file_path}
#             else:
#                 # 如果已记录文件的时间戳较大，删除当前文件
#                 os.remove(file_path)
#         else:
#             # 记录新的五元组及文件信息
#             file_map[five_tuple] = {'timestamp': timestamp, 'file_path': file_path}
#
# print("处理完成，保留了最新的CSV文件。")

# # csv随机抽取10000行
# import pandas as pd
#
# # 加载CSV文件
# data = pd.read_csv("D:\CICIDS2017\TrafficLabelling_\Monday-labels\Monday-Benign-TCP.csv")
#
# # 如果数据行数多于10000行，则随机抽取10000行，否则选择所有行
# if len(data) > 60000:
#     sampled_data = data.sample(n=60000, random_state=42)
# else:
#     sampled_data = data
#
# # 保存抽取的数据到新的CSV文件
# sampled_data.to_csv("D:\\CICIDS2017\\TrafficLabelling_\\Monday-labels\\myDataset\\Monday-Benign-TCP.csv", index=False)
#
# print("随机抽取的数据已保存到新的CSV文件中。")



# # 提取Benign流,不比较时间戳
# import os
# import csv
# import shutil
#
# # Define the paths
# features_dir = 'D:/CICIDS2017/PCAPs/Tuesday/Tuesday-first20-features'
# labeling_file = 'D:/CICIDS2017/TrafficLabelling_/Tuesday-labels/myDataset/Benign/Tuesday-Benign-TCP.csv'
# target_dir = 'D:\\CICIDS2017\\PCAPs\\myDataset\\first20\\CIC-features\\Benign\\Tuesday-Benign'
#
# # Ensure the target directory exists
# os.makedirs(target_dir, exist_ok=True)
#
# # Load the labels data
# labels_data = {}
# with open(labeling_file, mode='r', newline='') as file:
#     reader = csv.reader(file)
#     next(reader)  # Skip header
#     for row in reader:
#         flow_tuple = row[0]  # Assuming the five-tuple is the first column
#         key = (flow_tuple)
#         labels_data[key] = 'Benign'
#
# # Process each CSV file in the directory
# n = 0
# for filename in os.listdir(features_dir):
#     if filename.endswith('.csv'):
#         n = n+1
#         file_path = os.path.join(features_dir, filename)
#
#         # Extract the five-tuple and timestamp from filename
#         base_name = filename[:-14]  # Remove '.csv'
#         flow_tuple = '-'.join(base_name.split('-')[:5])
#         key = (flow_tuple)
#
#         # Check if the tuple and timestamp match the labels data
#         if key in labels_data:
#             # Copy the file to the new directory and modify the copy
#             new_file_path = os.path.join(target_dir, filename)
#             shutil.copy(file_path, new_file_path)
#
#             with open(new_file_path, 'r+', newline='') as file:
#                 reader = csv.reader(file)
#                 content = list(reader)
#                 if len(content) > 1:
#                     # Replace "No Label" with "LDOS" in the last column of the second line
#                     content[1][-1] = labels_data[key]
#
#                     # Write the modified content back to the new file
#                     file.seek(0)
#                     writer = csv.writer(file)
#                     writer.writerows(content)
#                     file.truncate()
#             print(f"{n} Labeled and copied file: {filename}")
#         else:
#             print(f"{n} No label data for {filename}")



# #对齐数据集
# import os
#
# # 提取五元组加时间戳的函数
# def extract_tuple_with_timestamp(filename, is_graph_format=True):
#     if is_graph_format:
#         start = filename.find("('") + 2
#         end = filename.find("').json")
#         tuple_str = filename[start:end]
#         return tuple_str.replace("', '", "-").replace("'", "")
#
#     else:
#         parts = filename.split('-')
#         tuple_str = "-".join(parts[:5])
#         timestamp = parts[5].split('.')[0]  # 提取时间戳部分（去掉扩展名）
#         return f"{tuple_str}-{timestamp}"
#
# # 获取文件夹中的文件名及其五元组+时间戳
# def get_file_tuples_with_names(directory, is_graph_format=True):
#     file_map = {}
#     for filename in os.listdir(directory):
#         if is_graph_format and filename.endswith(".json"):
#             file_tuple = extract_tuple_with_timestamp(filename, True)
#         elif not is_graph_format and filename.endswith(".csv"):
#             file_tuple = extract_tuple_with_timestamp(filename, False)
#         else:
#             continue
#         file_map[file_tuple] = filename
#     return file_map
#
# # 目录路径
# graph_dir = r"D:\CICIDS2017\PCAPs\myDataset\first20\graph\Web Attack\Web Attack Brute Force"
# cic_features_dir = r"D:\CICIDS2017\PCAPs\myDataset\first20\CIC-features\Web Attack\Web Attack Brute Force"
#
# # 获取每个目录的五元组+时间戳映射到文件名的字典
# graph_files_map = get_file_tuples_with_names(graph_dir, is_graph_format=True)
# cic_features_files_map = get_file_tuples_with_names(cic_features_dir, is_graph_format=False)
#
# # 转换字典的key集合
# graph_tuples = set(graph_files_map.keys())
# cic_features_tuples = set(cic_features_files_map.keys())
#
# # 找出多余的文件
# extra_in_graph = graph_tuples - cic_features_tuples
# extra_in_cic_features = cic_features_tuples - graph_tuples
#
# # 输出多余的文件名
# if extra_in_graph:
#     print("Graph 文件夹中多余的文件名：")
#     for extra_tuple in extra_in_graph:
#         print(graph_files_map[extra_tuple])
# else:
#     print("Graph 文件夹中没有多余的文件。")
#
# if extra_in_cic_features:
#     print("CIC-Features 文件夹中多余的文件名：")
#     for extra_tuple in extra_in_cic_features:
#         print(cic_features_files_map[extra_tuple])
# else:
#     print("CIC-Features 文件夹中没有多余的文件。")
#
# print("请手动处理以上列出的多余文件。")