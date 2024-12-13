# 将边使用包间隔时间来标识
import os
import csv
from scapy.all import rdpcap

def read_csv_filenames(directory):
    # 读取CSV文件名并提取五元组和时间戳
    flow_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # 文件名格式：“172.16.0.1-192.168.10.50-32782-80-6-1499260101.pcap_Flow.csv”
            parts = filename.split('-')
            # 时间戳部分需要进一步处理以去除“.pcap_Flow.csv”
            timestamp = parts[5].split('.')[0]  # 从“1499260101.pcap_Flow.csv”获取“1499260101”
            # 五元组和时间戳：源IP, 目标IP, 源端口, 目标端口, 协议, 时间戳
            five_tuple = (parts[0], parts[1], parts[2], parts[3], parts[4], timestamp)
            flow_data.append(five_tuple)
    return flow_data

def read_pcap_files(flow_data, directory, output_csv):
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        # 遍历所有部分的文件夹
        for i in range(1, 20):
            part_dir = os.path.join(directory, f"Wednesday_first20_part{i}")
            if not os.path.exists(part_dir):
                print(f"No dir {part_dir}")
                continue
            for pcap_filename in os.listdir(part_dir):
                pcap_path = os.path.join(part_dir, pcap_filename)
                parts = pcap_filename.split('-')
                pcap_tuple = (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5].split('.')[0])
                if pcap_tuple in flow_data:
                    # 读取PCAP文件并处理
                    print(f"Processing {pcap_path}")
                    try:
                        packets = rdpcap(pcap_path)
                        lengths = []
                        time_intervals = []
                        previous_timestamp = None

                        for packet in packets:
                            if packet.haslayer('IP'):
                                ip_layer = packet['IP']
                                # 只处理源和目标IP匹配的包
                                if (ip_layer.src, ip_layer.dst) == (pcap_tuple[0], pcap_tuple[1]):
                                    lengths.append(len(packet))
                                    current_timestamp = packet.time
                                    if previous_timestamp is not None:
                                        # 计算时间间隔
                                        time_interval = (current_timestamp - previous_timestamp)*1000
                                        time_intervals.append(time_interval)
                                    else:
                                        time_intervals.append(0)  # 第一个包的时间间隔为0
                                    previous_timestamp = current_timestamp
                                else:
                                    lengths.append(-len(packet))
                                    time_intervals.append(0)  # 与目标IP不匹配的包的时间间隔设置为0

                            if len(lengths) == 50:
                                break

                        # 如果不足50个数据包，使用空填充
                        lengths += [''] * (50 - len(lengths))
                        time_intervals += [''] * (50 - len(time_intervals))

                        # 写入数据
                        # 在第一行写入五元组标识和包长度
                        writer.writerow([f"{pcap_tuple}"] + lengths)
                        # 在第二行写入时间间隔
                        writer.writerow([""] + time_intervals)
                    except Exception as e:
                        print(f"Error processing {pcap_path}: {e}")

# 设置文件夹路径
csv_directory = r"D:\CICIDS2017\PCAPs\myDataset\first20\CIC-features\LDoS\slowhttptest"
pcap_directory = r"D:\CICIDS2017\PCAPs\Wednesday\Wednesday-first20"
output_csv = "D:\\CICIDS2017\\PCAPs\\myDataset\\first20\\graph2\\LDoS\\slowhttptest-first20-lengths.csv"

# 步骤1: 读取CSV文件名中的五元组和时间戳
flow_data = read_csv_filenames(csv_directory)

# 步骤2: 读取和处理PCAP文件
read_pcap_files(flow_data, pcap_directory, output_csv)


