# CICflowmeter+LSTM
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import chardet
import time


def load_data(base_path):
    """加载数据并为每个类别添加标签"""
    categories = {
        'Benign': ['Monday-Benign', 'Wednesday-Benign'],
        'LDoS': ['GoldenEye', 'Hulk', 'slowloris', 'slowhttptest'],
        'Bot': [],
        'DDoS': [],
        'Patator': ['FTP-Patator', 'SSH-Patator'],
        'PortScan': [],
        'Web Attack': ['Web Attack XSS', 'Web Attack Sql Injection', 'Web Attack Brute Force']
    }

    data = pd.DataFrame()
    labels = []

    for category, subdirs in categories.items():
        path = os.path.join(base_path, category)

        if subdirs:
            for subdir in subdirs:
                subdir_path = os.path.join(path, subdir)
                for filename in os.listdir(subdir_path):
                    if filename.endswith('.csv'):
                        file_path = os.path.join(subdir_path, filename)
                        df = load_csv(file_path)
                        data = pd.concat([data, df])
                        labels.extend([category] * len(df))
        else:
            for filename in os.listdir(path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(path, filename)
                    df = load_csv(file_path)
                    data = pd.concat([data, df])
                    labels.extend([category] * len(df))

    # 处理缺失值
    imputer = SimpleImputer(strategy='median')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # 标准化数据
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data, labels


def load_csv(file_path):
    """读取CSV文件并处理编码问题"""
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']
    df = pd.read_csv(file_path, encoding=encoding, usecols=list(range(7, 83)))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


if __name__ == "__main__":
    base_path = "D:\\CICIDS2017\\PCAPs\\myDataset\\first20\\CIC-features"

    print("开始加载数据...")
    start_time = time.time()
    X, y = load_data(base_path)
    end_time = time.time()
    print(f"数据加载完毕，耗时：{end_time - start_time:.2f} 秒")

    # 标签编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=42)

    # 将DataFrame转换为NumPy数组
    X_train = X_train.values
    X_test = X_test.values

    # 调整形状以适应LSTM输入
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # LSTM模型
    print("开始构建LSTM模型...")
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(y_categorical.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("开始训练LSTM模型...")
    start_time = time.time()
    model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))
    end_time = time.time()
    print(f"LSTM模型训练完毕，耗时：{end_time - start_time:.2f} 秒")

    # 保存模型
    model_filename = 'lstm_model.h5'
    model.save(model_filename)
    print(f"LSTM模型已保存为：{model_filename}")

    # 评估模型
    print("开始评估LSTM模型...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"测试集准确率: {accuracy:.2f}")