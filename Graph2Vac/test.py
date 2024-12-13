#graph2Vac+RF 大标签
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import chardet
import time


def load_data(base_path):
    """加载数据并为每个类别添加标签"""
    categories = {
        'Benign': ['Monday-Benign.csv', 'Wednesday-Benign.csv'],
        'LDoS': ['Hulk.csv', 'Slowloris.csv', 'slowhttptest.csv', 'GoldenEye.csv'],
        'Bot': ['Bot.csv'],
        'DDoS': ['DDoS.csv'],
        'Patator': ['SSH-Patator.csv', 'FTP-Patator.csv'],
        'PortScan': ['PortScan.csv'],
        'Web Attack': ['Web-Attack-XSS.csv', 'Web-Attack-Brute-Force.csv', 'Web-Attack-Sql-Injection.csv']
    }

    data = pd.DataFrame()
    labels = []

    for category, files in categories.items():
        path = os.path.join(base_path, category)

        for filename in files:
            file_path = os.path.join(path, filename)
            if os.path.exists(file_path):
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
    df = pd.read_csv(file_path, encoding=encoding, index_col=0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


if __name__ == "__main__":
    base_path = "D:\\CICIDS2017\\PCAPs\\myDataset\\first20\\graph2\\LINE-features"

    print("开始加载数据...")
    start_time = time.time()
    X, y = load_data(base_path)
    end_time = time.time()
    print(f"数据加载完毕，耗时：{end_time - start_time:.2f} 秒")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 模型训练
    print("开始训练模型...")
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    end_time = time.time()
    print(f"模型训练完毕，耗时：{end_time - start_time:.2f} 秒")

    # 保存模型
    model_filename = 'random_forest_model.joblib'
    joblib.dump(rf_model, model_filename)
    print(f"模型已保存为：{model_filename}")

    # 预测与评估
    print("开始预测...")
    start_time = time.time()
    loaded_rf_model = joblib.load(model_filename)
    predictions = loaded_rf_model.predict(X_test)
    end_time = time.time()
    print(f"预测完毕，耗时：{end_time - start_time:.2f} 秒")

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print("准确率:", accuracy)
    print("分类报告:\n", report)

