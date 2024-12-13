# CICFlowmeter+随机森林 小标签
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
    categories = {
        'Benign': ['Monday-Benign', 'Wednesday-Benign'],
        'LDoS': ['Hulk', 'Slowloris', 'slowhttptest', 'GoldenEye'],
        'Bot': [],
        'DDoS': [],
        'Patator': ['SSH-Patator', 'FTP-Patator'],
        'PortScan': [],
        'Web Attack': ['Web Attack XSS', 'Web Attack Brute Force', 'Web Attack Sql Injection']
    }

    data = pd.DataFrame()
    labels = []

    for category, subfolders in categories.items():
        category_path = os.path.join(base_path, category)

        if subfolders:
            for subfolder in subfolders:
                subfolder_path = os.path.join(category_path, subfolder)
                for filename in os.listdir(subfolder_path):
                    if filename.endswith('.csv'):
                        file_path = os.path.join(subfolder_path, filename)
                        with open(file_path, 'rb') as file:
                            result = chardet.detect(file.read())
                            encoding = result['encoding']
                        df = pd.read_csv(file_path, encoding=encoding, usecols=list(range(7, 83)))
                        df.replace([np.inf, -np.inf], np.nan, inplace=True)
                        data = pd.concat([data, df])
                        # 如果是Benign类，使用母目录名称作为标签
                        if category == 'Benign':
                            labels.extend([category] * len(df))
                        else:
                            labels.extend([subfolder] * len(df))
        else:
            for filename in os.listdir(category_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(category_path, filename)
                    with open(file_path, 'rb') as file:
                        result = chardet.detect(file.read())
                        encoding = result['encoding']
                    df = pd.read_csv(file_path, encoding=encoding, usecols=list(range(7, 83)))
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    data = pd.concat([data, df])
                    labels.extend([category] * len(df))

    imputer = SimpleImputer(strategy='median')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data, labels

if __name__ == "__main__":
    base_path = "D:\\CICIDS2017\\PCAPs\\myDataset\\first20\\CIC-features"
    print("开始加载数据...")
    start_time = time.time()
    X, y = load_data(base_path)
    end_time = time.time()
    print(f"数据加载完毕，耗时：{end_time - start_time:.2f}秒")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("开始训练模型...")
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    end_time = time.time()
    print(f"模型训练完毕，耗时：{end_time - start_time:.2f} 秒")

    model_filename = 'random_forest_model.joblib'
    joblib.dump(rf_model, model_filename)
    print(f"模型已保存为：{model_filename}")

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
