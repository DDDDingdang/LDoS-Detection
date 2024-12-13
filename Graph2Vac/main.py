import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import chardet
import time
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

plt.rcParams['font.sans-serif'] = ['SimHei']   # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号 # 有中文出现的情况，需要u'内容'

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

    # 检查是否存在重复行
    duplicate_rows = data[data.duplicated()]
    print(f"重复的行数：{duplicate_rows.shape[0]}")

    return data, labels


def load_csv(file_path):
    """读取CSV文件并处理编码问题"""
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']
    df = pd.read_csv(file_path, encoding=encoding, index_col=0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def k_fold_cross_validation(model, X_train, y_train, n_splits=10):
    """10折交叉验证"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kf)
    return scores

def plot_learning_curve(model, X, y):
    """绘制学习曲线"""
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 20))
    train_scores_mean = train_scores.mean(axis=1)
    test_scores_mean = test_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label='训练集得分')
    plt.plot(train_sizes, test_scores_mean, label='测试集得分')
    plt.xlabel('训练样本数量')
    plt.ylabel('准确率')
    plt.title('学习曲线')
    plt.legend()
    plt.grid()
    plt.show()


def plot_loss_curve(rf_model, X_train, X_test, y_train, y_test):
    """绘制损失曲线 (log_loss)"""
    train_losses = []
    test_losses = []

    # 计算训练集和测试集的 log loss
    for n_estimators in range(1, rf_model.n_estimators + 1):
        rf_model.set_params(n_estimators=n_estimators)
        rf_model.fit(X_train, y_train)

        # 预测概率
        train_probs = rf_model.predict_proba(X_train)
        test_probs = rf_model.predict_proba(X_test)

        # 计算 log loss
        train_loss = log_loss(y_train, train_probs)
        test_loss = log_loss(y_test, test_probs)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    # 绘制损失曲线
    plt.figure()
    plt.plot(range(1, rf_model.n_estimators + 1), train_losses, label='训练集损失')
    plt.plot(range(1, rf_model.n_estimators + 1), test_losses, label='测试集损失')
    plt.xlabel('树的数量 (n_estimators)')
    plt.ylabel('Log Loss')
    plt.title('损失曲线')
    plt.legend()
    plt.grid()
    plt.show()

def plot_top_n_feature_importance(rf_model, feature_names, n=10):
    """绘制前 n 个最重要的特征"""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]  # 排序特征重要性，逆序排列
    top_n_indices = indices[:n]  # 取前 n 个特征

    # 可视化
    plt.figure(figsize=(12, 8))
    plt.title(f"前 {n} 个最重要的特征")
    plt.bar(range(n), importances[top_n_indices], align="center")
    plt.xticks(range(n), np.array(feature_names)[top_n_indices], rotation=90)
    plt.xlim([-1, n])
    plt.tight_layout()
    plt.show()

    # 输出前 n 个特征及其重要性
    top_n_features = np.array(feature_names)[top_n_indices]
    top_n_importances = importances[top_n_indices]
    print(f"前 {n} 个最重要的特征：")
    for feature, importance in zip(top_n_features, top_n_importances):
        print(f"{feature}: {importance:.4f}")


def check_train_test_overlap(X_train, X_test):
    """检查训练集和测试集是否存在重叠数据"""
    overlap = pd.merge(X_train, X_test, how='inner')
    if not overlap.empty:
        print(f"训练集和测试集中有 {overlap.shape[0]} 行重复数据。")
    else:
        print("训练集和测试集没有重复数据。")


if __name__ == "__main__":
    base_path = "D:\\CICIDS2017\\PCAPs\\myDataset\\first20\\graph2\\graph-features"

    print("开始加载数据...")
    start_time = time.time()
    X, y = load_data(base_path)
    end_time = time.time()
    print(f"数据加载完毕，耗时：{end_time - start_time:.2f} 秒")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 保存训练集和测试集
    X_train.to_csv('train_features.csv', index=False)
    pd.DataFrame(y_train).to_csv('../../train_labels.csv', index=False)
    X_test.to_csv('test_features.csv', index=False)
    pd.DataFrame(y_test).to_csv('../../test_labels.csv', index=False)
    print("训练集和测试集已保存。")

    # 检查训练集和测试集是否存在重叠
    check_train_test_overlap(X_train, X_test)

    # 模型训练
    print("开始训练模型...")
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42, oob_score=True)
    rf_model.fit(X_train, y_train)
    end_time = time.time()
    print(f"模型训练完毕，耗时：{end_time - start_time:.2f} 秒")

    # 保存模型
    model_filename = '../../random_forest_model.joblib'
    joblib.dump(rf_model, model_filename)
    print(f"模型已保存为：{model_filename}")

    # 10折交叉验证
    print("开始 10 折交叉验证...")
    start_time = time.time()
    scores = k_fold_cross_validation(rf_model, X_train, y_train, n_splits=10)
    end_time = time.time()
    print(f"10 折交叉验证完毕，耗时：{end_time - start_time:.2f} 秒")
    print(f"10 折交叉验证平均准确率: {scores.mean():.2f} ± {scores.std():.2f}")

    # 训练集和测试集的表现比较
    train_accuracy = rf_model.score(X_train, y_train)
    test_accuracy = rf_model.score(X_test, y_test)
    print(f"训练集准确率: {train_accuracy:.2f}")
    print(f"测试集准确率: {test_accuracy:.2f}")

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

    # 绘制学习曲线
    print("X length:", len(X))
    plot_learning_curve(rf_model, X, y)

    # 绘制损失曲线
    # plot_loss_curve(rf_model, X_train, X_test, y_train, y_test)

    # 为没有特征名的数据生成默认的特征名
    num_features = X.shape[1]
    feature_names = [f'Feature {i+1}' for i in range(num_features)]
    # 绘制前 10 个特征的重要性
    plot_top_n_feature_importance(rf_model, feature_names, n=10)