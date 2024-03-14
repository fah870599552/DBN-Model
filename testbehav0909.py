import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
class testbehavCompute():
    def __init__(self):
        pass

    def determine_best_num_clusters(self, df, feature_column,weights):
        silhouette_scores = []
        # 尝试不同的簇数
        for num_clusters in range(2, 8):
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            # 对特征列进行归一化
            scaler = StandardScaler()
            df[feature_column] = scaler.fit_transform(df[feature_column])
            # 乘以权重
            weighted_features = df[feature_column].values * weights
            cluster_labels = kmeans.fit_predict(weighted_features)
            silhouette_avg = silhouette_score(df[feature_column], cluster_labels)
            silhouette_scores.append(silhouette_avg)
        # 找到最佳簇数
        best_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        # 加2是因为从2开始尝试的
        return best_num_clusters

    def plot_cluster_scatter(self, df, feature_column, num_clusters,weights):
        # 创建KMeans模型并进行聚类
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        # 对特征列进行归一化
        scaler = StandardScaler()
        df[feature_column] = scaler.fit_transform(df[feature_column])
        # # 提取第一列和第二列作为X和Y轴
        # x = df[feature_column].iloc[:, 0]
        # y = df[feature_column].iloc[:, 1]
        # z = df[feature_column].iloc[:, 2]
        # 乘以权重
        weighted_features = df[feature_column].values * weights
        df['cluster_label'] = kmeans.fit_predict(weighted_features)
        # 反归一化特征列
        df[feature_column] = scaler.inverse_transform(df[feature_column])
        # 提取第一列和第二列作为X和Y轴
        x = df[feature_column].iloc[:, 0]
        y = df[feature_column].iloc[:, 1]
        z = df[feature_column].iloc[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 绘制虚拟散点图以生成图例
        # 创建自定义的颜色映射
        custom_cmap = ListedColormap(['red', 'blue', 'green','purple'])  # 可以根据需要定义更多颜色
        # 绘制散点图，并根据聚类标签着色
        ax.scatter = ax.scatter(x, y, z, c=df['cluster_label'], cmap='viridis')
        ax.set_xlabel(feature_column[0])
        ax.set_ylabel(feature_column[1])
        ax.set_zlabel(feature_column[2])
        plt.title(f'Cluster Results for {num_clusters} Clusters')
        # 为图例添加标签
        plt.legend()
        plt.show()
    def plot_cluster_scatter2test(self, df, feature_column, num_clusters,weights):
        # 创建KMeans模型并进行聚类
        # 驾驶风格的时候用的
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        # 对特征列进行归一化
        # scaler = StandardScaler()
        # df[feature_column] = scaler.fit_transform(df[feature_column])
        # 创建一个空的 DataFrame
        style_data = pd.DataFrame()
        style_data = df[feature_column].copy()
        style_data['plus'] = style_data.sum(axis=1)
        style_data['plus1'] = style_data['plus']
        # # 提取第一列和第二列作为X和Y轴
        # x = df[feature_column].iloc[:, 0]
        # y = df[feature_column].iloc[:, 1]
        # z = df[feature_column].iloc[:, 2]
        # 乘以权重
        # weighted_features = df[feature_column].values * weights
        df['cluster_label'] = kmeans.fit_predict(style_data[['plus1','plus']])
        # 反归一化特征列
        # df[feature_column] = scaler.inverse_transform(df[feature_column])
        # 提取第一列和第二列作为X和Y轴
        x = df[feature_column].iloc[:, 0]
        y = df[feature_column].iloc[:, 1]
        z = df[feature_column].iloc[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 绘制虚拟散点图以生成图例
        # 创建自定义的颜色映射
        custom_cmap = ListedColormap(['red', 'blue', 'green','purple'])  # 可以根据需要定义更多颜色
        # 绘制散点图，并根据聚类标签着色
        ax.scatter = ax.scatter(x, y, z, c=df['cluster_label'], cmap='viridis')
        ax.set_xlabel(feature_column[0])
        ax.set_ylabel(feature_column[1])
        ax.set_zlabel(feature_column[2])
        plt.title(f'Cluster Results for {num_clusters} Clusters')
        # 为图例添加标签
        plt.legend()
        plt.show()
    def map_values_based_on_dict(self, df, target_column, cluster_label_dict):
        # 使用 pandas 的 map 方法根据字典进行值的映射
        df[target_column] = df['cluster_label'].map(cluster_label_dict)
        return df
    def cluster_and_assign(self ,scene ,df, feature_column, target_column, num_clusters,weights):
        # 确定最佳聚类簇数
        # best_num_clusters = self.determine_best_num_clusters(df, feature_column,weights)
        best_num_clusters = num_clusters
        print(f"最佳聚类簇数: {best_num_clusters}")
        self.plot_cluster_scatter(df, feature_column, best_num_clusters,weights)
        # 创建KMeans模型并进行聚类
        kmeans = KMeans(n_clusters=best_num_clusters, random_state=42)
        # 对特征列进行归一化
        scaler = StandardScaler()
        df[feature_column] = scaler.fit_transform(df[feature_column])
        # 乘以权重
        weighted_features = df[feature_column].values * weights
        df['cluster_label'] = kmeans.fit_predict(weighted_features)
        # 反归一化特征列
        df[feature_column] = scaler.inverse_transform(df[feature_column])
        # 获取各个类的中心点
        cluster_centers = kmeans.cluster_centers_
        # 计算轮廓系数
        silhouette_avg = silhouette_score(df[feature_column], df['cluster_label'])
        # 打印聚类结果信息
        print(f"聚类中心点:\n{cluster_centers}")
        print(f"轮廓系数: {silhouette_avg}")
        # 根据聚类标签更新目标列，根据聚类结果，进行标签赋值
        # 创建一个示例 DataFrame
        # 定义转换字典
        #本车驾驶动作对应字典
        scene_dict = {
            'brake': {2: 0, 1: 1, 0: 2},
            'cutin': {1: 0, 0: 1, 2: 2},
            'ghost': {2: 0, 0: 1, 1: 2 },
            'zuozhi': {2: 0, 0: 1, 1: 2 }
        }

        cluster_label_dict = scene_dict[scene]
        # 调用函数进行转换
        df = self.map_values_based_on_dict(df, target_column, cluster_label_dict)
        # 删除临时的聚类标签列
        df.drop(columns=['cluster_label'], inplace=True)
        return df

    def cluster_and_assign2test(self ,scene ,df, feature_column, target_column, num_clusters,weights):
        # 聚类风格的时候用的
        # best_num_clusters = self.determine_best_num_clusters(df, feature_column,weights)
        best_num_clusters = num_clusters
        print(f"最佳聚类簇数: {best_num_clusters}")
        self.plot_cluster_scatter2test(df, feature_column, best_num_clusters,weights)
        # 创建KMeans模型并进行聚类
        kmeans = KMeans(n_clusters=best_num_clusters, random_state=42)
        # 对特征列进行归一化
        # scaler = StandardScaler()
        # df[feature_column] = scaler.fit_transform(df[feature_column])
        # 创建一个空的 DataFrame
        style_data = pd.DataFrame()
        style_data = df[feature_column].copy()
        style_data['plus'] = style_data.sum(axis=1)
        style_data['plus1'] = style_data['plus']
        # 乘以权重
        # weighted_features = df[feature_column].values * weights
        df['cluster_label'] = kmeans.fit_predict( style_data[['plus', 'plus1']])
        # 反归一化特征列
        # df[feature_column] = scaler.inverse_transform(df[feature_column])
        # 获取各个类的中心点
        cluster_centers = kmeans.cluster_centers_
        # 计算轮廓系数
        silhouette_avg = silhouette_score(df[feature_column], df['cluster_label'])
        # 打印聚类结果信息
        print(f"聚类中心点:\n{cluster_centers}")
        print(f"轮廓系数: {silhouette_avg}")
        # 根据聚类标签更新目标列，根据聚类结果，进行标签赋值
        # 创建一个示例 DataFrame
        # 定义转换字典
        #本车驾驶动作对应字典
        if target_column == 'drive_c':
            # 0:gentle 1:moderate 2:hasty
            scene_dict = {
                'brake': {1: 0, 0: 1, 2: 2},
                'cutin': {0: 0, 2: 1, 1: 2},
                'ghost': {0: 0, 1: 1, 2: 2},
                'zuozhi': {0: 0, 2: 1, 1: 2}
            }

        cluster_label_dict = scene_dict[scene]
        # 调用函数进行转换
        df = self.map_values_based_on_dict(df, target_column, cluster_label_dict)
        # 删除临时的聚类标签列
        df.drop(columns=['cluster_label'], inplace=True)
        return df
    def testbehav_kmeans(self,df1:pd.DataFrame,scene):
        if scene == 'brake':
            clusternum = 3
            weights = [1, 1, 0]
            cluster_fe = ['fixed_test_brake_mean', 'fixed_test_throttle_mean', 'fixed_test_steer_mean']
            df1 = self.cluster_and_assign(scene, df1, cluster_fe, 'longit_c',clusternum, weights)
        # 鬼探头：自行车探出公交车，走过主车车道2m
        if scene == 'ghost':
            clusternum = 3
            weights = [1, 1, 0]
            cluster_fe = ['fixed_test_brake_mean', 'fixed_test_throttle_mean', 'fixed_test_steer_mean']
            df1 = self.cluster_and_assign(scene, df1, cluster_fe, 'longit_c', clusternum, weights)
        # 无保护左转：交互车出发
        if scene == 'zuozhi':
            clusternum = 3
            weights = [1, 1, 0]
            cluster_fe = ['fixed_test_brake_mean', 'fixed_test_throttle_mean', 'fixed_test_steer_mean']
            df1 = self.cluster_and_assign(scene, df1, cluster_fe, 'longit_c', clusternum, weights)
        if scene == 'cutin':
            clusternum = 3
            weights = [1, 1, 0]
            cluster_fe = ['fixed_test_brake_mean', 'fixed_test_throttle_mean', 'fixed_test_steer_mean']
            df1 = self.cluster_and_assign(scene, df1, cluster_fe, 'longit_c', clusternum, weights)
        return df1

    def teststyle_kmeans(self,df1:pd.DataFrame,scene):
        if scene == 'brake':
            clusternum = 3
            weights = [1, 1, 1]
            cluster_fe = ['fixed_test_brake_std','fixed_test_throttle_std','fixed_test_steer_std']
            df1 = self.cluster_and_assign2test(scene, df1, cluster_fe, 'drive_c',clusternum, weights)
        # 鬼探头：自行车探出公交车，走过主车车道2m
        if scene == 'ghost':
            clusternum = 3
            weights = [1, 1, 1]
            cluster_fe = ['fixed_test_brake_std', 'fixed_test_throttle_std', 'fixed_test_steer_std']
            df1 = self.cluster_and_assign2test(scene, df1, cluster_fe, 'drive_c', clusternum, weights)
        # 无保护左转：交互车出发
        if scene == 'zuozhi':
            clusternum = 3
            weights = [1, 1, 1]
            cluster_fe = ['fixed_test_brake_std', 'fixed_test_throttle_std', 'fixed_std_test_steer_std']
            df1 = self.cluster_and_assign2test(scene, df1, cluster_fe, 'drive_c', clusternum, weights)
        if scene == 'cutin':
            clusternum = 3
            weights = [1, 1, 1]
            cluster_fe = ['fixed_test_brake_std', 'fixed_test_throttle_std', 'fixed_test_steer_std']
            df1 = self.cluster_and_assign2test(scene, df1, cluster_fe, 'drive_c', clusternum, weights)
        return df1