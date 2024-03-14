import re
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pgmpy.estimators import HillClimbSearch, BicScore
import pickle
import numpy as np
import ast
from TTC_compute_0906 import ttcCompute
from sklearn.cluster import KMeans
from npcbehav0909 import threatCompute
from sklearn.metrics import silhouette_score
from testbehav0909 import testbehavCompute
global scene
import math
"""
统一计算所有所有节点特征值的计算
"""

# ==============================================================================
#1、提取数据
# ==============================================================================
#转化带分隔符的[ ]数据
def trans(x):
    """
    :param x: str
    :return: list of float
    """
    df[x] = df[x].astype(str)
    df[x] = df[x].apply(lambda x: re.findall(r'[-]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', x))
    df[x] = df[x].apply(lambda x: [float(i) for i in x])
    return df[x]
def get_index(row, scene):
    # Convert the string to a list
    name_dict = {
        'cutin':'cut0',
        'brake':'bra0',
        'ghost':'gho0',
        'zuozhi':'zuo0'
    }
    npc_name = name_dict[scene]
    list_of_names = ast.literal_eval(row)
    # Remove the extra quotes and spaces from each element
    list_of_names = [name.replace('"', '').replace('\'', '').strip() for name in list_of_names]
    # Find the index of 'cut0'
    if npc_name in list_of_names:
        return list_of_names.index(npc_name)
    else:
        return None

def preprocess(df):
    for col in ['veh_speed', 'x', 'y', 'test_a', 'test_throttle', 'test_steer', 'test_brake']:
        trans(col)
    df['npc_index'] = df.apply(lambda row: get_index(row['car_name'], row['Scenes']), axis=1)
    df = df.dropna().reset_index(drop=True)
    df['test_x'] = df['x'].apply(lambda x: x[0])
    df['test_y'] = df['y'].apply(lambda y: y[0])
    df['test_v'] = df['veh_speed'].apply(lambda v: v[0])
    df['npc_x'] = df.apply(lambda row: row['x'][int(row['npc_index'])], axis=1)
    df['npc_y'] = df.apply(lambda row: row['y'][int(row['npc_index'])], axis=1)
    df['npc_v'] = df.apply(lambda row: row['veh_speed'][int(row['npc_index'])], axis=1)
    df['test_throttle'] = df['test_throttle'].apply(lambda t: t[0])
    df['test_steer'] = df['test_steer'].apply(lambda s: s[0])
    df['test_brake'] = df['test_brake'].apply(lambda b: b[0])
    df['tester_a'] = df['test_a'].apply(lambda a: a[0])
    df['npc_a'] = df['test_a'].apply(lambda a: a[1])
    df = df.dropna().reset_index(drop=True)
    return df
scene='brake'
file_oripath = "original_data/" + scene + "/" + scene + "csv0810.csv"
df = pd.read_csv(file_oripath)
df = preprocess(df)

# ==============================================================================
#2、处理异常数据
# ==============================================================================
def calculate_npc_heading(group):
    dx = group['npc_x'].diff()
    dy = group['npc_y'].diff()
    group['npc_heading'] = np.arctan2(dy, dx)
    return group

def calculate_npc_jerk(group):
    group['npc_jerk'] = group['npc_a'].diff() / group['time'].diff()
    return group

def handle_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.05)
        Q3 = df[col].quantile(0.95)
        IQR = Q3 - Q1
        lower_threshold = Q1 - 1.5 * IQR
        upper_threshold = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_threshold, upper_threshold)
    return df
df = df.groupby('filename').apply(calculate_npc_jerk)
df.reset_index(drop=True, inplace=True)
df = df.groupby('filename').apply(calculate_npc_heading)
df.reset_index(drop=True, inplace=True)
df = df.dropna().reset_index(drop=True)
df['tester_a'] = df['tester_a'].clip(-10, 10)
df['npc_a'] = df['npc_a'].clip(-10, 10)
df = handle_outliers(df, ['test_jerk', 'npc_jerk'])
ttcComputeImplement = ttcCompute()
dfNew = ttcComputeImplement.ttc_process(df,scene)
#依据time列和filename，将dfNew中的ttc列值赋给df中的ttc列
df = df.merge(dfNew[['time','filename','ttc']], how='left', on=['time','filename'])
#将ttc列中的nan值替换为99
df['ttc'] = df['ttc'].fillna(99)
# ==============================================================================
#以10个时间点（0.4s）滚动窗口求最大test_v、平均值test_v等
# ==============================================================================
def calculate_rolling_features(group, windownum):
    group["min_ttc"] = group['ttc'].rolling(window=windownum).min()
    return group
def calculate_window_stats(group, window_size):
    for col in ['test_v','test_steer','tester_a', 'test_jerk', 'test_throttle', 'test_brake', 'npc_a', 'npc_jerk', 'npc_v']:
        group[f'fixed_{col}_mean'] = group[col].rolling(window=window_size, min_periods=1).mean()
        group[f'fixed_{col}_std'] = group[col].rolling(window=window_size, min_periods=1).std()

    return group
df1 = df.groupby('filename').apply(calculate_rolling_features, 10).reset_index(drop=True)
df1 = df1.groupby('filename').apply(lambda x: calculate_window_stats(x, window_size=10)).reset_index(drop=True)
df1 = df1.dropna().reset_index(drop=True)

#==================================================
#1、npc_a,npc_a划分函数
def tn_a_cluster(dataframe, column_name, interval):
    # 创建等级的区间
    max_value = dataframe[column_name].max()
    min_value = dataframe[column_name].min()
    # 使用numpy.arange()函数创建等级的区间
    bins = np.arange(min_value-interval, max_value + 2*interval, interval)
    # 使用pd.cut函数将指定列划分成等级，并将结果存储到新列中
    dataframe['{}_c'.format(column_name)] = pd.cut(dataframe[column_name], bins, labels=False)
    return dataframe
df1 = tn_a_cluster(df1, 'npc_a', 0.2)
df1 = tn_a_cluster(df1, 'tester_a', 0.2)

# ==============================================================================
#3、碰撞风险、驾驶激进度评级
# ==============================================================================
#碰撞风险
def risk_judge(min_ttc):
    if min_ttc>=3.5 :
        risk_level=0#1表示无碰撞风险或碰撞风险低
    elif min_ttc<=0:
        risk_level=0#1表示无碰撞风险或碰撞风险低
    elif 1.75<min_ttc<=3.5:
        risk_level=1#2表示碰撞风险一般
    elif 0<min_ttc<=1.75:
        risk_level=2#3表示碰撞风险高
    return risk_level
ri_g=[]
for i in range(0,len(df1['min_ttc'])):
    ri_g.append(risk_judge(df1['min_ttc'][i]))
df1['ri_g'] = ri_g


#驾驶员主观风格量表分数评级
df1['radi_score']=df1['ques_score']+df1['style_score']
# 设置条件和对应的值，这个条件再看一下
conditions = [
    (df1['radi_score'] >= 55),
    (df1['radi_score'] < 55) & (df1['radi_score'] > 45),
    (df1['radi_score'] <= 45)
]
#2是激进，1中立，0保守
values = [2, 1, 0]

# 使用 np.select() 根据条件进行赋值,四个场景统一的特征
df1['radi_grade'] = np.select(conditions, values, default=2)

# ==============================================================================
######4.1.1聚类1： 情绪聚类
# ==============================================================================
emo_data = df1[['emo_p', 'emo_a', 'emo_d']]
emo_data['emo_p'] = emo_data['emo_p'].clip(-1,0.1)
with open('./pklresult/emo_kmeans_model/522_emo_kmeans_model.pkl', 'rb') as f:
    emo_loaded_model = pickle.load(f)

emo_data = df1[['emo_p', 'emo_a', 'emo_d']]
df1['e_c']=emo_loaded_model.predict(emo_data)
df1['e_c'] = df1['e_c'] + 1
#将6、5、4都变为0
df1['e_c'] = df1['e_c'].replace(6, 0)
df1['e_c'] = df1['e_c'].replace(5, 0)
df1['e_c'] = df1['e_c'].replace(4, 0)
#将3、2变成9
df1['e_c'] = df1['e_c'].replace(3, 9)
df1['e_c'] = df1['e_c'].replace(2, 9)
#将1变成2
df1['e_c'] = df1['e_c'].replace(1, 2)
#将9变成1
df1['e_c'] = df1['e_c'].replace(9, 1)
#e_c对应的编号数字
#2：惊恐，1：愤怒，0：中性,确认是对的

### 交互车动作，交互车驾驶动作识别
##急刹：a<急刹a（-1）
threatComputeImple = threatCompute()
if scene != 'ghost':
    df1 = threatComputeImple.threat_kmeans(df1, scene)
# 鬼探头：自行车探出公交车，走过主车车道2m
if scene == 'ghost':
    condition = abs(df1['npc_y'] - df1['test_y']) < 5
    df1['n_thr'] = np.where(condition, 1, 0)
#### 本车动作，本车驾驶动作识别：steer,brake,throttle力度标量，10个时间窗口的平均值
# # 先根据阈值判断转向=====================================================
##车辆方向盘（左、直、右）####设置阈值为±0.1，对应方向盘转角为5.19°！0.077对应4°
def determine_steer_cluster(value):
    if value > 0.077:
        #右转
        return 6
    elif value < -0.077:
        #左转
        return 3
    else:
        #保持直行
        return 0
# def determine_steer_cluster(value):
#     return 1
df1['steer_c'] = df1['fixed_test_steer_mean'].apply(determine_steer_cluster)
steer_data = df1[['fixed_test_steer_mean','steer_c']]
counts = steer_data['steer_c'].value_counts()
print(counts)
testbehavComputeImple = testbehavCompute()
df1 = testbehavComputeImple.testbehav_kmeans(df1, scene)
# 聚类得到动作的风格聚类，drive_c, gentle,hasty
df1 = testbehavComputeImple.teststyle_kmeans(df1, scene)
# 回归得到所有动作的组合
df1['man'] = df1['longit_c']+ 10 + df1['steer_c']
man_counts = df1['man'].value_counts()
print('本车动作所有动作组合及占比为')
print(man_counts)

man_counts = df1['drive_c'].value_counts()
print('本车动作所有风格占比为')
print(man_counts)
df1['ri_gt-1'] = df1.groupby('filename')['ri_g'].shift(1)
df1['e_ct-1'] = df1.groupby('filename')['e_c'].shift(1)
df1['n_thrt-1'] = df1.groupby('filename')['n_thr'].shift(1)
df1['mant-1'] = df1.groupby('filename')['man'].shift(1)
df1['drive_ct-1'] = df1.groupby('filename')['drive_c'].shift(1)
df1['npc_a_ct-1'] = df1.groupby('filename')['npc_a_c'].shift(1)
df1['tester_a_ct-1'] = df1.groupby('filename')['tester_a_c'].shift(1)
df1 = df1.dropna().reset_index(drop=True)
# #将1、1组成1；2、1组成10；3、1组成19......→style_maneuver：B
df1['sty_man']=df1['drive_c'] * 10 + df1['man']
filename_counts = df1['filename'].value_counts()
print("Number of unique filenames:", len(filename_counts))
print("Filename counts:")
print(filename_counts)
newscenedata_savepath = "./newscene_data/" + scene +'/new' + scene + 'csv0918.csv'
df1.to_csv(newscenedata_savepath)
