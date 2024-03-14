import pandas as pd
from pgmpy.estimators import BayesianEstimator
from collections import defaultdict
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
import networkx as nx
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np
import matplotlib.pyplot as plt
from fignew import picturedraw
from sklearn.model_selection import train_test_split
global scene
global save_path
global bn_name
scene='brake'
bn_name = 'cpgcn0918'
save_path = "./bnresult/xlsx/" + scene
# 读取csv文件，并保存到一个DataFrame中
newscenedata_savepath = "./newscene_data/" + scene +'/new' + scene + 'csv0918.csv'
df1 = pd.read_csv(newscenedata_savepath)
# ==============================================================================
#1、时间自相关性分析
# ==============================================================================
# 将数据按 'filename' 列进行分组，并统计每个分组的行数
grouped = df1.groupby('filename').size().reset_index(name='row_count')
# 筛选出行数大于或等于 150 的分组
valid_groups = grouped[grouped['row_count'] >= 150]
# 提取有效分组的 'filename' 列
valid_filenames = valid_groups['filename']
# 根据有效的 'filename' 列筛选原始 DataFrame
new_df1_acf = df1[df1['filename'].isin(valid_filenames)]
# ==============================================================================
#1、聚类算法计算指标
# ==============================================================================


# ==============================================================================
#1、贝叶斯网络结构学习
# ==============================================================================
node_list = ['radi_grade', 'e_c','e_ct-1','ri_g','ri_gt-1','drive_c','drive_ct-1',
                      'man', 'mant-1','sty_man', 'tester_a_c', 'npc_a_c','tester_a_ct-1', 'npc_a_ct-1']
new_df1 = df1.loc[:, node_list]
# 初始化Hill Climbing算法
df_shuffled = new_df1.sample(frac=1, random_state=12)  # 使用随机种子以确保可重复性
train_ratio = 0.80
train_df, test_df = train_test_split(df_shuffled, test_size=1 - train_ratio, random_state=4)
hc = HillClimbSearch(new_df1)
scoring_method = BicScore(new_df1)
best_model = hc.estimate()
if bn_name == 'cpsorgcn0918':
    expert_knowledge = [
        ('man', 'drive_c'),
        ('ri_g', 'tester_a_c'),
        ('radi_grade','tester_a_c'),
        ('npc_a_c','tester_a_c'),
        ('npc_a_c', 'e_c'),
        ('npc_a_c', 'ri_g'),
        ('e_c', 'tester_a_c'),
        ('e_c', 'drive_c'),
        ('e_c', 'man'),
        ('tester_a_c', 'man'),
        ('tester_a_c', 'drive_c'),
        ('drive_c', 'sty_man'),
        ('man', 'sty_man'),
        ('radi_grade', 'drive_c'),
        ('radi_grade', 'e_c'),
        ('radi_grade', 'tester_a_c'),
        #t-1时间片上的结果
        ('npc_a_ct-1','npc_a_c'),
        ('ri_gt-1','ri_g'),
        ('tester_a_ct-1','tester_a_c'),
        ('e_ct-1','e_c'),
        ('drive_ct-1', 'drive_c'),
        ('mant-1', 'man')
    ]
    white_list = [
        ('ri_g', 'e_c'),
        ('ri_g', 'npc_a_c'),
        ('ri_g', 'tester_a_c'),
        ('npc_a_c', 'tester_a_c'),
        ('npc_a_c', 'e_c'),
        ('npc_a_c', 'ri_g'),
        ('e_c', 'tester_a_c'),
        ('e_c', 'drive_c'),
        ('e_c', 'man'),
        ('tester_a_c', 'man'),
        ('tester_a_c', 'drive_c'),
        ('tester_a_c', 'e_c'),
        ('drive_c', 'sty_man'),
        ('man', 'sty_man'),
        ('radi_grade', 'drive_c'),
        ('radi_grade', 'e_c'),
        ('radi_grade', 'man'),
        # t-1时间片上的结果
        ('npc_a_ct-1', 'npc_a_c'),
        ('ri_gt-1', 'ri_g'),
        ('tester_a_ct-1', 'tester_a_c'),
        ('e_ct-1', 'e_c'),
        ('drive_ct-1', 'drive_c'),
        ('mant-1', 'man'),
        #另外考虑的其他边的可能性
        ('ri_g', 'drive_c'),
        ('ri_g', 'man'),
        ('npc_a_c', 'drive_c'),
        ('npc_a_c', 'man'),
        ('radi_grade', 'drive_c'),
        ('radi_grade', 'e_c'),
        ('radi_grade', 'tester_a_c'),
        ('radi_grade', 'man'),
    ]
if bn_name == 'cpsorgcn0918_other':
    expert_knowledge = [
        ('min_ttc_c','e_c'),
        ('min_ttc_c', 'tester_a_c'),
        ('npc_a_c','tester_a_c'),
        ('npc_a_c', 'e_c'),
        ('npc_a_c', 'min_ttc_c'),
        ('e_c', 'tester_a_c'),
        ('e_c', 'drive_c'),
        ('e_c', 'man'),
        ('tester_a_c', 'man'),
        ('tester_a_c', 'drive_c'),
        ('drive_c', 'sty_man'),
        ('man', 'sty_man'),
        ('radi_grade', 'drive_c'),
        ('radi_grade', 'e_c'),
        # ('radi_grade', 'man'),
        #t-1时间片上的结果
        ('npc_a_ct-1','npc_a_c'),
        ('ri_gt-1','min_ttc_c'),
        ('tester_a_ct-1','tester_a_c'),
        ('e_ct-1','e_c'),
        ('drive_ct-1', 'drive_c'),
        ('mant-1', 'man')
    ]
    white_list = [
        ('min_ttc_c', 'e_c'),
        ('min_ttc_c', 'npc_a_c'),
        ('min_ttc_c', 'tester_a_c'),
        ('npc_a_c', 'tester_a_c'),
        ('npc_a_c', 'min_ttc_c'),
        ('e_c', 'drive_c'),
        ('e_c', 'man'),
        ('tester_a_c', 'man'),
        ('tester_a_c', 'drive_c'),
        ('tester_a_c', 'e_c'),
        ('drive_c', 'sty_man'),
        ('man', 'sty_man'),
        ('radi_grade', 'drive_c'),
        ('radi_grade', 'e_c'),
        ('radi_grade', 'man'),
        # t-1时间片上的结果
        ('npc_a_ct-1', 'npc_a_c'),
        ('ri_gt-1', 'min_ttc_c'),
        ('tester_a_ct-1', 'tester_a_c'),
        ('e_ct-1', 'e_c'),
        ('drive_ct-1', 'drive_c'),
        ('mant-1', 'man'),
        #另外考虑的其他边的可能性
        ('min_ttc_c', 'drive_c'),
        ('min_ttc_c', 'man'),
        ('npc_a_c', 'drive_c'),
        ('npc_a_c', 'man'),
        ('radi_grade', 'drive_c'),
        ('radi_grade', 'e_c'),
        ('radi_grade', 'tester_a_c'),
        ('radi_grade', 'man'),
    ]

if bn_name == 'cpgcn0918':
    #待改
    expert_knowledge = [
        ('radi_grade', 'tester_a_c'),
        ('npc_a_c', 'tester_a_c'),
        ('radi_grade', 'ri_g'),
        ('npc_a_c', 'ri_g'),
        ('ri_g','tester_a_c'),
        ('tester_a_c','man'),
        ('tester_a_c', 'drive_c'),
        ('ri_g', 'man'),
        ('ri_g', 'drive_c'),
        ('e_c','man'),
        ('e_c', 'drive_c'),
        ('man','sty_man'),
        ('drive_c','sty_man'),
        #t-1 --> t
        ('npc_a_ct-1', 'npc_a_c'),
        ('ri_gt-1', 'ri_g'),
        ('tester_a_ct-1', 'tester_a_c'),
        ('e_ct-1', 'e_c'),
        ('drive_ct-1', 'drive_c'),
        ('mant-1', 'man')
    ]
    white_list= [
        ('radi_grade', 'tester_a_c'),
        ('npc_a_c', 'tester_a_c'),
        ('radi_grade', 'ri_g'),
        ('npc_a_c', 'ri_g'),
        ('tester_a_c', 'man'),
        ('tester_a_c', 'drive_c'),
        ('ri_g', 'man'),
        ('ri_g', 'drive_c'),
        ('e_c', 'man'),
        ('e_c', 'drive_c'),
        ('man', 'sty_man'),
        ('drive_c', 'sty_man'),
        # t-1 --> t
        ('npc_a_ct-1', 'npc_a_c'),
        ('ri_gt-1', 'ri_g'),
        ('tester_a_ct-1', 'tester_a_c'),
        ('e_ct-1', 'e_c'),
        ('drive_ct-1', 'drive_c'),
        ('mant-1', 'man'),
        #
        ('tester_a_c','ri_g'),
        ('ri_g','tester_a_c')
    ]
white_list_length = len(white_list)
print(white_list_length)
# 使用贪心搜索找到最佳网络结构，并添加专家知识约束条件
best_model = hc.estimate(scoring_method=scoring_method, fixed_edges=expert_knowledge,white_list=white_list)
# 输出最佳网络结构的边
print("Edges in the optimal Bayesian network:")
print(best_model.edges())
best_model_length = len(best_model.edges())
print(best_model_length)
# Compute BIC
bic = BicScore(new_df1)
bic_score = bic.score(best_model)
print("BIC Score: ", bic_score)
# 绘制贝叶斯网络
def plot_bayesian_network(model):
    global save_path
    global bn_name
    G = nx.DiGraph()
    G.add_edges_from(model.edges())
    print(model.edges())
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=2000, node_color='lightblue', font_size=14, ax=ax, arrows = True)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=14, ax=ax)
    ax.set_title('Optimal Bayesian Network')
    graph_path = save_path + '\\' + bn_name + '_graph.adjlist'
    nx.write_adjlist(G, graph_path)
    print("图对象保存完成")
    plt.show()
# 在最佳网络结构上调用绘图函数
plot_bayesian_network(best_model)

def state_read(df, node_list):
    state_names = {}
    for node in node_list:
        name_list = list(sorted(df[node].unique()))
        state_names[node] = name_list
    return state_names

state_names = state_read(new_df1,node_list)
# 将值按照大小顺序排序并映射到整数
man_dict = {value: idx for idx, value in enumerate(sorted(state_names['man']))}
# # # ==============================================================================
# # #2、计算前向推理的后验概率表
# # # ==============================================================================
print('结构学习的节点列表')
print(best_model.nodes())
# Convert the best_model to a BayesianModel
bayesian_best_model = BayesianModel(best_model.edges())
# Estimate the conditional probability tables for each node in the model
prior_type = "BDeu"  # Bayesian Dirichlet equivalent uniform prior
pseudo_counts = 1    # Equivalent sample size for the uniform prior
be = BayesianEstimator(bayesian_best_model,new_df1)
conditional_prob_tables = defaultdict(dict)
for node in bayesian_best_model.nodes():
    conditional_prob_tables[node] = be.estimate_cpd(node, prior_type=prior_type, equivalent_sample_size=pseudo_counts)

# # # ==============================================================================
# # #9、计算所有的后向推理条件概率表
# # # ==============================================================================
def set_uniform_cpds(model, node, additional_state_names=None):
    cpd = model.get_cpds(node)
    if cpd is not None:
        state_names = cpd.state_names[node]
        num_states = len(state_names)
        uniform_prob = 1.0 / num_states

        new_values = np.full((num_states, 1), uniform_prob)

        state_names_dict = {node: state_names}
        if additional_state_names is not None:
            state_names_dict.update(additional_state_names)

        cpd = TabularCPD(node, num_states, new_values, evidence=[], evidence_card=[], state_names=state_names_dict)

        model.add_cpds(cpd)

# # # ==============================================================================
# # #8、计算给定节点的后验概率表，绘制a_a_p图片
# # # ==============================================================================
# Fit the Bayesian model with the estimated conditional probability tables
for cpd in conditional_prob_tables.values():
    bayesian_best_model.add_cpds(cpd)
def read_discretefactor(factor):
    # 获取概率分布的概率值数组
    cardinality = factor.cardinality
    probability_distribution = factor.values
    value_list = []
    p_list = []
    for i in range(cardinality[0]):
        value_list.append(i)
        p_list.append(probability_distribution[i])
    return value_list, p_list

def drawaap_real(df1, target_node,evidence_node):
    # 真实数据执行以上操作
    df_real = df1[[target_node,evidence_node]]
    real_pos = df_real.values.tolist()
    element_countsr = {}
    for element in real_pos:
        if tuple(element) in element_countsr:
            element_countsr[tuple(element)] += 1
        else:
            element_countsr[tuple(element)] = 1
    # 创建新的列表，包含元素 [x, y, z]
    real_list = [[y, x, z] for [x, y], z in element_countsr.items()]
    pos_arrr = np.array(real_list)
    X = pos_arrr[:, 0]
    Y = pos_arrr[:, 1]
    Z = pos_arrr[:, 2]
    title = 'Distribution of ' + target_node + ' and ' + evidence_node + ''
    xis_name = {
        'x': evidence_node,
        'y': target_node,
        'z': 'Counts'
    }
    path = './figdraw0920/BNresult' + '/' + scene + '/'
    label_name = evidence_node + ' to ' + target_node
    label = bn_name + '_real_' + label_name
    figdraw = picturedraw(title, xis_name, label, path)
    figdraw.rainbow_3d_bars(X, Y, Z)
    figdraw.plt_save()
    del figdraw


def drawaap(model, df1,target_node,evidence_node):
    "绘制概率的分布图"
    inference = VariableElimination(model)
    evi_unique = list(df1[evidence_node].unique())
    nodes_pos = []
    for state in evi_unique:
        evidence = {evidence_node: state}
        result = inference.query([target_node], evidence=evidence)
        y,z = read_discretefactor(result)
        for i in range(len(y)):
            pos_t = [state, y[i], z[i]]
            nodes_pos.append(pos_t)
    pos_arr = np.array(nodes_pos)
    X = pos_arr[:,0]
    Y = pos_arr[:,1]
    Z = pos_arr[:,2]
    title = 'Distribution of P(' + target_node + '|'+evidence_node + ')'
    xis_name = {
        'x': evidence_node,
        'y': target_node,
        'z': 'Probability'
    }
    path = './figdraw0920/BNresult' + '/' + scene + '/'
    label_name = evidence_node + ' to ' + target_node
    label = bn_name + '_' + label_name
    figdraw = picturedraw(title, xis_name, label, path)
    figdraw.rainbow_3d_bars(X, Y, Z)
    figdraw.plt_save()
    del figdraw
    #看一下是啥类型

    return

drawaap(bayesian_best_model, new_df1, 'tester_a_c','radi_grade')
drawaap_real(new_df1, 'tester_a_c','radi_grade')
drawaap(bayesian_best_model, new_df1, 'tester_a_c','npc_a_c')
drawaap_real(new_df1, 'tester_a_c','npc_a_c')
drawaap(bayesian_best_model, new_df1, 'sty_man','npc_a_c')
drawaap_real(new_df1, 'sty_man','npc_a_c')
drawaap(bayesian_best_model, new_df1, 'sty_man','ri_g')
drawaap_real(new_df1, 'sty_man','ri_g')
drawaap(bayesian_best_model, new_df1, 'sty_man','radi_grade')
drawaap_real(new_df1, 'sty_man','radi_grade')
drawaap(bayesian_best_model, new_df1, 'man','radi_grade')
drawaap_real(new_df1, 'man','radi_grade')
drawaap(bayesian_best_model, new_df1, 'drive_c','radi_grade')
drawaap_real(new_df1, 'drive_c','radi_grade')
drawaap(bayesian_best_model, new_df1, 'drive_c','e_c')
drawaap_real(new_df1, 'drive_c','e_c')
drawaap(bayesian_best_model, new_df1, 'man','e_c')
drawaap_real(new_df1, 'man','e_c')
drawaap(bayesian_best_model, new_df1, 'sty_man','e_c')
drawaap_real(new_df1, 'sty_man','e_c')
drawaap(bayesian_best_model, new_df1, 'drive_c','npc_a_c')
drawaap_real(new_df1, 'drive_c','npc_a_c')
drawaap(bayesian_best_model, new_df1, 'ri_g','e_c')
drawaap_real(new_df1, 'ri_g','e_c')
# # # ==============================================================================
# # #10、概率推断精度评价
# # # ==============================================================================

def resultevalu(model, df1, target_node:list, man_dict):
    global scene
    """
     target_node = ['man', 'drive_c']
    """
    inference = VariableElimination(model)
    evidence_nodes = ['radi_grade', 'e_c','e_ct-1','ri_g','drive_ct-1',
                       'mant-1','sty_man', 'tester_a_c', 'npc_a_c','tester_a_ct-1']
    evidence = {}
    result_pos = []
    for index, row in df1.iterrows():
        for node in evidence_nodes:
            evidence[node] = row[node]
        result_row_man = inference.query([target_node[0]], evidence=evidence)
        result_row_dri = inference.query([target_node[1]], evidence=evidence)
        # 获取目标节点的最可能值
        most_p_value_man = result_row_man.values.argmax()
        most_p_value_dri = result_row_dri.values.argmax()
        evidence.clear()
        #这是list还是什么
        pot_temp = [most_p_value_man,most_p_value_dri]
        result_pos.append(pot_temp)
    # 创建一个字典来跟踪元素的出现次数
    element_counts = {}
    # 遍历输入列表并计算每个元素的出现次数
    for element in result_pos:
        if tuple(element) in element_counts:
            element_counts[tuple(element)] += 1
        else:
            element_counts[tuple(element)] = 1

    # 创建新的列表，包含元素 [x, y, z]
    result_list = [[x, y, z] for [x, y], z in element_counts.items()]
    pos_arr = np.array(result_list)
    predict_data = pd.DataFrame(pos_arr, columns=['x', 'y', 'z'])
    #真实数据执行以上操作
    df_real = df1[target_node]
    df_real[target_node[0]] = df_real[target_node[0]].replace(man_dict)
    real_pos = df_real.values.tolist()
    element_countsr = {}
    for element in real_pos:
        if tuple(element) in element_countsr:
            element_countsr[tuple(element)] += 1
        else:
            element_countsr[tuple(element)] = 1
    # 创建新的列表，包含元素 [x, y, z]
    real_list = [[x, y, z] for [x, y], z in element_countsr.items()]
    pos_arrr = np.array(real_list)
    real_data = pd.DataFrame(pos_arrr, columns=['x', 'y', 'z'])
    #绘图
    title = 'Prediction Results of DBN'
    xis_name = {
        'x': 'maneuver',
        'y': 'driving operating style',
        'z': 'counts'
    }
    path = './figdraw0920/BNresult' + '/' + scene + '/'
    label = bn_name + '_result'
    figdraw = picturedraw(title, xis_name, label, path)
    figdraw.resultevalue_3d_bars(real_data, predict_data)
    figdraw.plt_save()
    del figdraw
    return

taget_node_res = ['man','drive_c']
resultevalu(bayesian_best_model,test_df, taget_node_res,man_dict)

def infer_backward_probability(inference, parent, child):
    parent_state_names = inference.model.get_cpds(parent).state_names[parent]
    child_state_names = inference.model.get_cpds(child).state_names[child]
    backward_prob = []

    for child_state_name in child_state_names:
        evidence = {child: child_state_name}
        query_result = inference.query(variables=[parent], evidence=evidence)
        probs = query_result.values
        backward_prob.append(probs)

    return backward_prob

inference = VariableElimination(bayesian_best_model)
backward_prob_tables = defaultdict(dict)
for edge in bayesian_best_model.edges():
    parent, child = edge
    result_df = infer_backward_probability(inference, parent, child)
    backward_prob_tables[parent][child] = result_df

def save_backward_probabilities_to_excel(model, backward_prob_tables):
    global scene
    global save_path
    global bn_name
    xlsx_path = save_path+ "\\" + bn_name + ".xlsx"
    writer = pd.ExcelWriter(xlsx_path, engine='xlsxwriter')

    for edge in model.edges():
        parent, child = edge
        table = backward_prob_tables[parent][child]
        df = pd.DataFrame(table)
        df.to_excel(writer, sheet_name=f"{parent}_to_{child}", index=False)

    writer.close()
save_backward_probabilities_to_excel(bayesian_best_model, backward_prob_tables)








