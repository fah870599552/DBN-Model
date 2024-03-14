import pandas as pd
import re
import numpy
class ttcCompute():
    def __init__(self):
        pass
    def ttc_process(self,trajData:pd.DataFrame,scene):
        if scene == 'cutin' or scene == 'brake' or scene == 'ghost':
            filelist = trajData['filename'].unique().tolist()
            self.trajDict = {}
            for filename in filelist:
                data = trajData[trajData['filename'] == filename]
                data = data.sort_values(by = 'time')
                test_x = data['test_x'].diff()
                npc_x = data['npc_x'].diff()
                test_time = data['time'].diff()
                data['test_x_speed'] = test_x / test_time
                data['npc_x_speed'] = npc_x / test_time
                #当npc_x列大于test_x列，说明npc车辆在行车方向前方，当test_x_speed大于npc_x_speed，说明npc车辆在行车方向前方且本车速度大于npc速度
                #根据上述条件筛选
                data = data[(data['npc_x'] > data['test_x']) & (data['test_x_speed'] > data['npc_x_speed'])]
                #计算TTC
                data['ttc'] = (data['npc_x'] - data['test_x']) / (data['test_x_speed'] - data['npc_x_speed'])
                #剔除ttc列inf和nan值对应的行
                data = data.dropna(subset = ['ttc'])
                data = data[data['ttc'] != numpy.inf]
                self.trajDict[filename] = data
            self.trajData = pd.concat(self.trajDict.values())
        else: #左直交互
            filelist = trajData['filename'].unique().tolist()
            self.trajDict = {}
            for filename in filelist:
                data = trajData[trajData['filename'] == filename]
                data = data.sort_values(by = 'time')
                test_y = data['test_y'].diff()
                npc_y = data['npc_y'].diff()
                test_time = data['time'].diff()
                data['test_y_speed'] = test_y / test_time
                data['npc_y_speed'] = npc_y / test_time
                #当npc_y列大于test_y列，说明npc车辆在行车方向前方
                #根据上述条件筛选
                data = data[(data['npc_y'] > data['test_y'])]
                #计算TTC
                data['ttc'] = (data['npc_y'] - data['test_y']) / (data['test_y_speed'] - data['npc_y_speed'])
                #剔除ttc列inf和nan值对应的行
                data = data.dropna(subset = ['ttc'])
                data = data[data['ttc'] != numpy.inf]
                self.trajDict[filename] = data
            self.trajData = pd.concat(self.trajDict.values())
        return self.trajData



        