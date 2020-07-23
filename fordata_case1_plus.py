import scipy.io as sio
import math
import numpy as np
import glob

mat_file_folder = './data/case1plus/'
mat_file_names = ['Normal', 'InnerRacewayFault', 'BallFault',\
                  'OuterRacewayFault_3', 'OuterRacewayFault_6', 'OuterRacewayFault_12']

# Normal.mat -> normal_DE_box
# OuterRacewayFault_12.mat -> OR_12_DE_box
# OuterRacewayFault_6.mat -> OR_6_DE_box
# OuterRacewayFault_3.mat -> OR_3_DE_box
# BallFault.mat -> B_DE_box
# InnerRacewayFault.mat -> IR_DE_box

# mat_file_dic = {'Normal':'normal_DE_box', 'OuterRacewayFault_12':'OR_12_DE_box', 'OuterRacewayFault_6':'OR_6_DE_box',\
#                 'OuterRacewayFault_3':'OR_3_DE_box', 'BallFault':'B_DE_box', 'InnerRacewayFault':'IR_DE_box'}

mat_file_cls = {'Normal':1, 'OuterRacewayFault_12':6, 'OuterRacewayFault_6':5,\
                'OuterRacewayFault_3':4, 'BallFault':3, 'InnerRacewayFault':2}


train_test_ratio = 0.5

sequence_len = 400
train_data = np.zeros((sequence_len,1))
train_label = [0]
test_data = np.zeros((sequence_len,1))
test_label = [0]

for folder in mat_file_names:
    mat_files = glob.glob(mat_file_folder+folder+'/*.mat')

    for mat in mat_files:
        data = sio.loadmat(mat)
        data = data[mat.split('/')[-1].split('.')[0]]

        # data.shape -> [400, 300]
        data_num = data.shape[1]
        data_dim = data.shape[0]
        data_cls = mat_file_cls[folder]

        train_num = math.ceil(data_num*train_test_ratio)
        test_num = data_num - train_num

        ## randomly split train-test sets
        # index = np.random.permutation(data_num)

        ## first 90% for train, last 10% for test
        index = range(data_num)

        train_index = index[0:train_num]
        test_index = index[train_num:]

        train_data_tmp = data[:,train_index]
        train_label_tmp = [data_cls]*train_num
        test_data_tmp = data[:,test_index]
        test_label_tmp = [data_cls]*test_num

        train_data = np.concatenate((train_data,train_data_tmp), axis=1)
        train_label = np.append(train_label,train_label_tmp)

        test_data = np.concatenate((test_data,test_data_tmp), axis=1)
        test_label = np.append(test_label,test_label_tmp)
        a = 1


train_data = train_data[:,1:]
train_label = train_label[1:]
test_data = test_data[:,1:]
test_label = test_label[1:]

train_data = np.transpose(train_data)
test_data = np.transpose(test_data)

if len(train_label) == len(train_data) and len(test_label) == len(test_data):
    print([len(train_label), len(test_label)])
    print('What\'s a coincidence!')
    sio.savemat(mat_file_folder+'data'+str(train_test_ratio)+'.mat', \
                {'train_data':train_data, 'train_label':train_label, 'test_data':test_data, 'test_label':test_label})
else:
    print('Please check the code!')

