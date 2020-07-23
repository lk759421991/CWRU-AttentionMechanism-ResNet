import scipy.io as sio
import math
import numpy as np
import glob

mat_file_folder = './data/case1/'
mat_file_names = glob.glob(mat_file_folder + '*.mat')

# Normal.mat -> normal_DE_box
# OuterRacewayFault_12.mat -> OR_12_DE_box
# OuterRacewayFault_6.mat -> OR_6_DE_box
# OuterRacewayFault_3.mat -> OR_3_DE_box
# BallFault.mat -> B_DE_box
# InnerRacewayFault.mat -> IR_DE_box

mat_file_dic = {'Normal.mat':'normal_DE_box', 'OuterRacewayFault_12.mat':'OR_12_DE_box', 'OuterRacewayFault_6.mat':'OR_6_DE_box',\
                'OuterRacewayFault_3.mat':'OR_3_DE_box', 'BallFault.mat':'B_DE_box', 'InnerRacewayFault.mat':'IR_DE_box'}

mat_file_cls = {'Normal.mat':1, 'OuterRacewayFault_12.mat':6, 'OuterRacewayFault_6.mat':5,\
                'OuterRacewayFault_3.mat':4, 'BallFault.mat':3, 'InnerRacewayFault.mat':2}

train_test_ratio = 0.66

file_name = mat_file_names[0]
mat = sio.loadmat(file_name)
data = mat[mat_file_dic[file_name.split('/')[-1]]]

# data.shape -> [400, 300]
data_num = data.shape[1]
data_dim = data.shape[0]
data_cls = mat_file_cls[file_name.split('/')[-1]]

train_num = math.ceil(data_num*train_test_ratio)
test_num = data_num - train_num

train_data = data[:,0:train_num]
train_label = [data_cls]*train_num

test_data = data[:,train_num:]
test_label = [data_cls]*test_num

for file_name in mat_file_names[1:]:
    mat = sio.loadmat(file_name)
    data = mat[mat_file_dic[file_name.split('/')[-1]]]

    # data.shape -> [400, 300]
    data_num = data.shape[1]
    data_dim = data.shape[0]
    data_cls = mat_file_cls[file_name.split('/')[-1]]

    train_num = math.ceil(data_num*train_test_ratio)
    test_num = data_num - train_num

    train_data_tmp = data[:,0:train_num]
    train_label_tmp = [data_cls]*train_num
    test_data_tmp = data[:,train_num:]
    test_label_tmp = [data_cls]*test_num

    train_data = np.concatenate((train_data,train_data_tmp), axis=1)
    train_label = np.append(train_label,train_label_tmp)

    test_data = np.concatenate((test_data,test_data_tmp), axis=1)
    test_label = np.append(test_label,test_label_tmp)

train_data = np.transpose(train_data)
test_data = np.transpose(test_data)

if len(train_label) == len(train_data) and len(test_label) == len(test_data):
    print([len(train_label), len(test_label)])
    print('What\'s a coincidence!')
    sio.savemat(mat_file_folder+'vali/data.mat', \
                {'train_data':train_data, 'train_label':train_label, 'test_data':test_data, 'test_label':test_label})
else:
    print('Please check the code!')