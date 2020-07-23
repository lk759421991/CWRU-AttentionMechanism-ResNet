import scipy.io as sio
import numpy as np
import math

folder = './data/case1withnoise/'
# snr = '25'

fileName = {'BallFault_1730':3, 'BallFault_1750':3, 'BallFault_1772':3, 'BallFault_1797':3,\
            'InnerRacewayFault_1730':2, 'InnerRacewayFault_1750':2, 'InnerRacewayFault_1772':2,'InnerRacewayFault_1797':2,\
            'Normal_1730':1, 'Normal_1750':1, 'Normal_1772':1, 'Normal_1797':1,\
            'OuterRacewayFault_3_1730':4, 'OuterRacewayFault_3_1750':4,'OuterRacewayFault_3_1772':4, 'OuterRacewayFault_3_1797':4,\
            'OuterRacewayFault_6_1730':5, 'OuterRacewayFault_6_1750':5, 'OuterRacewayFault_6_1772':5, 'OuterRacewayFault_6_1797':5,\
            'OuterRacewayFault_12_1730':6, 'OuterRacewayFault_12_1750':6, 'OuterRacewayFault_12_1772':6, 'OuterRacewayFault_12_1797':6}

train_test_ratio = 0.5

for k in range(27):
    snr = str(k)
    sequence_len = 400
    train_data = np.zeros((sequence_len,1))
    train_label = [0]
    test_data = np.zeros((sequence_len,1))
    test_label = [0]

    Data = sio.loadmat(folder+snr+'.mat')

    for item in fileName:
        data = Data[item]
        data_cls = fileName[item]

        data_num = data.shape[1]
        data_dim = data.shape[0]

        train_num = math.ceil(data_num * train_test_ratio)
        test_num = data_num - train_num
        index = range(data_num)

        train_index = index[0:train_num]
        test_index = index[train_num:]

        train_data_tmp = data[:, train_index]
        train_label_tmp = [data_cls] * train_num
        test_data_tmp = data[:, test_index]
        test_label_tmp = [data_cls] * test_num

        train_data = np.concatenate((train_data, train_data_tmp), axis=1)
        train_label = np.append(train_label, train_label_tmp)

        test_data = np.concatenate((test_data, test_data_tmp), axis=1)
        test_label = np.append(test_label, test_label_tmp)
        a = 1

    train_data = train_data[:, 1:]
    train_label = train_label[1:]
    test_data = test_data[:, 1:]
    test_label = test_label[1:]

    train_data = np.transpose(train_data)
    test_data = np.transpose(test_data)


    if len(train_label) == len(train_data) and len(test_label) == len(test_data):
        print([len(train_label), len(test_label)])
        print('What\'s a coinci-dance!')
        sio.savemat(folder+'snr='+snr+'ttr='+str(train_test_ratio)+'.mat', \
                    {'train_data':train_data, 'train_label':train_label, 'test_data':test_data, 'test_label':test_label})
    else:
        print('Please check the code!')
