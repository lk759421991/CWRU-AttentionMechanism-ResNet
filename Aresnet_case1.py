import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import hdf5storage
import os

from tqdm import tqdm
from models.model import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 32
num_epochs = 100

for k in range(27):
    snr = str(k)
    # case1
    # net = ResNet(BasicBlock, [1,1,1,1], num_classes=6)
    net = ResNetAttention(BasicBlock, [1,1,1,1], num_classes=6)
    # data = sio.loadmat('./data/case1plus/data0.5.mat')
    # data = sio.loadmat('./data/case1withnoise/snr='+str(snr)+'ttr=0.5.mat')
    data = sio.loadmat('./data/case1withnoisevali/snr='+str(snr)+'ttr=0.5.mat')


    # case2
    # net = ResNet(BasicBlock, [2,2,2,2], num_classes=5)
    # # net = ResNetAttention(BasicBlock, [2,2,2,2], num_classes=5)
    # # data = sio.loadmat('./data/case1plus/data0.5.mat')
    # data = sio.loadmat('./data/case2withnoise/snr=0ttr=0.5.mat')

    train_data = data['train_data']
    train_label = data['train_label']
    train_label = train_label-1

    num_train_instances = len(train_data)

    train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label = torch.from_numpy(train_label).type(torch.LongTensor)
    train_data = train_data.view(num_train_instances, 1, -1)
    train_label = train_label.view(num_train_instances, 1)

    train_dataset = TensorDataset(train_data, train_label)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_data = data['test_data']
    test_label = data['test_label']
    test_label = test_label-1

    num_test_instances = len(test_data)

    test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label = torch.from_numpy(test_label).type(torch.LongTensor)
    test_data = test_data.view(num_test_instances, 1, -1)
    test_label = test_label.view(num_test_instances, 1)

    test_dataset = TensorDataset(test_data, test_label)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    # vali
    vali_data = data['vali_data']
    vali_label = data['vali_label']
    vali_label = vali_label - 1

    num_vali_instances = len(vali_data)

    vali_data = torch.from_numpy(vali_data).type(torch.FloatTensor)
    vali_label = torch.from_numpy(vali_label).type(torch.LongTensor)
    vali_data = vali_data.view(num_vali_instances, 1, -1)
    vali_label = vali_label.view(num_vali_instances, 1)

    vali_dataset = TensorDataset(vali_data, vali_label)
    vali_data_loader = DataLoader(dataset=vali_dataset, batch_size=batch_size, shuffle=False)

    print([num_train_instances,num_vali_instances,num_test_instances])

    if k == 0:
        # train_label_matrix = np.zeros([26,num_train_instances])
        test_label_matrix = np.zeros([27,num_epochs,num_test_instances])
        # vali_label_matrix = np.zeros([26,num_vali_instances])

    validation_label = []
    prediction_label = []

    net = net.cuda()

    criterion = nn.CrossEntropyLoss(size_average=False).cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 250, 300], gamma=0.5)
    # train_loss = np.zeros([num_epochs, 1])
    # test_loss = np.zeros([num_epochs, 1])
    train_acc = np.zeros([num_epochs, 1])
    test_acc = np.zeros([num_epochs, 1])
    vali_acc = np.zeros([num_epochs, 1])

    for epoch in range(num_epochs):
        print('SNR:',snr,'  Epoch:', epoch)
        net.train()
        scheduler.step()
        # loss_x = 0
        for i, (samples, labels) in enumerate(train_data_loader):
        # for (samples, labels) in tqdm(train_data_loader):
            samplesV = Variable(samples.cuda())
            labels = labels.squeeze()
            # print(labels)
            labelsV = Variable(labels.cuda())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            predict_label = net(samplesV)

           # predict_label = caspyra(samplesV)

            loss = criterion(predict_label[0], labelsV)
            # print(loss.item())

            # loss_x += loss.item()

            loss.backward()
            optimizer.step()

        # train_loss[epoch] = loss_x / num_train_instances

        net.eval()
        # loss_x = 0
        correct_train = 0
        for i, (samples, labels) in enumerate(train_data_loader):
            with torch.no_grad():
                samplesV = Variable(samples.cuda())
                labels = labels.squeeze()
                # print(labels)
                labelsV = Variable(labels.cuda())
                # labelsV = labelsV.view(-1)

                predict_label = net(samplesV)
                prediction = predict_label[0].data.max(1)[1]
                # print(prediction)
                correct_train += prediction.eq(labelsV.data.long()).sum()

                loss = criterion(predict_label[0], labelsV)
                # loss_x += loss.item()

        print("Training accuracy:", (100*float(correct_train)/num_train_instances))

        # train_loss[epoch] = loss_x / num_train_instances
        train_acc[epoch] = 100*float(correct_train)/num_train_instances

        trainacc = str(100*float(correct_train)/num_train_instances)[0:6]
    #
    #
    #     loss_x = 0
        correct_test = 0
        prediction_label = []
        for i, (samples, labels) in enumerate(test_data_loader):
            with torch.no_grad():
                samplesV = Variable(samples.cuda())
                labels = labels.squeeze()
                labelsV = Variable(labels.cuda())
                # labelsV = labelsV.view(-1)

            predict_label = net(samplesV)
            prediction = predict_label[0].data.max(1)[1]

            prediction_label.append(prediction.cpu().numpy())
            correct_test += prediction.eq(labelsV.data.long()).sum()

        # a = []
        for batch in range(len(prediction_label)):
            if batch==0:
                a = prediction_label[0]
            else:
                a = np.concatenate((a, prediction_label[batch]))

        test_label_matrix[k,epoch,:] = a

        testacc = str(100 * float(correct_test) / num_test_instances)[0:6]
        # sio.savemat('matfiles/still_prediction'+testacc +'.mat', {'prediction_label': prediction_label})

            # loss = criterion(predict_label[0], labelsV)
            # loss_x += loss.item()

        print("Test accuracy:", (100 * float(correct_test) / num_test_instances))

        # test_loss[epoch] = loss_x / num_test_instances
        test_acc[epoch] = 100 * float(correct_test) / num_test_instances

        testacc = str(100 * float(correct_test) / num_test_instances)[0:6]


        # valiation set
        correct_vali = 0
        validation_label = []
        for i, (samples, labels) in enumerate(vali_data_loader):
            with torch.no_grad():
                samplesV = Variable(samples.cuda())
                labels = labels.squeeze()
                labelsV = Variable(labels.cuda())
                # labelsV = labelsV.view(-1)

            validate_label = net(samplesV)
            validation = validate_label[0].data.max(1)[1]

            validation_label.append(validation.cpu().numpy())
            correct_vali += validation.eq(labelsV.data.long()).sum()

        valiacc = str(100 * float(correct_vali) / num_vali_instances)[0:6]
        # sio.savemat('matfiles/still_prediction'+testacc +'.mat', {'prediction_label': prediction_label})

        # loss = criterion(predict_label[0], labelsV)
        # loss_x += loss.item()

        print("Vali accuracy:", (100 * float(correct_vali) / num_vali_instances))

        # test_loss[epoch] = loss_x / num_test_instances
        vali_acc[epoch] = 100 * float(correct_vali) / num_vali_instances

        valiacc = str(100 * float(correct_vali) / num_vali_instances)[0:6]

        # for i in len(pre)

    # #
    #     if epoch == 0:
    #         temp_test = correct_test
    #         temp_train = correct_train
    #     elif correct_test>temp_test:
    #         torch.save(caspyra, 'weights/changingResnet/StillSpeed_Train' + trainacc + 'Test' + testacc + '.pkl')
    #         temp_test = correct_test
    #         temp_train = correct_train
    #
    # sio.savemat('result/changingResnet/TrainLoss_' + 'ChangingSpeed_Train' + str(100*float(temp_train)/num_train_instances)[0:6] + 'Test' + str(100*float(temp_test)/num_test_instances)[0:6] + '.mat', {'train_loss': train_loss})
    # sio.savemat('result/changingResnet/TestLoss_' + 'ChangingSpeed_Train' + str(100*float(temp_train)/num_train_instances)[0:6] + 'Test' + str(100*float(temp_test)/num_test_instances)[0:6] + '.mat', {'test_loss': test_loss})
    # sio.savemat('result/changingResnet/TrainAccuracy_' + 'ChangingSpeed_Train' + str(100*float(temp_train)/num_train_instances)[0:6] + 'Test' + str(100*float(temp_test)/num_test_instances)[0:6] + '.mat', {'train_acc': train_acc})
    # sio.savemat('result/changingResnet/TestAccuracy_' + 'ChangingSpeed_Train' + str(100*float(temp_train)/num_train_instances)[0:6] + 'Test' + str(100*float(temp_test)/num_test_instances)[0:6] + '.mat', {'test_acc': test_acc})
    # print(str(100*float(temp_test)/num_test_instances)[0:6])

    sio.savemat('./results/case1/CM_WithValiACNN_SNR'+ str(snr)+'accuracy.mat',\
                {'train_accuracy': train_acc, 'test_accuracy':test_acc, 'vali_accuracy':vali_acc,'test_label_matrix':test_label_matrix})

    # print(train_acc)
    # print(test_acc)
    # #
    #
    #
    # for j in range(epoch):
    # correct_test = 0
    # for i, (samples, labels) in enumerate(test_data_loader):
    #     with torch.no_grad():
    #         samplesV = Variable(samples.cuda())
    #         labels = labels.squeeze()
    #         labelsV = Variable(labels.cuda())
    #         # labelsV = labelsV.view(-1)
    #
    #         predict_label_1, predict_label_2, predict_label_3, predict_label_4 = caspyra(samplesV)
    #         prediction = predict_label_1.data.max(1)[1]
    #         # print(prediction)
    #         prediction_label.append(prediction.cpu().numpy())
    #         correct_test += prediction.eq(labelsV.data.long()).sum()
    #
    # testacc = str(100 * float(correct_test) / num_test_instances)[0:6]
    # sio.savemat('matfiles/'+ testacc +'_still_prediction.mat', {'prediction_label': prediction_label})

