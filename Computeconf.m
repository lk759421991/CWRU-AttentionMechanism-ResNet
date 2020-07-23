clear;
load('./data/case1withnoisevali/snr=26ttr=0.5.mat','test_label');
test_label = test_label';
item_num = length(test_label);
cls_num = length(unique(test_label));
epoch_num = 100;

for snr = 0:26
    fileName = 'CM_WithValiCNN_SNR'+string(snr)+'accuracy.mat';
    load('./results/case1/'+fileName,'test_accuracy','train_accuracy','test_label_matrix','vali_accuracy');
    
    confMatrix = zeros(cls_num,cls_num,epoch_num);
    precision = zeros(cls_num,epoch_num);
    recall = zeros(cls_num,epoch_num);
    f1 = zeros(cls_num,epoch_num);
    
    for epoch = 1:epoch_num
       predict_label = squeeze(test_label_matrix(snr+1,epoch,:))+1;
    
       for i = 1:item_num
           confMatrix(predict_label(i), test_label(i),epoch) = confMatrix( predict_label(i), test_label(i),epoch) + 1;    
       end
       
       for i = 1:cls_num
          precision(i,epoch) = confMatrix(i,i,epoch)/sum(confMatrix(i,:,epoch));
          recall(i,epoch) = confMatrix(i,i,epoch)/sum(confMatrix(:,i,epoch));
          f1(i,epoch) = 2*precision(i,epoch)*recall(i,epoch)/(precision(i,epoch)+recall(i,epoch));
       end

    end
   
    save('./results/case1_f/PRF_'+fileName,'test_accuracy','train_accuracy','vali_accuracy','predict_label',...
    'confMatrix','precision','recall','f1');
end
    

