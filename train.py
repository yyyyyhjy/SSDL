import math
import numpy as np
from torch.optim import Adam
import torch
from WGAT import WGATConv
from build_Graph1 import BuildGraph
import pandas as pd
from model import SelfMove
import sklearn.metrics as metric
from utils import build_dictionary,read_pois,read_times,read_cats,flatten,extract_words_vocab,pad_sentence_batch,\
    load_poi_time,load_poi_cat,convert_data,convert_het_data,positional_encoding

voc_poi = list()
city='NYC'
train_file='data/'+city+'/'+city+'_traj_train.txt'
test_file='data/'+city+'/'+city+'_traj_test.txt'
time_train_file='data/'+city+'/'+city+'_traj_time_train.txt'
time_test_file='data/'+city+'/'+city+'_traj_time_test.txt'
cat_train_file='data/'+city+'/'+city+'_traj_cat_train.txt'
cat_test_file='data/'+city+'/'+city+'_traj_cat_test.txt'
static_aug_train_file='data/invariant_aug/'+city+'_traj_train.txt'
static_aug_test_file='data/invariant_aug/'+city+'_traj_test.txt'
dynamic_aug_train_file='data/varying_aug/'+city+'_traj_train.txt'
dynamic_aug_test_file='data/varying_aug/'+city+'_traj_test.txt'
poi_time_file='data/'+city+'/'+city+'_poi_time.txt'
poi_cat_file='data/'+city+'/'+city+'_poi_cat.txt'


voc_poi=build_dictionary(train_file,test_file,voc_poi)
#read pois
H_DATA,Train_DATA, Train_USER, Test_DATA, Test_USER=read_pois(train_file,test_file)
T_TIME,Train_TIME, _, Test_TIME, _=read_times(time_train_file,time_test_file)
T_CAT,Train_CAT, _, Test_CAT, _=read_cats(cat_train_file,cat_test_file)

T=Train_DATA+Test_DATA
total_check=len(flatten(T))
total_user=set(flatten(Train_USER+Test_USER))
user_number=len(set(total_user))

int_to_vocab, vocab_to_int=extract_words_vocab(voc_poi)
print('Dictionary Length', len(int_to_vocab),'POI number',len(int_to_vocab)-3)
TOTAL_POI=len(int_to_vocab)
print('Total check-ins',total_check,TOTAL_POI)
print('Total Users',user_number)

poi_time_list=load_poi_time(poi_time_file,int_to_vocab)
poi_cat_list=load_poi_cat(poi_cat_file,int_to_vocab)

History={}
for key in H_DATA.keys(): # index char
    temp=H_DATA[key]
    temp=flatten(temp)
    new_temp=[]
    for i in temp:
        new_temp.append(vocab_to_int[i])  #word:idx
        #print vocab_to_int[i]
    History[key]=new_temp

History_time={}
for key in T_TIME.keys(): # index char
    temp=T_TIME[key]
    temp=flatten(temp)
    new_temp = []
    for i in temp:
        new_temp.append(int(i))
    History_time[key]=new_temp

History_cat={}
for key in T_CAT.keys(): # index char
    temp=T_CAT[key]
    temp=flatten(temp)
    new_temp = []
    for i in temp:
        new_temp.append(int(i))
    History_cat[key]=new_temp

new_trainT=convert_data(Train_DATA,vocab_to_int)
new_testT=convert_data(Test_DATA,vocab_to_int)
new_trainTime=convert_het_data(Train_TIME)
new_testTime=convert_het_data(Test_TIME)
new_trainCat=convert_het_data(Train_CAT)
new_testCat=convert_het_data(Test_CAT)

def get_onehot(index):
    x = [0] * TOTAL_POI
    x[index] = 1
    return x

#parameters
batch_size=32
train_iters=50

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G=BuildGraph("../", "train").process()
pos_encoding=positional_encoding(len(vocab_to_int),128).to(device)

def train(loc_emb,g_dropout,alpha,beta,gamma,s_dim,r_dim,hidden_units):
    model = SelfMove(emb_dim=loc_emb, out_size=loc_emb, num_heads=2, dropout=g_dropout, node_num=TOTAL_POI,
                    s_dim=s_dim,r_dim=r_dim,hidden_units=hidden_units).to(device)

    model_dict = model.state_dict()
    pretrained_dict=torch.load('pretrains/NYC_'+str(alpha)+'_'+str(beta)+'_'+str(gamma)+'_'+str(loc_emb)+'_'+str(g_dropout)+'_'
                               +str(s_dim)+'_'+str(r_dim)+'_'+str(hidden_units) + '.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 取出预训练模型中与新模型的dict中重合的部分
    model_dict.update(pretrained_dict)  # 用预训练模型参数更新new_model中的部分参数
    model.load_state_dict(model_dict)  # 将更新后的model_dict加载进new model中

    fw_res = open('Result_NYC.txt', 'a')
    fw_res.flush()
    fw_res.write('\n#'+str(alpha)+'_'+str(beta)+'_'+str(gamma)+'_'+str(loc_emb)+'_'+str(g_dropout)+'_'+str(s_dim)+'_'+str(r_dim)+'_'+str(hidden_units)+'\n')
    fw_res.close()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.99), eps=1e-08, weight_decay=0.001, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # sort original data
    index_T = {}
    trainT = []
    trainU = []
    trainTime=[]
    trainCat=[]
    for i in range(len(new_trainT)):
        index_T[i] = len(new_trainT[i])
    temp_size = sorted(index_T.items(), key=lambda item: item[1])
    for i in range(len(temp_size)):
        Tid = temp_size[i][0]
        trainT.append(new_trainT[Tid])
        trainTime.append(new_trainTime[Tid])
        trainCat.append(new_trainCat[Tid])
        trainU.append(Train_USER[Tid])

    # sort for test dataset
    index_T = {}
    testT = []
    testTime=[]
    testCat=[]
    testU = []
    for i in range(len(new_testT)):
        index_T[i] = len(new_testT[i])
    temp_size = sorted(index_T.items(), key=lambda item: item[1])
    for i in range(len(temp_size)):
        Tid = temp_size[i][0]
        testT.append(new_testT[Tid])
        testTime.append(new_testTime[Tid])
        testCat.append(new_testCat[Tid])
        testU.append(Test_USER[Tid])
    # -----------------------------------
    # epoch
    train_size = len(trainT) % batch_size
    test_size = len(testT) % batch_size
    trainT = trainT + trainT[-(batch_size - train_size):]  # copy data and fill the last batch size
    trainTime = trainTime + trainTime[-(batch_size - train_size):]
    trainCat = trainCat + trainCat[-(batch_size - train_size):]
    trainU = trainU + trainU[-(batch_size - train_size):]

    testT = testT + testT[-(batch_size - test_size):]  # copy data and fill the last batch size
    testTime = testTime + testTime[-(batch_size - test_size):]
    testCat = testCat + testCat[-(batch_size - test_size):]
    testU = testU + testU[-(batch_size - test_size):]

    #train
    learning_rate = 0.001  # learning rate
    epoch = 0
    Flag = True
    while (Flag): #每一epoch
        epoch += 1
        if epoch==train_iters:
            Flag = False
            break  # end computing

        #train
        model.train()
        print('learning rate', learning_rate)
        step = 0
        epoch_loss = 0
        temp_loss = 0
        Train_Acc_1 = 0
        Train_Acc_5 = 0
        Train_Acc_10=0
        temp_acc = 0
        while step < len(trainU) // batch_size: #每一step
            start_i = step * batch_size
            batch_x = []
            batch_time=[]
            batch_cat=[]
            label = []
            History_batch=[]
            history_time=[]
            history_cat=[]
            # for test
            input_x = trainT[start_i:start_i + batch_size]
            input_y = trainU[start_i:start_i + batch_size]
            input_time=trainTime[start_i:start_i + batch_size]
            input_cat=trainCat[start_i:start_i + batch_size]
            for uid in input_y:
                History_batch.append(History[uid])
                history_time.append(History_time[uid])
                history_cat.append(History_cat[uid])
            history_x = pad_sentence_batch(History_batch, 0)
            history_time=pad_sentence_batch(history_time,0)
            history_cat=pad_sentence_batch(history_cat,0)
            # print history_x

            for i in input_x:
                batch_x.append(i[:-1])  # add n-1 pois
                label.append(i[-1])
            for i in input_time:
                batch_time.append(i[:-1])
            for i in input_cat:
                batch_cat.append(i[:-1])

            batch_x = pad_sentence_batch(batch_x, 0)
            batch_x=torch.tensor(batch_x).to(device)
            batch_time = pad_sentence_batch(batch_time, 0)
            batch_time = torch.tensor(batch_time).to(device)
            batch_cat = pad_sentence_batch(batch_cat, 0)
            batch_cat = torch.tensor(batch_cat).to(device)

            label=torch.tensor(label).to(device)

            history_x=torch.tensor(history_x).to(device)
            history_time = torch.tensor(history_time).to(device)
            history_cat = torch.tensor(history_cat).to(device)

            poi_time = torch.tensor(poi_time_list).float().to(device)
            poi_cat = torch.tensor(poi_cat_list).float().to(device)

            optimizer.zero_grad()
            predicted_sample,loss=model(G,batch_x,label,history_x,batch_time,history_time,batch_cat,history_cat,poi_time,poi_cat,pos_encoding)
            #predicted_sample,loss=model(G,batch_x,label)
            loss.backward() #retain_graph=True
            epoch_loss += loss.item()
            optimizer.step()

            #predicted_sample = predicted_sample.squeeze(1)
            predicted_sample = predicted_sample.cpu().detach().numpy()

            for i in range(batch_size):
                value = predicted_sample[i]
                top1 = np.argpartition(a=-value, kth=1)[:1]
                # print(top1,input_x[i][-1])
                top5 = np.argpartition(a=-value, kth=5)[:5]
                top10 = np.argpartition(a=-value, kth=10)[:10]
                if top1 == input_x[i][-1]:
                    Train_Acc_1 += 1
                    temp_acc += 1
                if input_x[i][-1] in top5:
                    Train_Acc_5 += 1
                if input_x[i][-1] in top10:
                    Train_Acc_10 += 1

            if(step%100==0):
               print('step：', step, ', loss：', epoch_loss / (100 * batch_size))
               print('step：', step, ', temp_acc：', temp_acc,Train_Acc_5,Train_Acc_10)

            step+=1

        print('epoch is', epoch)
        print('train item', step, epoch_loss)

        #eval
        model.eval()
        with torch.no_grad():
            step = 0
            Test_Acc_1 = 0
            Test_Acc_5 = 0
            Test_Acc_10 = 0
            Prob_Y = []
            Y = []
            y = []
            p_y = []
            while step < len(testU) // batch_size:
                batch_x = []
                label=[]
                batch_time = []
                batch_cat = []
                History_batch = []
                history_time = []
                history_cat = []
                start_i = step * batch_size
                # for test
                input_x = testT[start_i:start_i + batch_size]
                input_time = testTime[start_i:start_i + batch_size]
                input_cat = testCat[start_i:start_i + batch_size]
                input_y = testU[start_i:start_i + batch_size]
                for uid in input_y:
                    History_batch.append(History[uid])
                    history_time.append(History_time[uid])
                    history_cat.append(History_cat[uid])
                history_x = pad_sentence_batch(History_batch, 0)
                history_time = pad_sentence_batch(history_time, 0)
                history_cat = pad_sentence_batch(history_cat, 0)

                for i in input_x:
                    batch_x.append(i[:-1])  # add n-1 pois
                    label.append(i[-1])
                for i in input_time:
                    batch_time.append(i[:-1])
                for i in input_cat:
                    batch_cat.append(i[:-1])

                batch_x = pad_sentence_batch(batch_x, 0)
                batch_x = torch.tensor(batch_x).to(device)
                batch_time = pad_sentence_batch(batch_time, 0)
                batch_time = torch.tensor(batch_time).to(device)
                batch_cat = pad_sentence_batch(batch_cat, 0)
                batch_cat = torch.tensor(batch_cat).to(device)

                label = torch.tensor(label).to(device)

                history_x = torch.tensor(history_x).to(device)
                history_time = torch.tensor(history_time).to(device)
                history_cat = torch.tensor(history_cat).to(device)

                poi_time = torch.tensor(poi_time_list).float().to(device)
                poi_cat=torch.tensor(poi_cat_list).float().to(device)

                predicted_sample,loss = model(G,batch_x,label,history_x,batch_time,history_time,batch_cat,history_cat,poi_time,poi_cat,pos_encoding)
                #predicted_sample, loss = model(G, batch_x, label)
                predicted_sample = predicted_sample.cpu().detach().numpy()

                for i in range(batch_size):
                    value = predicted_sample[i]
                    true_value = get_onehot(input_x[i][-1])
                    top1 = np.argpartition(a=-value, kth=1)[:1]
                    top5 = np.argpartition(a=-value, kth=5)[:5]
                    top10 = np.argpartition(a=-value, kth=10)[:10]

                    Y.append(true_value)
                    Prob_Y.append(value)
                    y.append(input_x[i][-1])
                    p_y.append(top1)
                    # print top1
                    # print input_x[i][-1]
                    if top1 == input_x[i][-1]:
                        Test_Acc_1 += 1
                    if input_x[i][-1] in top5:
                        Test_Acc_5 += 1
                    if input_x[i][-1] in top10:
                        Test_Acc_10 += 1
                step += 1

        scheduler.step()

        print('\n --Test accuracy@1: ', Test_Acc_1 / (step * batch_size), Test_Acc_1 / len(testU))
        print('\n --Test accuracy@5: ', Test_Acc_5 / (step * batch_size), Test_Acc_5 / len(testU))
        print('\n --Test accuracy@10: ', Test_Acc_10 / (step * batch_size), Test_Acc_10 / len(testU))
        # print len(Y)
        # print len(Prob_Y)
        auc=0
        map=0
        if (epoch+1)%5==0:
          Y = np.array(Y)
          Prob_Y = np.array(Prob_Y)
          auc = metric.roc_auc_score(Y.T, Prob_Y.T, average='micro')
          map = metric.average_precision_score(Y.T, Prob_Y.T, average='micro')
          print('auc_value', auc)  # ,,average='micro'
          print('MAP_value', map)  # ,,average='micro'
        print('---------------------------\n')
        print('---------------------------\n')
        ACC_1=Test_Acc_1 / len(testU)
        ACC_5=Test_Acc_5 / len(testU)
        ACC_10=Test_Acc_10 / len(testU)
        output(epoch,ACC_1,ACC_5,ACC_10,auc,map)



def output(index, ACC_1, ACC_5, ACC_10,auc,map):
    fw_res = open('Result_NYC.txt', 'a')
    fw_res.flush()
    fw_res.write('epoch' + str(index) + '\tACC@1\t' + str(ACC_1) + '\tACC@5\t' + str(ACC_5) + '\tACC@10\t' + str(ACC_10) +
                 '\t' + str(auc)+'\t'+str(map)+ '\n')
    fw_res.close()


train(loc_emb=256,g_dropout=0,alpha=1,beta=1,gamma=0.1,s_dim=256,r_dim=32,hidden_units=300)