import math
import numpy as np
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.nn.functional as F
from WGAT import WGATConv
from build_Graph1 import BuildGraph
import pandas as pd
from model_pretrain import PreSelfMove
from loss import contrastive_loss, compute_mi
from mutual_info import logsumexp, log_density, log_importance_weight_matrix
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
#read locations
H_DATA,Train_DATA, Train_USER, Test_DATA, Test_USER=read_pois(train_file,test_file)
H_DATA_ca,Train_DATA_ca, _, Test_DATA_ca, _=read_pois(static_aug_train_file,static_aug_test_file)
H_DATA_ma,Train_DATA_ma, _, Test_DATA_ma, _=read_pois(dynamic_aug_train_file,dynamic_aug_test_file)
# read times and cats
T_TIME,Train_TIME, _, Test_TIME, _=read_times(time_train_file,time_test_file)
T_CAT,Train_CAT, _, Test_CAT, _=read_cats(cat_train_file,cat_test_file)

#print History['1']
T=Train_DATA+Test_DATA
total_check=len(flatten(T))
total_user=set(flatten(Train_USER+Test_USER))
user_number=len(set(total_user))

int_to_vocab, vocab_to_int=extract_words_vocab(voc_poi)
print(len(int_to_vocab))
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
new_trainT_ca=convert_data(Train_DATA_ca,vocab_to_int)
new_testT_ca=convert_data(Test_DATA_ca,vocab_to_int)
new_trainT_ma=convert_data(Train_DATA_ma,vocab_to_int)
new_testT_ma=convert_data(Test_DATA_ma,vocab_to_int)

new_trainTime=convert_het_data(Train_TIME)
new_testTime=convert_het_data(Test_TIME)
new_trainCat=convert_het_data(Train_CAT)
new_testCat=convert_het_data(Test_CAT)

def get_onehot(index):
    x = [0] * TOTAL_POI
    x[index] = 1
    return x

def get_max_index(list):
    length = len(list) - 1
    return length

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G=BuildGraph("../", "train").process()
pos_encoding=positional_encoding(len(vocab_to_int),128).to(device)

contras_fn = contrastive_loss(tau=0.5, normalize=True)
def cal_loss(x,f_mean, f_logvar, f, z_post_mean, z_post_logvar, z_post, z_prior_mean, z_prior_logvar, z_prior, recon_x,
            f_mean_c, f_logvar_c, f_c,z_post_mean_m, z_post_logvar_m, z_post_m,f_dim,z_dim,alpha,gamma,beta):
    f_dim=f_dim
    z_dim=z_dim
    dataset_size=len(new_trainT)
    weight_f=alpha
    weight_z=alpha
    weight_fz=gamma
    weight_c_aug=beta
    weight_m_aug=beta

    batch_size, n_frame, _ = z_post_mean.size()

    mi_xs = compute_mi(f, (f_mean, f_logvar))
    n_bs = z_post.shape[0]

    mi_xzs = [compute_mi(z_post_t, (z_post_mean_t, z_post_logvar_t)) \
              for z_post_t, z_post_mean_t, z_post_logvar_t in \
              zip(z_post.permute(1, 0, 2), z_post_mean.permute(1, 0, 2), z_post_logvar.permute(1, 0, 2))]
    mi_xz = torch.stack(mi_xzs).sum()

    l_recon = nn.CrossEntropyLoss()
    recon_x = torch.transpose(recon_x, 1, 2)
    l_recon = l_recon(recon_x, x)

    f_mean = f_mean.view((-1, f_mean.shape[-1]))  # [128, 256]
    f_logvar = f_logvar.view((-1, f_logvar.shape[-1]))  # [128, 256]
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean, 2) - torch.exp(f_logvar))

    z_post_var = torch.exp(z_post_logvar)  # [128, 8, 32]
    z_prior_var = torch.exp(z_prior_logvar)  # [128, 8, 32]
    kld_z = 0.5 * torch.sum(z_prior_logvar - z_post_logvar +
                            ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)

    l_recon, kld_f, kld_z = l_recon / batch_size, kld_f / batch_size, kld_z / batch_size

    batch_size, n_frame, z_dim = z_post_mean.size()

    con_loss_c = contras_fn(f_mean, f_mean_c)
    con_loss_m = contras_fn(z_post_mean.view(batch_size, -1), z_post_mean_m.view(batch_size, -1))

    # calculate the mutual infomation of f and z
    mi_fz = torch.zeros((1)).cuda()
    if True:
        _logq_f_tmp = log_density(f.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, batch_size, 1, f_dim),
                                  f_mean.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size, f_dim),
                                  f_logvar.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size,f_dim))
        _logq_z_tmp = log_density(z_post.transpose(0, 1).view(n_frame, batch_size, 1, z_dim),
                                  z_post_mean.transpose(0, 1).view(n_frame, 1, batch_size, z_dim),
                                  z_post_logvar.transpose(0, 1).view(n_frame, 1, batch_size, z_dim))

        _logq_fz_tmp = torch.cat((_logq_f_tmp, _logq_z_tmp), dim=3)
        logq_f = (logsumexp(_logq_f_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * dataset_size))
        logq_z = (logsumexp(_logq_z_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * dataset_size))
        logq_fz = (logsumexp(_logq_fz_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * dataset_size))
        # n_frame x batch_size
        mi_fz = F.relu(logq_fz - logq_f - logq_z).mean()

    loss = l_recon + kld_f * weight_f + kld_z * weight_z + mi_fz * weight_fz
    loss += con_loss_c * weight_c_aug
    loss += con_loss_m * weight_m_aug

    return loss

#parameters
batch_size=32
train_iters=2

def train(loc_emb,g_dropout,alpha,beta,gamma,s_dim,r_dim,hidden_units):
    model=PreDisMove(emb_dim=loc_emb, out_size=loc_emb, num_heads=2, dropout=g_dropout, node_num=TOTAL_POI,
                     s_dim=s_dim,r_dim=r_dim,hidden_units=hidden_units).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99), eps=1e-08, weight_decay=0.00001, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)

    # sort original data
    index_T = {}
    trainT = []
    trainT_ca = []
    trainT_ma = []
    trainU = []
    trainTime = []
    trainCat = []
    for i in range(len(new_trainT)):
        index_T[i] = len(new_trainT[i])
    temp_size = sorted(index_T.items(), key=lambda item: item[1])
    for i in range(len(temp_size)):
        id = temp_size[i][0]
        trainT.append(new_trainT[id])
        trainT_ca.append(new_trainT_ca[id])
        trainT_ma.append(new_trainT_ma[id])
        trainTime.append(new_trainTime[id])
        trainCat.append(new_trainCat[id])
        trainU.append(Train_USER[id])

    # sort for test dataset
    index_T = {}
    testT = []
    testTime = []
    testCat = []
    testU = []
    testT_ca = []
    testT_ma = []
    for i in range(len(new_testT)):
        index_T[i] = len(new_testT[i])
    temp_size = sorted(index_T.items(), key=lambda item: item[1])
    for i in range(len(temp_size)):
        id = temp_size[i][0]
        testT.append(new_testT[id])
        testTime.append(new_testTime[id])
        testCat.append(new_testCat[id])
        testU.append(Test_USER[id])
        testT_ca.append(new_testT_ca[id])
        testT_ma.append(new_testT_ma[id])
    # -----------------------------------
    # epoch
    train_size = len(trainT) % batch_size
    test_size = len(testT) % batch_size
    trainT = trainT + trainT[-(batch_size - train_size):]  # copy data and fill the last batch size
    trainT_ca = trainT_ca + trainT_ca[-(batch_size - train_size):]
    trainT_ma = trainT_ma + trainT_ma[-(batch_size - train_size):]
    trainTime = trainTime + trainTime[-(batch_size - train_size):]
    trainCat = trainCat + trainCat[-(batch_size - train_size):]
    trainU = trainU + trainU[-(batch_size - train_size):]

    testT = testT + testT[-(batch_size - test_size):]  # copy data and fill the last batch size
    testTime = testTime + testTime[-(batch_size - test_size):]
    testCat = testCat + testCat[-(batch_size - test_size):]
    testU = testU + testU[-(batch_size - test_size):]
    testT_ca = testT_ca + testT_ca[-(batch_size - test_size):]
    testT_ma = testT_ma + testT_ma[-(batch_size - test_size):]

    # train
    learning_rate = 0.001  # learning rate
    epoch = 0
    Flag = True
    while (Flag):  # 每一epoch
        epoch += 1
        if epoch == train_iters:
            Flag = False
            break  # end computing

        # train
        model.train()
        print('learning rate', learning_rate)
        step = 0
        epoch_loss = 0
        temp_loss = 0
        Train_Acc_1 = 0
        Train_Acc_5 = 0
        Train_Acc_10 = 0
        temp_acc = 0
        while step < len(trainU) // batch_size:  # 每一step
            start_i = step * batch_size
            batch_x = []
            batch_x_ca = []
            batch_x_ma = []
            batch_time = []
            batch_cat = []
            label = []
            # for test
            input_x = trainT[start_i:start_i + batch_size]
            input_y = trainU[start_i:start_i + batch_size]
            input_x_ca = trainT_ca[start_i:start_i + batch_size]
            input_x_ma = trainT_ma[start_i:start_i + batch_size]
            input_time = trainTime[start_i:start_i + batch_size]
            input_cat = trainCat[start_i:start_i + batch_size]

            for i in input_x:
                #batch_x.append(i[:-1])  # add n-1 pois
                batch_x.append(i[:-1])
                label.append(i[-1])
            for i in input_time:
                batch_time.append(i[:-1])
            for i in input_cat:
                batch_cat.append(i[:-1])
            for i in input_x_ca:
                batch_x_ca.append(i[:-1])
            for i in input_x_ma:
                batch_x_ma.append(i[:-1])

            batch_x = pad_sentence_batch(batch_x, 0)
            batch_x = torch.tensor(batch_x).to(device)
            batch_x_ca = pad_sentence_batch(batch_x_ca, 0)
            batch_x_ca = torch.tensor(batch_x_ca).to(device)
            batch_x_ma = pad_sentence_batch(batch_x_ma, 0)
            batch_x_ma = torch.tensor(batch_x_ma).to(device)

            batch_time = pad_sentence_batch(batch_time, 0)
            batch_time = torch.tensor(batch_time).to(device)
            batch_cat = pad_sentence_batch(batch_cat, 0)
            batch_cat = torch.tensor(batch_cat).to(device)

            poi_time = torch.tensor(poi_time_list).float().to(device)
            poi_cat = torch.tensor(poi_cat_list).float().to(device)

            optimizer.zero_grad()
            f_mean, f_logvar, f, z_post_mean, z_post_logvar, z_post, z_prior_mean, z_prior_logvar, z_prior, recon_x = model(
                G,batch_x,batch_time,batch_cat,poi_time,poi_cat)
            f_mean_c, f_logvar_c, f_c, _, _, _, _, _, _, _ = model(G,batch_x_ca,batch_time,batch_cat,poi_time,poi_cat)
            _, _, _, z_post_mean_m, z_post_logvar_m, z_post_m, _, _, _, _ = model(G,batch_x_ma,batch_time,batch_cat,poi_time,poi_cat)
            loss = cal_loss(batch_x, f_mean, f_logvar, f, z_post_mean, z_post_logvar, z_post, z_prior_mean,
                            z_prior_logvar, z_prior, recon_x,f_mean_c, f_logvar_c, f_c, z_post_mean_m, z_post_logvar_m, z_post_m,
                            s_dim,r_dim,
                            alpha=alpha, gamma=gamma,beta=beta)

            loss.backward()  # retain_graph=True
            epoch_loss += loss.item()
            optimizer.step()

            if (step % 500 == 0):
                print('step：', step, ', loss：', epoch_loss / (500 * batch_size))
            step += 1


        step=0
        while step < len(testU) // batch_size:  # 每一step
            start_i = step * batch_size
            batch_x = []
            batch_x_ca = []
            batch_x_ma = []
            batch_time = []
            batch_cat = []
            label = []
            # for test
            input_x = testT[start_i:start_i + batch_size]
            input_y = testU[start_i:start_i + batch_size]
            input_x_ca = testT_ca[start_i:start_i + batch_size]
            input_x_ma = testT_ma[start_i:start_i + batch_size]
            input_time = testTime[start_i:start_i + batch_size]
            input_cat = testCat[start_i:start_i + batch_size]

            for i in input_x:
                batch_x.append(i[:-1])  # add n-1 pois
                label.append(i[-1])
            for i in input_time:
                batch_time.append(i[:-1])
            for i in input_cat:
                batch_cat.append(i[:-1])
            for i in input_x_ca:
                batch_x_ca.append(i[:-1])
            for i in input_x_ma:
                batch_x_ma.append(i[:-1])

            batch_x = pad_sentence_batch(batch_x, 0)
            batch_x = torch.tensor(batch_x).to(device)
            batch_x_ca = pad_sentence_batch(batch_x_ca, 0)
            batch_x_ca = torch.tensor(batch_x_ca).to(device)
            batch_x_ma = pad_sentence_batch(batch_x_ma, 0)
            batch_x_ma = torch.tensor(batch_x_ma).to(device)

            batch_time = pad_sentence_batch(batch_time, 0)
            batch_time = torch.tensor(batch_time).to(device)
            batch_cat = pad_sentence_batch(batch_cat, 0)
            batch_cat = torch.tensor(batch_cat).to(device)

            poi_time = torch.tensor(poi_time_list).float().to(device)
            poi_cat = torch.tensor(poi_cat_list).float().to(device)

            optimizer.zero_grad()
            f_mean, f_logvar, f, z_post_mean, z_post_logvar, z_post, z_prior_mean, z_prior_logvar, z_prior, recon_x = model(
                G,batch_x,batch_time,batch_cat,poi_time,poi_cat)
            f_mean_c, f_logvar_c, f_c, _, _, _, _, _, _, _ = model(G,batch_x_ca,batch_time,batch_cat,poi_time,poi_cat)
            _, _, _, z_post_mean_m, z_post_logvar_m, z_post_m, _, _, _, _ = model(G,batch_x_ma,batch_time,batch_cat,poi_time,poi_cat)
            loss = cal_loss(batch_x, f_mean, f_logvar, f, z_post_mean, z_post_logvar, z_post, z_prior_mean,
                            z_prior_logvar, z_prior, recon_x,f_mean_c, f_logvar_c, f_c, z_post_mean_m, z_post_logvar_m, z_post_m,
                            s_dim,r_dim,
                            alpha=alpha, gamma=gamma,beta=beta)

            loss.backward()  # retain_graph=True
            epoch_loss += loss.item()
            optimizer.step()

            if (step % 500 == 0):
                print('step：', step, ', loss：', epoch_loss / (500 * batch_size))
            step += 1
        print('epoch is', epoch)
        print('train item', step, epoch_loss)

        if (epoch+1)%2==0:
            model.eval()
            # save the model
            torch.save(model.state_dict(), './pretrains/NYC_'+str(alpha)+'_'+str(beta)+'_'+str(gamma)+'_'+str(loc_emb)+'_'+str(g_dropout)+'_'
                       +str(s_dim)+'_'+str(r_dim)+'_'+str(hidden_units) + '.pth')  # 保存的文件名后缀一般是.pt或.pth



train(loc_emb=256,g_dropout=0,alpha=1,beta=1,gamma=0.1,s_dim=256,r_dim=32,hidden_units=300)

