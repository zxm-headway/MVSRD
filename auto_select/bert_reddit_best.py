import argparse
import os
from pickletools import optimize
import random
import string
import time
from math import log
import numpy as np
import scipy.sparse as sp
from nltk.corpus import stopwords
from stanfordcorenlp import StanfordCoreNLP
from torch import Tensor, nn
from tqdm import tqdm
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
import pickle as pkl
import json
from tools import utils
from ordered_set import OrderedSet
from sklearn.model_selection import KFold
import base_model_best as BS
# from torchfm.layer import MultiLayerPerceptron
# from transformers import (AdamW, get_cosine_schedule_with_warmup,get_cosine_with_hard_restarts_schedule_with_warmup)


# 将一个批次中的多个数据项按特定规则组合并填充（pad），以便它们可以被批量处理。
def pad_collate_reddit(batch):
    target = [item[0] for item in batch]
    tweet = [item[1] for item in batch]
    lens = [len(x) for x in tweet]
    feature = [item[2] for item in batch]
    tweet = nn.utils.rnn.pad_sequence(tweet, batch_first=True, padding_value=0)
    target = torch.tensor(target)
    lens = torch.tensor(lens)
    feature = torch.stack(feature)
    return [target, tweet, lens, feature]



def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embed_size", type=int, default=768)
    parser.add_argument("--max_len", default=200, type=int)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--seed", default=24, type=int)
    parser.add_argument("--classnum", default=5, type=int)
    parser.add_argument("--use_pretrain", default=True, type=bool)
    return parser.parse_args(args)


class RedditDataset(Dataset):
    def __init__(self, labels, tweets, days=200):
        super().__init__()
        self.labels = labels
        self.tweets = tweets # 预训练的嵌入向量.
        # days代表是POST的数量还是其他东西？？？,用户发表的帖子的嵌入
        self.days = days

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        labels = torch.tensor(self.labels['labels'].iloc[item], dtype=torch.long)
        feature = torch.tensor(self.labels.iloc[item,:-1].values,dtype=torch.float32)
        if self.days > len(self.tweets[item]):
            tweets = torch.tensor(self.tweets[item], dtype=torch.float32)
        else:
            tweets = torch.tensor(self.tweets[item][:self.days], dtype=torch.float32)
            print('进行了截取')
        return [labels, tweets,feature]


# 增强模型对输入数据的某些部分的关注度
class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first # 指示输入数据的第一个维度是否是批次大小。
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        # self.SelfAttention = SelfAttention(hidden_size, batch_first=True)
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        

        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)  # (batch_size, hidden_size, 1)
                            )

        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)
        mask = torch.ones(attentions.size(), requires_grad=True).cuda()
        
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                # 排除填充为零的部分的注意力
                mask[i, l:] = 0
        # 这块目的去除填充为零的部分,然后重新计算每个帖子的权重
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row
        attentions = masked.div(_sums)
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze()
        return representations, attentions

class Attention1(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention1, self).__init__()
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size , hidden_size ))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size , 1))
        self.batch_first = batch_first
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs,lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]
        u = torch.tanh(torch.matmul(inputs, self.w_omega))         #[batch, seq_len, hidden_dim]
        att = torch.matmul(u, self.u_omega)                   #[batch, seq_len, 1]
        attentions = torch.softmax(F.relu(att.squeeze()), dim=-1)
        mask = torch.ones(attentions.size(), requires_grad=True).cuda()
        
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                # 排除填充为零的部分的注意力
                mask[i, l:] = 0
        # 这块目的去除填充为零的部分,然后重新计算每个帖子的权重
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row
        attentions = masked.div(_sums)
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze()
        return representations, attentions



class SelfAttention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first # 指示输入数据的第一个维度是否是批次大小。
        self.W_query = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.W_key   = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.W_value = nn.Parameter(torch.rand(hidden_size, hidden_size))

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        keys = torch.bmm(inputs, self.W_key.unsqueeze(0).expand(inputs.size(0), *self.W_key.size()))
        queries = torch.bmm(inputs, self.W_query.unsqueeze(0).expand(inputs.size(0), *self.W_query.size()))
        values = torch.bmm(inputs, self.W_value.unsqueeze(0).expand(inputs.size(0), *self.W_value.size()))
        attn_scores = torch.bmm(queries, keys.transpose(1, 2))
        attn_scores = attn_scores / (self.hidden_size ** 0.5)
        attentions= torch.softmax(attn_scores, dim=-1)
        weighted = torch.bmm(attentions, values)
        return weighted

# 加入通用文本特征选择模块

class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layer, dropout) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size,num_layers=num_layer, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = Attention(hidden_size * 2 ,batch_first=True)
        # self.attention = Attention1(hidden_size * 2 ,batch_first=True)

        # self.attention = SelfAttention(hidden_size * 2 ,batch_first=True)
  
    def forward(self, inputs, x_len):
        inputs = self.dropout(inputs)
        x = nn.utils.rnn.pack_padded_sequence(inputs, x_len, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm(x)
        # todo 进行一次注意力计算，在有两种方法。一种数普通的注意力机制，一种是加了控制器的选择策略
        x, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # x1 = torch.mean(x, 1)
        # x1 = torch.sum(x, 1)
        representations, attentions = self.attention(x, lengths)  # (batch_size, hidden_size)
        # representations, attentions = self.attention(x)  # (batch_size, hidden_size)
        # return  x1, x
        return representations, attentions
    
  
class MyLSTMATT(nn.Module):
    def __init__(self, features_dic, class_num=5,engine_dim=100, embedding_dim=768, hidden_dim=128, lstm_layer=2, dropout=0.5):
        super(MyLSTMATT, self).__init__()
        self.embedding_dim = embedding_dim
        self.engine_dim  = engine_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        # 多层感知机
        self.fc_1 = nn.Linear(hidden_dim*2+self.engine_dim, hidden_dim)
        # self.fc_1 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.fc_2 = nn.Linear(hidden_dim, class_num)

        self.historic_model = BiLSTM(self.embedding_dim, self.hidden_dim, lstm_layer, dropout)
        self.controller = BS.AFS_ZXM(input_dims=self.engine_dim,inputs_dim=features_dic)
        # self.controller = BS.AdaFS_soft(input_dims=self.engine_dim,inputs_dim=features_dic,num=5)
        # self.controller = BS.MvFS_MLP(input_dims=self.engine_dim, nums=5,inputs_dim=features_dic, num_selections=6)

    # 预测每个用户的分类
    def get_pred(self, feat,features):
        bert_feature = feat
        engine_feature  = features

        engine_feature = self.controller(engine_feature)
        # 将bert_feature与engine_feature进行拼接
        all_feature = torch.cat((bert_feature,engine_feature),dim=1)
        all_feature = self.dropout(all_feature)

        feat = self.fc_1(all_feature)
        # feat = self.fc_1(bert_feature)
        return self.fc_2(feat)

    def forward(self, tweets, lengths, labels,features):

        # print(tweets.shape,'tweets.shape')
        h, _ = self.historic_model(tweets, lengths)

        # print(h.shape)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        e = self.get_pred(h,features)
        return e
    

#  上述模型已完成，下面是训练和测试的代码


def read_reddit_embeddings():
    # Load the reddit embeddings
    with open('../data/bert_embeddings.pkl', 'rb') as f:
        reddit_embeddings = pkl.load(f)
    return reddit_embeddings



def train(args):
    bert_embeddings = read_reddit_embeddings()
    labels = []
    posts = [] 
    for i in range(len(bert_embeddings)):
        labels.append(bert_embeddings[i]['label'])
        posts.append(bert_embeddings[i]['embeddings'])

    features = pd.read_csv('../data_analy/feature.csv')

    features_dic = {
        'pos':36,
        'tidif':50,
        'nrc':10,
        'sui':4
    }

    features_dim = features.shape[1]
    labels = pd.DataFrame(labels, columns=['labels'])
    features_labels = pd.concat([features,labels],axis=1)

    # 开始划分数据集，进行70%的训练集和30%的测试集
    train_data, test_data, train_labels, test_labels = train_test_split(posts, features_labels, test_size=0.2, random_state=args.seed,stratify=features_labels['labels'].values)
    # train_data, val_data,train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.25, random_state=args.seed, stratify=train_labels['labels'].values) 
    test_data, val_data,test_labels, val_labels = train_test_split(test_data, test_labels, test_size=0.5, random_state=args.seed, stratify=test_labels['labels'].values) 
    # print(train_data)

    # print(train_labels)

    # 将数据转换为Dataset
    train_dataset = RedditDataset(train_labels, train_data)
    val_dataset = RedditDataset(val_labels, val_data)
    test_dataset = RedditDataset(test_labels, test_data)

    # 将数据转换为DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_reddit)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_reddit)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_reddit)

    # 初始化模型
    model = MyLSTMATT(features_dic = features_dic,class_num=args.classnum, engine_dim = features_dim,embedding_dim=args.embed_size, hidden_dim=args.hidden_size, lstm_layer=2, dropout=args.dropout)
    model = model.cuda()
    # criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_f1 = 0

    
    if args.use_pretrain:
        print("Using pre-trained model")
        model.load_state_dict(torch.load('best_model.pth'))
    else:
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for i, (labels, tweets, lengths,features) in enumerate(tqdm(train_loader)):
                

                # unique, counts = np.unique(labels, return_counts=True)
                # class_counts = dict(zip(unique, counts))
                # print("Train class counts:", class_counts)

                labels = labels.cuda()
                tweets = tweets.cuda()
                features = features.cuda()
                optimizer.zero_grad()
                outputs = model(tweets, lengths, labels,features)
                loss = utils.loss_function(outputs, labels, loss_type='ce', expt_type=args.classnum, scale=2)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # print(f"Epoch {epoch} Loss: {total_loss}")
        model.eval()
        val_preds = []
        val_labels = []
        for i, (labels, tweets, lengths,features) in enumerate(tqdm(val_loader)):
            labels = labels.cuda()
            tweets = tweets.cuda()
            features = features.cuda()
            outputs = model(tweets, lengths, labels,features)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

        # print(val_preds)
        # print(val_labels)

        # M = utils.gr_metrics(val_preds, fin_targets)

        M = utils.gr_metrics(val_preds, val_labels)

        # 计算accuracy
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        accuracy = np.mean(val_preds == val_labels)
        # print(f"Epoch {epoch} Validation Accuracy: {accuracy}")

        # f1 = accuracy
        print(f"Epoch {epoch} Validation Accuracy: {accuracy}")
        f1 = M[2]
        # print(f"Epoch {epoch} Validation GP: {M[0]} GR: {M[1]} FS: {M[2]}")
        if f1 >= best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_model.pth')

    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_preds = []
    test_labels = []

    for i, (labels, tweets, lengths,features) in enumerate(tqdm(test_loader)):
        labels = labels.cuda()
        tweets = tweets.cuda()
        features = features.cuda()
        outputs = model(tweets, lengths, labels,features)
        preds = torch.argmax(outputs, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

    fin_outputs =  np.hstack(test_preds)
    fin_targets =  np.hstack(test_labels)
    M = utils.gr_metrics(fin_outputs, fin_targets)
    accuracy = np.mean(fin_outputs == fin_targets)
    print(f" test accuracy: {accuracy}")
    print(f" test GP: {M[0]} GR: {M[1]} FS: {M[2]}")

def set_seed(args):
    """
    :param args:
    :return:
    """
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    set_seed(args)
    train(args)




    
if __name__ == '__main__':
    main()

