from torchfm.layer import CrossNetwork, FeaturesEmbedding
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np

# 定义简单的分类模型MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class MultiLayerPerceptron1(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=False):
        super().__init__()
        layers = list()
        self.mlps = nn.ModuleList()
        self.out_layer = output_layer
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            # layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
            self.mlps.append(nn.Sequential(*layers))
            layers = list()
        if self.out_layer:
            self.out = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        for layer in self.mlps:
            x = layer(x)
        if self.out_layer:
            x = self.out(x)
        return x

class controller_mlp(nn.Module):
    def __init__(self,input_dim, nums):
        super().__init__()
        self.inputdim = input_dim
        # self.mlp = MultiLayerPerceptron1(input_dim=self.inputdim, embed_dims=[self.inputdim//2,nums], output_layer=False, dropout=0.2)
        self.mlp = MLP(input_dim=self.inputdim, hidden_dim=self.inputdim//2, output_dim=nums)
        # self.mlp = MLP(input_dim=self.inputdim, hidden_dim=nums, output_dim=nums)
        # self.mlp = MLP(input_dim=self.inputdim, hidden_dim=self.inputdim//2, output_dim=nums)
        self.weight_init(self.mlp)
    
    def forward(self, emb_fields):
        # input_mlp = emb_fields.flatten(start_dim=1).float()
        output_layer = self.mlp(emb_fields)
        return torch.softmax(output_layer, dim=1)

    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


class AdaFS_soft(nn.Module): 
    def __init__(self,input_dims,inputs_dim,num):
        super().__init__()
        self.num = num
        self.embed_dim = input_dims
        self.mlp = MLP(input_dim=self.embed_dim, hidden_dim=input_dims//2, output_dim=num)
        self.controller = controller_mlp(input_dim=self.embed_dim, nums=self.num)
        self.inputs_dim = inputs_dim


    def forward(self, field):
          self.weight = self.controller(field)
          self.dims = []
          for k,v in self.inputs_dim.items():
              self.dims.append(v)
          offsets = np.cumsum(self.dims).tolist()
          offsets = [0] + offsets
          field1 = field.clone()
          for i in range(len(offsets)-1):
              field1[:, offsets[i]:offsets[i+1]] = field[:, offsets[i]:offsets[i+1]] * self.weight[:,i].unsqueeze(1)
        #   res = self.mlp(field1)
          return field1


class SelectionNetwork(nn.Module):
    def __init__(self, input_dims,num):
        super(SelectionNetwork, self).__init__()
        self.num = num
        #  得到评分
        self.mlp =  MLP(input_dim=input_dims, hidden_dim=input_dims//2, output_dim=num)
        self.weight_init(self.mlp)
                                        
    def forward(self, input_mlp):
        output_layer = self.mlp(input_mlp)
        return torch.softmax(output_layer, dim=1)
    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


class MvFS_Controller(nn.Module):
    # embed_dims是字段的长度，也就是有多少字段，每个元素表示对应字段的重要性
    def __init__(self, input_dim, nums, num_selections):
        super().__init__()
        self.inputdim = input_dim
        self.nums=nums
        # 专家评估网络的数量n
        self.num_selections = num_selections
        self.T = 1
        self.b = 5
        # 决定每个子网络的重要性
        self.gate = nn.Sequential(nn.Linear(self.nums * num_selections , num_selections))
        # 子网络
        self.SelectionNetworks = nn.ModuleList([SelectionNetwork(input_dim,self.nums) for i in range(num_selections)])


    def forward(self, inputs):
        # self.T = epoch
        input_mlp = inputs
        importance_list= []
        for i in range(self.num_selections):
            importance_vector = self.SelectionNetworks[i](input_mlp)
            importance_list.append(importance_vector)
        gate_input = torch.cat(importance_list, 1)
        selection_influence = self.gate(gate_input)
        selection_influence = torch.sigmoid(selection_influence)
        scores = None
        for i in range(self.num_selections):
            score = torch.mul(importance_list[i], selection_influence[:,i].unsqueeze(1))
            if i == 0 :
                scores = score
            else:
                scores = torch.add(scores, score)
        scores = scores / self.num_selections
        scores = torch.softmax(scores, dim=1)
        # all_scores = sum(scores)
        # scores = scores / all_scores
        return scores

class MvFS_MLP(nn.Module): 
    def __init__(self,input_dims, nums,inputs_dim, num_selections):
        super().__init__()
        self.nums = nums
        self.inputs_dim = inputs_dim
        self.input_dim = input_dims
        self.num_selections = num_selections
        self.mlp = MLP(input_dim=self.input_dim, hidden_dim=self.input_dim//2, output_dim=5)
        self.controller = MvFS_Controller(input_dim=self.input_dim,nums=self.nums, num_selections= self.num_selections)
        self.UseController = True
        self.useBN = False
        self.BN = nn.BatchNorm1d(self.input_dim)
        self.weight = 0
        self.stage = 1 
       

    def forward(self, input):
        field = input
        self.weight = self.controller(input)
        # 进行按权重分配权重
        field1 = field.clone()
        self.dims = []
        for k,v in self.inputs_dim.items():
            self.dims.append(v)
        offsets = np.cumsum(self.dims).tolist()
        offsets = [0] + offsets
        field1 = field.clone()
        for i in range(len(offsets)-1):
            field1[:, offsets[i]:offsets[i+1]] = field[:, offsets[i]:offsets[i+1]] * self.weight[:,i].unsqueeze(1)
  
        input_mlp = field1
        res = self.mlp(input_mlp)

        # return res
        return input_mlp


class FinedNetwork(nn.Module):
    def __init__(self, input_dims):
        super(FinedNetwork, self).__init__()
        # self.mlp =  MLP(input_dim=input_dims, hidden_dim=input_dims, output_dim=input_dims)
        self.mlp =  MultiLayerPerceptron1(input_dim=input_dims,embed_dims=[input_dims], dropout=0.2,)
        # self.mlp = nn.Linear(input_dims,input_dims)
        self.weight_init(self.mlp)
                                        
    def forward(self, input_mlp):
        output_layer = self.mlp(input_mlp)
        return torch.softmax(output_layer, dim=1)
    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    
class FinedController(nn.Module):
    def __init__(self, inputs_dim):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.fined_length = len(inputs_dim)
        print(self.fined_length)
        self.fined_class = nn.ModuleList(FinedNetwork(input_dims=inputs_dim[i]) for i in range(self.fined_length))

    def forward(self, inputs):
        # input_mlp = inputs
        importance_list= []
        offsets = np.cumsum(self.inputs_dim).tolist()
        offsets = [0] + offsets
        for i in range(len(offsets)-1):
            importance_vector = self.fined_class[i](inputs[:, offsets[i]:offsets[i+1]])
            importance_list.append(importance_vector)
        return torch.cat(importance_list, 1)
    
    
class AFS_ZXM(nn.Module):
    def __init__(self, input_dims, inputs_dim):
        super().__init__()
        # self.num = nums
        self.inputs_dim = inputs_dim
        self.get_dims()
        self.embed_dim = input_dims
        # self.mlp =MLP(input_dim=self.embed_dim, hidden_dim=self.embed_dim//2, output_dim=5)
        self.Finedcontroller = FinedController(self.dims)
        self.Adacontroller = controller_mlp(input_dim=self.embed_dim, nums=len(self.dims))

        self.weight = 0


    def get_dims(self):
        self.dims = []
        for k,v in self.inputs_dim.items():
            self.dims.append(v)
    
    def forward(self, field):
          weight = self.Finedcontroller(field)
          input_mlp = field  * weight
          weight1 = self.Adacontroller(input_mlp)
        #   weight1 = self.Adacontroller(weight)
          offsets = np.cumsum(self.dims).tolist()
          offsets = [0] + offsets
          field1 = field.clone()
          for i in range(len(offsets)-1):
              field1[:, offsets[i]:offsets[i+1]] = field[:, offsets[i]:offsets[i+1]] * weight1[:,i].unsqueeze(1)
        #   res = self.mlp(field1)
        #   res = self.mlp(input_mlp)
        #   return res
          return field1