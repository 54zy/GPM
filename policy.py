import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from copy import deepcopy
import torch.optim as optim
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from scipy.optimize import minimize

def hard_sample(logits, dim=-1):
    y_soft = F.softmax(logits, dim=-1)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret, index.squeeze(1)

def gumbel_softmax(logits, temperature=1.0, hard=False,dim = -1):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)  # Gumbel(0, 1) noise
    y = (logits + gumbel_noise) / temperature
    y = F.softmax(y, dim=-1)  # Gumbel-Softmax
    if hard:
        _, index = y.max(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(-1, index, 1.0)
        y = (y_hard - y).detach() + y
    index = y.max(dim, keepdim=True)[1]
    return y,index.squeeze(1)

def cal_sq_dis(feature,y_soft,inf_mask):
    
    if(feature.size()[0] < feature.size()[1]):
        feature  = feature.t()
    avg_feature = torch.matmul(y_soft, feature)
    feature  = feature.t()
    squared_distances = (feature[0] - avg_feature)**2
    squared_distances += inf_mask
    return squared_distances

def split_number(x, n):
    base_value = x // n  
    remainder = x % n    
    result = [base_value] * (n - remainder) + [base_value + 1] * remainder
    return result

def process(i,obs_layer,actor_layer,state,action_mask,feature,meta):
    hidden_state = obs_layer(state)
    logits= actor_layer(hidden_state)
    inf_mask = torch.clamp(torch.log(action_mask.float()),
                        min=torch.finfo(torch.float32).min)
    logits = logits + inf_mask
    y_soft = F.softmax(logits, dim=-1)
    squared_distances = cal_sq_dis(feature,y_soft,inf_mask)
    train_mask, actions = gumbel_softmax(squared_distances)
    if meta == False :
        actions[actions == (state.shape[1]-1)] = -state.shape[1] *i + i
        return train_mask[:, :-1], actions
    return train_mask, actions

class MyNet(nn.Module):
    def __init__(self, state_dim, action_dim, op_num,n_latent_var=256):
        super().__init__()
        # actor
        self.op_num = op_num
        if op_num==0:
            self.obs_layer = nn.Linear(state_dim, n_latent_var)
            self.actor_layer = nn.Sequential(
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim)
            )
        else:
            ids= split_number(state_dim, op_num+1)
            self.ids = ids
            self.obs_layer = nn.Linear(ids[0], n_latent_var)
            self.actor_layer = nn.Sequential(
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, ids[0])
            )

            self.obs_layers_op = nn.ModuleList([
                        nn.Linear(ids[i+1]+1, n_latent_var) for i in range(op_num)
                    ])
            self.actor_layers_op = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(n_latent_var, n_latent_var),
                    nn.Tanh(),
                    nn.Linear(n_latent_var, ids[j+1]+1)
                ) for j in range(op_num)
            ])


        #input: state(candidate question phi) action_mask（0 for chosen，1 for others）
    def forward(self, state, action_mask,feature):
        if(self.op_num==0):
            hidden_state = self.obs_layer(state)
            logits= self.actor_layer(hidden_state)
            inf_mask = torch.clamp(torch.log(action_mask.float()),
                                min=torch.finfo(torch.float32).min)
            logits = logits + inf_mask
            y_soft = F.softmax(logits, dim=-1)
            squared_distances = cal_sq_dis(feature,y_soft,inf_mask)
            #train_mask,actions = hard_sample(squared_distances)
            train_mask, actions = gumbel_softmax(squared_distances)
            return train_mask, actions
        else:
            split_state = [state[:, sum(self.ids[:i]):sum(self.ids[:i+1])] for i in range(len(self.ids))]
            split_action = [action_mask[:, sum(self.ids[:i]):sum(self.ids[:i+1])] for i in range(len(self.ids))]
            split_feature = [feature[:,sum(self.ids[:i]):sum(self.ids[:i+1])] for i in range(len(self.ids))]
            return_actions = []
            meta_mask,meta_action = process(0,self.obs_layer,self.actor_layer,split_state[0],split_action[0],split_feature[0],True)
            return_mask = meta_mask
            return_actions.append(meta_action)
            id = self.ids[0]

            column_of_zeros = torch.zeros((state.shape[0], 1)).to(device)
            column_of_one = torch.ones((state.shape[0], 1)).to(device)
            zero_feature = torch.tensor([[0.0]]).to(device)
            for i in range(self.op_num):
                op_feature = torch.cat((split_feature[i+1], zero_feature), dim=1)
                op_ac = torch.cat((split_action[i+1], column_of_one), dim=1)
                op_state = torch.cat((split_state[i+1], column_of_zeros), dim=1)
                op_mask,op_action=process(i+1,self.obs_layers_op[i],self.actor_layers_op[i],op_state,op_ac,op_feature,False)
                return_mask = torch.cat((return_mask,op_mask), dim=1)
                return_actions.append(op_action+id)
                id = id + self.ids[i+1]
            return return_mask,return_actions

class MyModel:
    def __init__(self, state_dim, action_dim, lr, betas,op_num):
        self.lr = lr 
        self.betas = betas
        self.policy = MyNet(state_dim, action_dim,op_num).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.95)
        self._alphas = []
        for n, p in self.policy.named_parameters():
            self._alphas.append((n, p))
    def alphas(self):
        for n, p in self._alphas:
            yield p
    def update(self, l_val_t,l_train_p,l_train_s,epl,lw):
        self.optimizer.zero_grad()
        dw = torch.autograd.grad(l_val_t, self.alphas(),retain_graph=True)
        dalpha_pos = torch.autograd.grad(l_train_p, self.alphas(),retain_graph=True)
        dalpha_neg = torch.autograd.grad(l_train_s, self.alphas(),retain_graph=True)
        hessian = [w - lw * (p-n) / 2.*epl for w, p, n in zip(dw,dalpha_pos, dalpha_neg)]
        params = [param for param in self.policy.parameters()]

        for param, h in zip(params, hessian):
            param.grad = h
        self.optimizer.step()

def main():
    pass


if __name__ == '__main__':
    main()

