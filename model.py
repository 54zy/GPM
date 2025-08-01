from collections import namedtuple
from torch.distributions import Categorical
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
def find_nonzeros(tensor):
    nz_idx = tensor.nonzero()  
    nz_val = tensor[nz_idx[:,0], nz_idx[:,1]]  
    return nz_idx, nz_val

def pick_random_sample(input_mask,n_query,n_question):
    if n_query==-1:
        return input_mask.detach().clone()
    train_mask = torch.zeros(input_mask.shape[0], n_question).long().to(device)
    actions = torch.multinomial(input_mask.float(), n_query, replacement=False)
    train_mask = train_mask.scatter(dim=1, index=actions, value=1)
    return train_mask

def get_inputs(batch):
    input_labels = batch['input_labels'].to(device).float()
    input_mask = batch['input_mask'].to(device)
    input_ans = None
    return input_labels, input_ans, input_mask

def get_outputs(batch):
    output_labels, output_mask = batch['output_labels'].to(
        device).float(), batch['output_mask'].to(device)  # B,948
    return output_labels, output_mask

def compute_loss(output, labels, mask, reduction= True):

    loss_function = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_function(output, labels) * mask
    if reduction:
        return loss.sum()/mask.sum()
    else:
        return loss.sum()
    
def compute_seq_loss(output, labels_S, labels_Q, mask, student_emb, reduction = 'none'):
    B,n = mask.size()
    
    loss_function = nn.BCEWithLogitsLoss(reduction='none')
    logitS = (loss_function(output, labels_S) * mask).sum()
    logitQ = loss_function(output, labels_Q).sum()

    grad_s = torch.autograd.grad(logitS, student_emb, create_graph=True)[0]
    grad_q = torch.autograd.grad(logitQ, student_emb, create_graph=True)[0]
    non_zero_counts = mask.sum(dim=-1, keepdim=True) + 1e-8
    ave_grad_s = grad_s / non_zero_counts
    ave_grad_q = grad_q / n

    grad_diff = ave_grad_s - ave_grad_q
    del grad_s, grad_q, ave_grad_s, ave_grad_q
    loss = torch.norm(grad_diff, p=2) / B
    return loss
    
    

def normalize_loss(output, labels, mask):
    loss_function = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_function(output, labels) * mask
    count = mask.sum(dim =-1)+1e-8#N,1
    loss = 10. * torch.sum(loss, dim =-1)/count
    return loss.sum()

class MAMLModel(nn.Module):
    def __init__(self, n_question,question_dim =1,dropout=0.2, sampling='active', n_query=10,emb = None,tp='irt'):
        super().__init__()
        self.n_query = n_query
        self.sampling = sampling
        self.sigmoid = nn.Sigmoid()
        self.n_question = n_question
        self.question_dim = question_dim
        self.tp = tp
        if tp == 'irt':
            self.question_difficulty = nn.Parameter(torch.zeros(question_dim,n_question))     
        else:
            self.prednet_input_len = emb.shape[1]
            self.prednet_len1, self.prednet_len2 = 128, 64  # changeable
            self.kn_emb = emb
            self.k_difficulty = nn.Parameter(torch.zeros(n_question,self.prednet_input_len))
            self.e_discrimination = nn.Parameter(torch.full((n_question,1), 0.5))
            self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
            self.drop_1 = nn.Dropout(p=0.5)
            self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
            self.drop_2 = nn.Dropout(p=0.5)
            self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
        
    def reset(self, batch):
        input_labels, _, input_mask = get_inputs(batch)
        obs_state = ((input_labels-0.5)*2.)  # B, 948
        train_mask = torch.zeros(
            input_mask.shape[0], self.n_question).long().to(device)
        env_states = {'obs_state': obs_state, 'train_mask': train_mask,
                      'action_mask': input_mask.clone()}
        return env_states
    
    def step(self, env_states):
        obs_state,  train_mask = env_states[
            'obs_state'], env_states['train_mask']
        state = obs_state*train_mask  # B, 948
        return state

    def pick_sample(self,sampling, config):
        if sampling == 'random':
            train_mask = pick_random_sample(
                config['available_mask'], self.n_query, self.n_question)
            config['train_mask'] = train_mask
            return train_mask

        elif sampling == 'active':
            student_embed = config['meta_param']
            n_student = len(config['meta_param'])
            action = self.pick_uncertain_sample(student_embed, config['available_mask'])
            config['train_mask'][range(n_student), action], config['available_mask'][range(n_student), action] = 1, 0
            return action
        
    def forward(self, batch, config):
        # get inputs
        input_labels = batch['input_labels'].to(device).float()
        student_embed = config['meta_param']#
        output = self.compute_output(student_embed)
        train_mask = config['train_mask']
        # compute loss
        if config['mode'] == 'train':
            output_labels, output_mask = get_outputs(batch)
            # meta model parameters 
            output_loss = compute_loss(output, output_labels, output_mask, reduction=False)/len(train_mask)
            # for adapting meta model parameters
            seq_loss = compute_seq_loss(output, output_labels, output_labels, output_mask, student_emb=student_embed)
            # for BEAT subset
            if self.n_query!=-1:
                input_loss = compute_loss(output, input_labels, train_mask, reduction=False)
            else:
                input_loss = normalize_loss(output, input_labels, train_mask)
            return {'loss': output_loss, 'train_loss': input_loss, 'seq_loss':seq_loss, 'output': self.sigmoid(output).detach().cpu().numpy()}
        else:
            input_loss = compute_loss(output, input_labels, train_mask,reduction=False)
            return {'output': self.sigmoid(output).detach().cpu().numpy(), 'train_loss': input_loss}

    def pick_uncertain_sample(self, student_embed, available_mask):
        with torch.no_grad():
            output = self.compute_output(student_embed)
            output = self.sigmoid(output)
            inf_mask = torch.clamp(
                torch.log(available_mask.float()), min=torch.finfo(torch.float32).min)
            scores = torch.min(1-output, output)+inf_mask
            actions = torch.argmax(scores, dim=-1)
            return actions

    def compute_output(self, student_embed):
        if self.tp=='irt':
            output = (student_embed - self.question_difficulty)
        else:
            k_difficulty = self.k_difficulty
            e_discrimination = self.e_discrimination
            kn_emb = self.kn_emb
            student_embed = student_embed.unsqueeze(1)
            os.environ['CUDA_VISIBLE_DEVICES'] = '2'
            input_x = e_discrimination * (student_embed - k_difficulty) *kn_emb.to(device)

            input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
            input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
            output = self.prednet_full3(input_x)
            output = output.squeeze()
        return output
    def difficulty_irt_old(self):
        if self.tp == 'irt':
            return self.question_difficulty
        else:
            return self.k_difficulty
