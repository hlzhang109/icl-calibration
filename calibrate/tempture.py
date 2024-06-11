import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
from utils import *
from losses import ECELoss

def temperature_scale(logits, temperature):
    """
    Perform temperature scaling on logits
    """
    # Expand temperature to match the size of logits
    temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
    return logits / temperature

def train_temperature(params, train_sentences, train_labels, num_shots=0, batch=128, epochs=50, lr=0.01):
    select_string = lambda x, idx: [x[id] for id in idx]
    temperature = torch.ones(1).cuda() * 1.0
    temperature.requires_grad = True
    nll_criterion = nn.CrossEntropyLoss().cuda()
    
    ece_criterion = ECELoss().cuda()
    optimizer = torch.optim.Adam([temperature], lr=lr)
    
    train_labels = np.array(train_labels)
    eces_before, eces_after, nll_before, nll_after = [], [], [], []
    for e in range(epochs):
        indices = np.arange(0, len(train_sentences))
        random.shuffle(indices)
        indices_val, index_train = indices[:batch], indices[batch:]
        prompt, prompt_labels = random_sampling(select_string(train_sentences,index_train), train_labels[index_train], num_shots)
        valid, labels = select_string(train_sentences,indices_val), train_labels[indices_val]

        raw_resp_test = get_model_response(params, prompt, prompt_labels, valid)
        all_label_probs = get_label_logits(params, raw_resp_test, prompt, prompt_labels, valid)
        logits = torch.from_numpy(all_label_probs).cuda().float()
        labels = torch.from_numpy(labels).cuda()
        
        loss = nll_criterion(temperature_scale(logits, temperature), labels)
        
        before_temperature_ece = ece_criterion(logits, labels).item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        after_temperature_ece = ece_criterion(logits, labels, temperature.item()).item()
        
        eces_before.append(before_temperature_ece)
        nll_before.append(nll_criterion(logits,labels).item())
        eces_after.append(after_temperature_ece)
        nll_after.append(nll_criterion(temperature_scale(logits, temperature), labels).item())
    
    eces_before, eces_after = np.array(eces_before), np.array(eces_after)
    nll_before, nll_after = np.array(nll_before), np.array(nll_after)
    print('Optimal temperature: %.3f' % temperature.item())
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (nll_before[-1], eces_before[-1]))
    print('After temperature - NLL: %.3f, ECE: %.3f' % (nll_after[-1], eces_after[-1]))
    return temperature.item()

def train_temperature_specific_prompts(params, train_sentences, train_labels, prompt, prompt_labels, batch=32, epochs=50, lr=0.01):
    select_string = lambda x, idx: [x[id] for id in idx]
    temperature = torch.ones(1).cuda() * 1.0
    temperature.requires_grad = True
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = ECELoss().cuda()
    optimizer = torch.optim.Adam([temperature], lr=lr)
    
    train_labels = np.array(train_labels)
    eces_before, eces_after, nll_before, nll_after = [], [], [], []
    for e in range(epochs):
        indices = np.arange(0, len(train_sentences))
        random.shuffle(indices)
        indices_val, index_train = indices[:batch], indices[batch:]
        valid, labels = select_string(train_sentences,indices_val), train_labels[indices_val]

        raw_resp_test = get_model_response(params, prompt, prompt_labels, valid)
        all_label_probs = get_label_logits(params, raw_resp_test, prompt, prompt_labels, valid)
        logits = torch.from_numpy(all_label_probs).cuda().float()
        labels = torch.from_numpy(labels).cuda()
        
        loss = nll_criterion(temperature_scale(logits, temperature), labels)
        
        before_temperature_ece = ece_criterion(logits, labels).item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        after_temperature_ece = ece_criterion(logits, labels, temperature.item()).item()
        
        eces_before.append(before_temperature_ece)
        nll_before.append(nll_criterion(logits,labels).item())
        eces_after.append(after_temperature_ece)
        nll_after.append(nll_criterion(temperature_scale(logits, temperature), labels).item())
    
    eces_before, eces_after = np.array(eces_before), np.array(eces_after)
    nll_before, nll_after = np.array(nll_before), np.array(nll_after)
    print('Optimal temperature: %.3f' % temperature.item())
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (nll_before[-1], eces_before[-1]))
    print('After temperature - NLL: %.3f, ECE: %.3f' % (nll_after[-1], eces_after[-1]))
    return temperature.item()