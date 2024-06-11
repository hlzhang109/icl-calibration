import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
from utils import *
from losses import ECELoss


def tune_temp(logits, labels, binary_search=True, lower=0.01, upper=5.0, eps=0.0001):

    logits = torch.FloatTensor(logits)
    labels = torch.LongTensor(labels)
    t_guess = torch.FloatTensor([0.5*(lower + upper)]).requires_grad_()

    while upper - lower > eps:
        if torch.autograd.grad(F.cross_entropy(logits / t_guess, labels), t_guess)[0] > 0:
            upper = 0.5 * (lower + upper)
        else:
            lower = 0.5 * (lower + upper)
        t_guess = t_guess * 0 + 0.5 * (lower + upper)

    t = min([lower, 0.5 * (lower + upper), upper], key=lambda x: float(F.cross_entropy(logits / x, labels)))
    return t

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
    eces_before, eces_after, nll_before, nll_after, all_logits, all_labels = [], [], [], [], None, None
    for e in range(epochs):
        indices = np.arange(0, len(train_sentences))
        random.shuffle(indices)
        indices_val, index_train = indices[:batch], indices[batch:]
        prompt, prompt_labels = random_sampling(select_string(train_sentences,index_train), train_labels[index_train], num_shots)
        valid, labels = select_string(train_sentences,indices_val), train_labels[indices_val]

        raw_resp_test = get_model_response(params, prompt, prompt_labels, valid)
        all_label_probs = get_label_logits(params, raw_resp_test, prompt, prompt_labels, valid)
        
        logits = torch.from_numpy(all_label_probs).float()
        labels = torch.from_numpy(labels)
        
        all_logits = logits if all_logits is None else torch.cat([all_logits, logits], dim=0)
        all_labels = labels if all_labels is None else torch.cat([all_labels, labels], dim=0)

    temperature = tune_temp(all_logits, all_labels)
    eces_before = ece_criterion(logits, labels).item()
    nll_before = nll_criterion(logits, labels)
    eces_after = ece_criterion(logits, labels, temperature).item()
    nll_after = nll_criterion(logits / temperature, labels)
    print('Optimal temperature: %.3f' % temperature)
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (nll_before, eces_before))
    print('After temperature - NLL: %.3f, ECE: %.3f' % (nll_after, eces_after))
    return temperature

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
        logits = torch.from_numpy(all_label_probs).float()
        labels = torch.from_numpy(labels)

        all_logits = logits if all_logits is None else torch.cat([all_logits, logits], dim=0)
        all_labels = labels if all_labels is None else torch.cat([all_labels, labels], dim=0)

    temperature = tune_temp(all_logits, all_labels)
    eces_before = ece_criterion(logits, labels).item()
    nll_before = nll_criterion(logits, labels)
    eces_after = ece_criterion(logits, labels, temperature).item()
    nll_after = nll_criterion(logits / temperature, labels)
    print('Optimal temperature: %.3f' % temperature)
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (nll_before, eces_before))
    print('After temperature - NLL: %.3f, ECE: %.3f' % (nll_after, eces_after))
    return temperature