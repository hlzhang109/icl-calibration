

import numpy as np

from calibrate import utils
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
from utils import *
from losses import ECELoss
import calibration

def cross_entropy_loss(logits, labels):
    probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    loss = -np.sum(np.log(probabilities[np.arange(len(labels)), labels]))
    return loss / len(labels)

def train_temperature(params, train_sentences, train_labels, num_shots=0, batch=128, epochs=50, lr=0.01):
    select_string = lambda x, idx: [x[id] for id in idx]    
    ece_criterion = ECELoss()
    
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

    all_logits = all_logits.cpu().numpy()
    all_labels = all_labels.cpu().numpy()

    calibrator = calibration.PlattBinnerMarginalCalibrator(num_calibration=all_labels.shape[0], num_bins=10)
    calibrator.train_calibration(all_logits, all_labels)
    eces_before = ece_criterion(all_logits, all_labels).item()
    nll_before = cross_entropy_loss(all_logits, all_labels)
    
    logits_calibrated = calibrator.calibrate(all_logits)
    eces_after = ece_criterion(logits_calibrated, all_labels).item()
    nll_after = cross_entropy_loss(logits_calibrated, all_labels)
    print('Before calibrated - NLL: %.3f, ECE: %.3f' % (nll_before, eces_before))
    print('After calibrated - NLL: %.3f, ECE: %.3f' % (nll_after, eces_after))
    return calibrator

def train_temperature_specific_prompts(params, train_sentences, train_labels, prompt, prompt_labels, batch=32, epochs=50, lr=0.01):
    select_string = lambda x, idx: [x[id] for id in idx]
    nll_criterion = nn.CrossEntropyLoss()
    ece_criterion = ECELoss()
    
    train_labels = np.array(train_labels)
    all_logits, all_labels = None, None
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

    all_logits = all_logits.cpu().numpy()
    all_labels = all_labels.cpu().numpy()

    calibrator = calibration.PlattBinnerMarginalCalibrator(num_calibration=all_labels.shape[0], num_bins=10)
    calibrator.train_calibration(all_logits, all_labels)
    eces_before = ece_criterion(all_logits, all_labels).item()
    nll_before = cross_entropy_loss(all_logits, all_labels)
    
    logits_calibrated = calibrator.calibrate(all_logits)
    eces_after = ece_criterion(logits_calibrated, all_labels).item()
    nll_after = cross_entropy_loss(logits_calibrated, all_labels)
    print('Before calibrated - NLL: %.3f, ECE: %.3f' % (nll_before, eces_before))
    print('After calibrated - NLL: %.3f, ECE: %.3f' % (nll_after, eces_after))
    return calibrator
