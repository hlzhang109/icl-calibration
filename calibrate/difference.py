import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
from utils import *
from losses import ECELoss


def get(params, train_sentences, train_labels, test_sentences):
    """
    Obtain model's responses on test sentences, given the training examples
    :param params: parameters for the experiment
    :param train_sentences: few-shot training sentences
    :param train_labels: few-shot training labels
    :param test_sentences: few-shot test sentences
    :param return_all_prompts: whether to return all the prompts
    :param num_tokens_to_predict_override: whether to override num token to predict
    :param override_prompt: whether to override prompt
    :return: a list of dictionaries
    """

    all_raw_answers = []
    prompt = construct_prompt(params, [], [], test_sentences[0])
    resp = complete(prompt, params['num_tokens_to_predict'], params['model'], num_log_probs=params['api_num_log_prob'], half=params['half'])
    
    
    prompt_train = construct_prompt(params, [], [], train_sentences[0])
    output_train = complete(prompt_train, params['num_tokens_to_predict'], params['model'], num_log_probs=params['api_num_log_prob'], half=params['half'])
    hidden_train = output_train['choices'][0]['logprobs']['hidden_states'][-1].cuda()
    label_train = torch.tensor(train_labels[0]).long().cuda()
    
    res_losses_mlp, res_losses_icl, res_eces_mlp, res_eces_icl, res_conf_mlp, res_conf_icl = [], [], [], [], [], []
    for i in range(0, 16):
        prompts = []
        if i == 0:
            prompts.append(construct_prompt(params, [], [], test_sentences[0], repeat=i, repeat_context=False))
        for test_sentence in test_sentences:
            prompts.append(construct_prompt(params, train_sentences, train_labels, test_sentence, repeat=i, repeat_context=False))
        
        labels_test = torch.from_numpy(np.array(test_labels)).cuda()
        # evaluate mlp
        logit_test = linear(hidden_test)
        logit_test = act(logit_test)
        res_eces_mlp.append(ece_criterion(logit_test, labels_test).item())
        res_conf_mlp.append(torch.max(logit_test).item())
        res_losses_mlp.append(nll_criterion(logit_test.unsqueeze(0), labels_test).item())
        
        # train the MLP
        logit_train = linear(hidden_train)
        logit_train = act(logit_train)
        loss = nll_criterion(logit_train, label_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # evaluate in-context-learning
        
        chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
        test_chunk_prompts = chunked_prompts[0]
        num_tokens_to_predict = params['num_tokens_to_predict']
        resp = complete(test_chunk_prompts, num_tokens_to_predict, params['model'], num_log_probs=params['api_num_log_prob'], half=params['half'])
        
        all_raw_answers = []
        for answer_id, answer in enumerate(resp['choices']):
            all_raw_answers.append(answer)
            
        all_label_probs = get_label_probs(params, all_raw_answers, train_sentences, train_labels, test_sentences)
        
        logits_test = torch.from_numpy(all_label_probs).float().cuda()
        logits_test = act(logits_test)
        res_conf_icl.append(torch.max(logits_test).item())
        res_losses_icl.append(nll_criterion(logits_test, labels_test).item())
        res_eces_icl.append(ece_criterion(logits_test, labels_test).item())
