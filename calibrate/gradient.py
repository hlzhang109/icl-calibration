import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
from utils import *
from losses import ECELoss


def linear_vs_icl(params, all_train_sentences, all_train_labels, test_sentences, test_labels, return_all_prompts=False,
                       num_tokens_to_predict_override=None, epochs=50, lr=0.01):
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
    all_test_sentences = test_sentences.copy()
    all_test_labels = test_labels.copy()
    while True:
        train_sentences, train_labels = random_sampling(all_train_sentences, all_train_labels, params['num_shots'])
        idxs = np.random.choice(len(all_test_labels), size=1, replace=False)
        test_sentences, test_labels = [all_test_sentences[idxs[0]]], [all_test_labels[idxs[0]]]
        
        all_raw_answers = []
        prompt = construct_prompt(params, [], [], test_sentences[0])
        resp = complete(prompt, params['num_tokens_to_predict'], params['model'], num_log_probs=params['api_num_log_prob'], half=params['half'])
        
        hidden_test = resp['choices'][0]['logprobs']['hidden_states'][-1].cuda()
        linear = nn.Linear(hidden_test.shape[-1], len(params["label_dict"])).cuda()
        act = nn.Softmax(dim=-1)
        optimizer = torch.optim.Adam(linear.parameters(), lr=lr)
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()
        
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

        print('res_eces_mlp', res_eces_mlp)
        print('res_eces_icl', res_eces_icl)
        print('res_conf_mlp', res_eces_mlp)
        print('res_conf_icl', res_eces_icl)
        print('res_losses_mlp', res_losses_mlp)
        print('res_losses_icl', res_losses_icl)

def linear_vs_icl_diff(params, all_train_sentences, all_train_labels, test_sentences, test_labels, return_all_prompts=False,
                       num_tokens_to_predict_override=None, epochs=50, lr=0.01):
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
    all_test_sentences = test_sentences.copy()
    all_test_labels = test_labels.copy()
    while True:
        
        idxs = np.random.choice(len(all_test_labels), size=1, replace=False)
        test_sentences, test_labels = [all_test_sentences[idxs[0]]], [all_test_labels[idxs[0]]]
        
        all_raw_answers = []
        prompt = construct_prompt(params, [], [], test_sentences[0])
        resp = complete(prompt, params['num_tokens_to_predict'], params['model'], num_log_probs=params['api_num_log_prob'], half=params['half'])
        
        hidden_test = resp['choices'][0]['logprobs']['hidden_states'][-1].cuda()
        linear = nn.Linear(hidden_test.shape[-1], len(params["label_dict"])).cuda()
        act = nn.Softmax(dim=-1)
        optimizer = torch.optim.Adam(linear.parameters(), lr=lr)
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()
        
        
        res_losses_mlp, res_losses_icl, res_eces_mlp, res_eces_icl, res_conf_mlp, res_conf_icl = [], [], [], [], [], []
        for i in range(0, 16):
            prompts = []
            if i == 0:
                prompts.append(construct_prompt(params, [], [], test_sentences[0], repeat=i, repeat_context=False))
                train_sentences, train_labels = [], []
            else:
                train_sentences, train_labels = random_sampling(all_train_sentences, all_train_labels, 16)
                prompt_train = construct_prompt(params, [], [], train_sentences[0])
                output_train = complete(prompt_train, params['num_tokens_to_predict'], params['model'], num_log_probs=params['api_num_log_prob'], half=params['half'])
                hidden_train = output_train['choices'][0]['logprobs']['hidden_states'][-1].cuda()
                label_train = torch.tensor(train_labels[0]).long().cuda()
            for test_sentence in test_sentences:
                prompts.append(construct_prompt(params, train_sentences, train_labels, test_sentence, repeat=0, repeat_context=False))

            labels_test = torch.from_numpy(np.array(test_labels)).cuda()
            # training the mlp
            if i > 0:
                hiddens = []
                label_train = torch.tensor(train_labels).long().cuda()
                for sentence in train_sentences:
                    prompt_train = construct_prompt(params, [], [], sentence)
                    output_train = complete(prompt_train, params['num_tokens_to_predict'], params['model'], num_log_probs=params['api_num_log_prob'], half=params['half'])
                    hidden_train = output_train['choices'][0]['logprobs']['hidden_states'][-1].cuda()
                    hiddens.append(hidden_train.unsqueeze(0))
                
                hidden_train = torch.cat(hiddens)
                logit_train = linear(hidden_train)
                logit_train = act(logit_train)
                loss = nll_criterion(logit_train, label_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # get the results of MLP
            logit_test = linear(hidden_test)
            logit_test = act(logit_test)
            res_eces_mlp.append(ece_criterion(logit_test, labels_test).item())
            res_conf_mlp.append(torch.max(logit_test).item())
            res_losses_mlp.append(nll_criterion(logit_test.unsqueeze(0), labels_test).item())
            
            # get the results of in-context-learning
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

        print('res_eces_mlp', res_eces_mlp)
        print('res_eces_icl', res_eces_icl)
        print('res_conf_mlp', res_eces_mlp)
        print('res_conf_icl', res_eces_icl)
        print('res_losses_mlp', res_losses_mlp)
        print('res_losses_icl', res_losses_icl)
        

def linear_vs_icl_diff_v2(params, all_train_sentences, all_train_labels, test_sentences, test_labels, return_all_prompts=False,
                       num_tokens_to_predict_override=None, epochs=50, lr=0.01):
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
    all_test_sentences = test_sentences.copy()
    all_test_labels = test_labels.copy()
    while True:
        
        idxs = np.random.choice(len(all_test_labels), size=1, replace=False)
        test_sentences, test_labels = [all_test_sentences[idxs[0]]], [all_test_labels[idxs[0]]]
        
        all_raw_answers = []
        prompt = construct_prompt(params, [], [], test_sentences[0])
        resp = complete(prompt, params['num_tokens_to_predict'], params['model'], num_log_probs=params['api_num_log_prob'], half=params['half'])
        
        hidden_test = resp['choices'][0]['logprobs']['hidden_states'][-1].cuda()
        linear = nn.Linear(hidden_test.shape[-1], len(params["label_dict"])).cuda()
        act = nn.Softmax(dim=-1)
        optimizer = torch.optim.Adam(linear.parameters(), lr=lr)
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()
        
        
        res_losses_mlp, res_losses_icl, res_eces_mlp, res_eces_icl, res_conf_mlp, res_conf_icl = [], [], [], [], [], []
        train_sentences_all, train_labels_all = random_sampling(all_train_sentences, all_train_labels, 16)
        label_train_all = torch.tensor(train_labels_all).long().cuda()
        for i in range(0, 16):
            prompts = []
            if i == 0:
                prompts.append(construct_prompt(params, [], [], test_sentences[0], repeat=i, repeat_context=False))
                
            train_sentences, label_train, train_labels = train_sentences_all[:i], label_train_all[:i], train_labels_all[:i]
            for test_sentence in test_sentences:
                prompts.append(construct_prompt(params, train_sentences, train_labels, test_sentence, repeat=0, repeat_context=False))

            labels_test = torch.from_numpy(np.array(test_labels)).cuda()
            # training the mlp
            if i > 0:
                hiddens = []
                for sentence in train_sentences:
                    prompt_train = construct_prompt(params, [], [], sentence)
                    output_train = complete(prompt_train, params['num_tokens_to_predict'], params['model'], num_log_probs=params['api_num_log_prob'], half=params['half'])
                    hidden_train = output_train['choices'][0]['logprobs']['hidden_states'][-1].cuda()
                    hiddens.append(hidden_train.unsqueeze(0))
                
                hidden_train = torch.cat(hiddens)
                logit_train = linear(hidden_train)
                logit_train = act(logit_train)
                loss = nll_criterion(logit_train, label_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # get the results of MLP
            logit_test = linear(hidden_test)
            logit_test = act(logit_test)
            res_eces_mlp.append(ece_criterion(logit_test, labels_test).item())
            res_conf_mlp.append(torch.max(logit_test).item())
            res_losses_mlp.append(nll_criterion(logit_test.unsqueeze(0), labels_test).item())
            
            # get the results of in-context-learning
            chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
            test_chunk_prompts = chunked_prompts[0]
            num_tokens_to_predict = params['num_tokens_to_predict']
            resp = complete(test_chunk_prompts, num_tokens_to_predict, params['model'], num_log_probs=params['api_num_log_prob'], half=params['half'])
            
            all_raw_answers = []
            for answer_id, answer in enumerate(resp['choices']):
                all_raw_answers.append(answer)
            
            all_label_probs = get_label_probs(params, all_raw_answers, train_sentences[:i], train_labels[:i], test_sentences)
            logits_test = torch.from_numpy(all_label_probs).float().cuda()
            logits_test = act(logits_test)
            res_conf_icl.append(torch.max(logits_test).item())
            res_losses_icl.append(nll_criterion(logits_test, labels_test).item())
            res_eces_icl.append(ece_criterion(logits_test, labels_test).item())

        print('res_eces_mlp', res_eces_mlp)
        print('res_eces_icl', res_eces_icl)
        print('res_conf_mlp', res_eces_mlp)
        print('res_conf_icl', res_eces_icl)
        print('res_losses_mlp', res_losses_mlp)
        print('res_losses_icl', res_losses_icl)