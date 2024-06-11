import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
from utils import *
from losses import ECELoss


def construct_prompt_(params, train_sentences, train_labels, test_sentence, max_train=2):
    """construct a single prompt to be fed into the model"""
    # special case when the user defines a custom prompt function. 
    if ('prompt_func' in params.keys()) and (params['prompt_func'] is not None):
        return params['prompt_func'](params, train_sentences, train_labels, test_sentence)

    # take the prompt template and fill in the training and test example
    prompt = params["prompt_prefix"]
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]
    for s in train_sentences:
        prompt += q_prefix
        prompt += s + "\n Choices:"
        mark = 'A '
        for l in range(max_train):
            if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): # integer labels for classification
                assert params['task_format'] == 'classification'
                l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
            else:
                assert isinstance(l, str) # string labels
                assert params['task_format'] == 'qa'
                l_str = l

            prompt += mark + a_prefix
            prompt += l_str + "\n\n"
            mark = str(chr((ord(mark)+ 1))) + ' '
    prompt += "Answer: " + str(chr((ord('A')+ train_labels[0]))) + ' '
    prompt += q_prefix
    prompt += test_sentence + "\n"
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1] 
    return prompt

def get_model_response_(params, train_sentences, train_labels, test_sentences, max_train=2, return_all_prompts=False,
                       num_tokens_to_predict_override=None, override_prompt=None):
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

    # can optionally ignore the normal prompt and feed in a custom prompt (used for contextual calibration)
    if override_prompt is None:
        prompts = []
        for test_sentence in test_sentences:
            prompts.append(construct_prompt_(params, train_sentences, train_labels, test_sentence, max_train=max_train))
    else:
        prompts = override_prompt

    chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
    for chunk_id, test_chunk_prompts in enumerate(chunked_prompts):
        if num_tokens_to_predict_override is not None:
            num_tokens_to_predict = num_tokens_to_predict_override
        else:
            num_tokens_to_predict = params['num_tokens_to_predict']
        resp = complete(test_chunk_prompts, num_tokens_to_predict, params['model'], num_log_probs=params['api_num_log_prob'], half=params['half'])
        for answer_id, answer in enumerate(resp['choices']):
            all_raw_answers.append(answer)
    if return_all_prompts:
        return all_raw_answers, prompts
    else:
        return all_raw_answers
    
class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, mlp_width, mlp_depth, mlp_dropout, act=nn.ReLU()):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, mlp_width)
        self.dropout = nn.Dropout(mlp_dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(mlp_width, mlp_width)
            for _ in range(mlp_depth-2)])
        self.output = nn.Linear(mlp_width, n_outputs)
        self.n_outputs = n_outputs
        self.act = act

    def forward(self, x, train=True):
        x = self.input(x)
        if train:
            x = self.dropout(x)
        x = self.act(x)
        for hidden in self.hiddens:
            x = hidden(x)
            if train:
                x = self.dropout(x)
            x = self.act(x)
        x = self.output(x)
        # x = F.sigmoid(x)
        return x
    
def train_value_head(params, train_sentences, train_labels, batch=128, epochs=50, lr_value=1e-3):
    select_string = lambda x, idx: [x[id] for id in idx]
    temperature = torch.ones(1).cuda() * 1.0
    temperature.requires_grad = True
    nll_criterion = nn.CrossEntropyLoss().cuda()
    value_head = MLP(max(train_labels), max(train_labels), 32, mlp_depth=1, mlp_dropout=0.).cuda()
    optimizer = torch.optim.Adam(value_head.parameters(), lr=lr_value)
    
    train_labels = np.array(train_labels)
    eces_before, eces_after, nll_before, nll_after = [], [], [], []
    for e in range(epochs):
        indices = np.arange(0, len(train_sentences))
        random.shuffle(indices)
        indices_val, index_train = indices[:batch], indices[batch:]
        prompt, prompt_labels = random_sampling(select_string(train_sentences,index_train), train_labels[index_train], params['num_shots'])
        valid, labels = select_string(train_sentences,indices_val), train_labels[indices_val]

        raw_resp_test = get_model_response_(params, prompt, prompt_labels, valid, max_train=max(train_labels))
        all_label_probs = get_label_probs(params, raw_resp_test, prompt, prompt_labels, valid)
        logits = torch.from_numpy(all_label_probs).cuda()
        labels = torch.from_numpy(labels).cuda()
        
        loss = nll_criterion(logits, labels)
        
        before_temperature_ece = ece_criterion(logits, labels).item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        eces_before.append(before_temperature_ece)
        nll_before.append(nll_criterion(logits,labels).item())
        # eces_after.append(after_temperature_ece)
        # nll_after.append(nll_criterion(temperature_scale(logits, temperature), labels).item())
    
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
        all_label_probs = get_label_probs(params, raw_resp_test, prompt, prompt_labels, valid)
        logits = torch.from_numpy(all_label_probs).cuda()
        labels = torch.from_numpy(labels).cuda()
        
        loss = nll_criterion(temperature_scale(logits, temperature), labels)
        
        before_temperature_ece = ece_criterion(logits, labels).item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        after_temperature_ece = ece_criterion(temperature_scale(logits, temperature), labels).item()
        
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