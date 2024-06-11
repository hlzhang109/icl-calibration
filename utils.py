import numpy as np
import time
from copy import deepcopy
import os
import sys
import torch
import pickle
import openai
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(ROOT_DIR, 'saved_results')
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
    print(f"mkdir at {SAVE_DIR} for saving results")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chunk_size_helper(params):
    # Set the batch size (the size of the chunks determines the batch size). Default to 4 for GPT-2 and 20 for OpenAI if
    # no batch size is specified.
    bs = params['bs']
    if bs is None:
        if 'gpt2' in params['model'] or 'llama' in params['model'].lower() or 'vicuna' in params['model'] or 'Mistral' in params['model'] or 'alpaca' in  params['model']:
            return 1
        else:
            assert params['model'] in ['ada', 'babbage', 'curie', 'davinci', 'ada-beta', 'babbage-beta', 'curie-beta', 'davinci-beta']
            return 20
    else:
        return bs

def random_sampling(sentences, labels, num):
    """randomly sample subset of the training pairs"""
    assert len(sentences) == len(labels)
    if num > len(labels):
        assert False, f"you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}"
    idxs = np.random.choice(len(labels), size=num, replace=False)
    selected_sentences = [sentences[i] for i in idxs]
    selected_labels = [labels[i] for i in idxs]
    return deepcopy(selected_sentences), deepcopy(selected_labels)

infer_model = None
infer_tokenizer = None

def setup_llama(model_name, half=False):
    from torch.distributed import rpc
    import tempfile
    import os
    # load the GPT-2 model
    from llama import LlamaTokenizer
    global infer_model
    global infer_tokenizer
    if infer_model is None:
        print("Setting up LlaMA model: ", model_name)
        from transformers import LlamaForCausalLM, LlamaTokenizer
        infer_tokenizer = LlamaTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            padding_side="left",
        )
        
        infer_model = LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage = True, torch_dtype=torch.float16)
        
        import tensor_parallel as tp
        n_gpus = torch.cuda.device_count()
        infer_model = tp.tensor_parallel(infer_model, [i for i in range(n_gpus)]) 
        infer_model.eval()
        
        # to batch generation, we pad on the left and mask those positions out.
        infer_tokenizer.padding_side = "left"
        infer_tokenizer.pad_token = infer_tokenizer.eos_token
        infer_tokenizer.pad_token_id = infer_model.config.eos_token_id
        print("Finished")
    return infer_model, infer_tokenizer


def setup_vicuna(model_name, half=False):
    from fastchat.model import load_model
    global infer_model
    global infer_tokenizer
    if infer_model is None:
        print("Setting up Vicuna model: ", model_name)
        num_gpus=torch.cuda.device_count()
        infer_model, infer_tokenizer = load_model(
            model_name,
            num_gpus=num_gpus,
        )
        
        # to batch generation, we pad on the left and mask those positions out.
        infer_tokenizer.padding_side = "left"
        infer_tokenizer.pad_token = infer_tokenizer.eos_token
        infer_tokenizer.pad_token_id = infer_model.config.eos_token_id
        print("Finished")
    return infer_model, infer_tokenizer

def setup_llama2(model_name, half=False):
    global infer_model
    global infer_tokenizer
    if infer_model is None:
        print("Setting up Llama2 or Mistral model: ", model_name)
        
        num_gpus=torch.cuda.device_count()
        infer_tokenizer = AutoTokenizer.from_pretrained(model_name)
        infer_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        import tensor_parallel as tp
        n_gpus = torch.cuda.device_count()
        infer_model = tp.tensor_parallel(infer_model, [i for i in range(n_gpus)]) 
        infer_model.eval()

        # to batch generation, we pad on the left and mask those positions out.
        infer_tokenizer.padding_side = "left"
        infer_tokenizer.pad_token = infer_tokenizer.eos_token
        infer_tokenizer.pad_token_id = infer_model.config.eos_token_id
        print("Finished")
    return infer_model, infer_tokenizer

def setup_gpt2(model_name, half=False):
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.parallel as parallel
    import os
    # load the GPT-2 model
    global infer_model
    global infer_tokenizer
    if infer_model is None:
        print("Setting up GPT-2 model")
        infer_model = GPT2LMHeadModel.from_pretrained(model_name)
        # Initialize the process group
        if half:
            infer_model.half()
        infer_model = infer_model.cuda()
        infer_model.eval()
        
        infer_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # to batch generation, we pad on the left and mask those positions out.
        infer_tokenizer.padding_side = "left"
        infer_tokenizer.pad_token = infer_tokenizer.eos_token
        infer_model.config.pad_token_id = infer_model.config.eos_token_id
        print("Finished")
    return infer_model, infer_tokenizer

def setup_gpt3():
    # get OpenAI access key
    with open(os.path.join(ROOT_DIR, 'openai_key.txt'), 'r') as f:
        key = f.readline().strip()
        openai.api_key = key

def complete_generation(prompt, l=10, model_name='gpt2-xl', num_log_probs=None, echo=False, half=False):
    ''' This function runs GPT-2 locally but places the outputs into an json that looks just like the one
     provided by the OpenAI API. '''
    with torch.no_grad():
        if isinstance(prompt, str):
            prompt = [prompt] # the code below assumes a list
        input_ids = infer_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True) # [36183]
        if (len(input_ids['input_ids'][0]) > 1023):
            input_ids['input_ids'] = input_ids['input_ids'][:, :1023]
            input_ids['attention_mask'] = input_ids['attention_mask'][:, :1023]
        # if 'alpaca' in model_name:
        #     input_ids['input_ids'] = input_ids['input_ids'][:, :511]
        #     input_ids['attention_mask'] = input_ids['attention_mask'][:, :511]
        # greedily generate l tokens
        if l > 0:
            # the generate function can handle left padded inputs automatically in HF
            # total_sequences is now the input + possible generated output
            total_sequences = infer_model.generate(input_ids=input_ids['input_ids'].cuda(), attention_mask=input_ids['attention_mask'].cuda(), max_length=l + len(input_ids['input_ids'][0]), do_sample=False)
        else:
            assert echo == True and l == 0
            total_sequences = input_ids['input_ids'].cuda()

        # they want the probs of the top tokens
        if num_log_probs is not None:
            # we are left padding, so we need to adjust the position IDs
            attention_mask = (total_sequences != 50256).float()
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # get the logits for the context and the next l tokens
            outputs = infer_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids, output_hidden_states=True,return_dict=True)
            logits = outputs.logits.detach().cpu().float()
            hidden_states = outputs.hidden_states[-1].detach().cpu().float()
            if not echo:
                # get the top tokens and probs for the generated l tokens
                logits = logits[:,-l-1:]
                hidden_states = hidden_states[:,-l-1:]
                probs = torch.softmax(logits, dim=2).cpu()
            else:
                # get the top tokens and probs for the context and the generated l tokens
                probs = torch.softmax(logits, dim=2).cpu()
            top_probs, top_tokens = torch.topk(probs, k=num_log_probs)
            top_logits, top_logit_tokens =  torch.topk(logits, k=num_log_probs)
            logprobs = torch.log(probs)
            top_log_probs = torch.log(top_probs)
            torch.cuda.empty_cache()
        # create the return value to resemble OpenAI
        return_json = {}
        choices = []
        for batch_id in range(len(prompt)):
            curr_json = {}
            # text is just the optional context and next l tokens
            if not echo:
                curr_json['text'] = infer_tokenizer.decode(total_sequences[batch_id][-l:], skip_special_tokens=True)
            else:
                curr_json['text'] = infer_tokenizer.decode(total_sequences[batch_id], skip_special_tokens=True)

            # fill the return json with the top tokens and probs to match the OpenAI return value.
            if num_log_probs is not None:
                curr_json['logprobs'] = {}
                curr_json['logprobs']['top_logprobs'] = []
                curr_json['logprobs']['token_logprobs'] = []
                curr_json['logprobs']['tokens'] = []
                curr_json['logprobs']['hidden_states'] = []
                curr_json['logprobs']['token_logits'] = []
                if not echo:
                    # cutoff the -1 here because the probs are shifted one over for LMs
                    for current_element_top_log_probs, current_element_top_tokens, token_logits, hidden in zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1], top_logits[batch_id][:-1], hidden_states[batch_id][:-1]):
                        # tokens is a list of the top token at each position
                        curr_json['logprobs']['tokens'].append(infer_tokenizer.decode([current_element_top_tokens[0]]))
                        # token_logprobs is a list of the logprob of the top token at each position
                        curr_json['logprobs']['token_logprobs'].append(current_element_top_log_probs[0].item())
                        # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
                        temp, temp_logits = {}, {}
                        for log_prob, token, logit in zip(current_element_top_log_probs, current_element_top_tokens, token_logits):
                            string_ = infer_tokenizer.decode(token.item())
                            if string_ not in temp.keys():
                                temp[string_] = log_prob.item()
                                temp_logits[string_] = logit.item()
                        curr_json['logprobs']['top_logprobs'].append(temp)
                        curr_json['logprobs']['token_logits'].append(temp_logits)
                        curr_json['logprobs']['hidden_states'].append(hidden)
                else:
                    # same as not above but small tweaks
                    # we add null to the front because for the GPT models, they have null probability for the first token
                    # (for some reason they don't have an beginning of sentence token)
                    curr_json['logprobs']['top_logprobs'].append('null')
                    # cutoff the -1 here because the probs are shifted one over for LMs
                    for index, (current_element_top_log_probs, current_element_top_tokens, token_logits) in enumerate(zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1], logits[batch_id][:-1])):
                        # skip padding tokens
                        if total_sequences[batch_id][index].item() == 50256:
                            continue
                        temp = {}
                        for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                            temp[infer_tokenizer.decode(token.item())] = log_prob.item()
                        curr_json['logprobs']['top_logprobs'].append(temp)
                    for index in range(len(probs[batch_id])):
                        curr_json['logprobs']['tokens'].append(infer_tokenizer.decode([total_sequences[batch_id][index]]))
                    curr_json['logprobs']['token_logprobs'].append('null')
                    for index, log_probs_token_position_j in enumerate(logprobs[batch_id][:-1]):
                        # probs are left shifted for LMs 
                        curr_json['logprobs']['token_logprobs'].append(log_probs_token_position_j[total_sequences[batch_id][index+1]])

            choices.append(curr_json)
        return_json['choices'] = choices
        torch.cuda.empty_cache()

        return return_json

def complete_gpt3(prompt, l, model_name, temp=0, num_log_probs=None, echo=False, n=None):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            response = openai.Completion.create(engine=model_name, prompt=prompt, max_tokens=l, temperature=temp,
                                                logprobs=num_log_probs, echo=echo, stop='\n', n=n)
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False

            print("API error:", error)
            time.sleep(1)
    return response

def complete(prompt, l, model, temp=0, num_log_probs=None, echo=False, n=None, half=False):
    """complete the prompt using a language model"""
    assert l >= 0
    assert temp >= 0
    if 'gpt2' in model:
        assert n == None # unsupported at the moment
        assert temp == 0 # unsupported at the moment
        setup_gpt2(model, half=half)
        return complete_generation(prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo, half=half)
    elif 'Llama-2' in model or 'Mistral' in model:
        assert temp == 0 # unsupported at the moment
        setup_llama2(model, half=half)
        return complete_generation(prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo, half=half)
    elif 'vicuna' in model:
        assert temp == 0 # unsupported at the moment
        setup_vicuna(model, half=half)
        return complete_generation(prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo, half=half)
    elif 'llama' in model or 'alpaca' in model:
        assert temp == 0 # unsupported at the moment
        setup_llama(model, half=half)
        return complete_generation(prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo, half=half)
    else:
        setup_gpt3()
        return complete_gpt3(prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo, n=n)

def construct_prompt(params, train_sentences, train_labels, test_sentence, repeat=1, repeat_context=False):
    """construct a single prompt to be fed into the model"""
    # special case when the user defines a custom prompt function. 
    if ('prompt_func' in params.keys()) and (params['prompt_func'] is not None):
        return params['prompt_func'](params, train_sentences, train_labels, test_sentence)

    # take the prompt template and fill in the training and test example
    
    prompt = params["prompt_prefix"]
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]
    for s, l in zip(train_sentences, train_labels):
        prompt += q_prefix
        prompt += s + "\n"
        if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): # integer labels for classification
            assert params['task_format'] == 'classification'
            l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
        else:
            assert isinstance(l, str) # string labels
            assert params['task_format'] == 'qa'
            l_str = l

        if repeat > 1:
            for i in range(repeat):
                if repeat_context:
                    if i == 0:
                        prompt = prompt[:-1] + ' '
                    prompt += s + " "
                    if i == repeat - 1:
                        prompt = prompt[:-1] + '\n'
                        prompt += a_prefix
                        prompt += l_str + "\n\n"          
                else:
                    if i != 0:
                        prompt += q_prefix + s + "\n"
                    prompt += a_prefix
                    prompt += l_str + "\n\n"                        
        else:
            prompt += a_prefix
            prompt += l_str + "\n\n"

    prompt += q_prefix
    prompt += test_sentence + "\n"
    
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
    return prompt



def get_model_response(params, train_sentences, train_labels, test_sentences, return_all_prompts=False,
                       num_tokens_to_predict_override=None, override_prompt=None, repeat=1, repeat_context=False):
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
            prompts.append(construct_prompt(params, train_sentences, train_labels, test_sentence, repeat=repeat, repeat_context=repeat_context))
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



def get_label_logits(params, raw_resp, train_sentences, train_labels, test_sentences):
    """Obtain model's label probability for each of the test examples. The returned prob is NOT normalized"""
    num_classes = len(params['label_dict'])
    approx = params['approx']
    assert len(raw_resp) == len(test_sentences)

    # Fill in the labels that is in the top k prob
    all_label_probs = []
    all_missing_positions = []
    for i, ans in enumerate(raw_resp):
        top_logprobs = ans['logprobs']['token_logits'][0]  # [0] since we only ask for complete one more token
        if 'sst2' in params['dataset'] and ('llama' in params['model'] or 'vicuna' in params['model'] or 'alpaca' in params['model']  or 'Llama' in params['model']):
            labels_dict_items = {0: ['Neg'], 1: ['Pos']}.items()
        else:
            labels_dict_items = params['label_dict'].items()
        label_probs = [0] * len(params['label_dict'].keys())
        for j, label_list in labels_dict_items:
            all_found = True
            for label in label_list:  # each possible label correspond to the same class
                label = label  # notice prompt does not have space after 'A:'
                if " " + label in top_logprobs:
                    label_probs[j] += top_logprobs[" " + label]
                elif label in top_logprobs:
                    label_probs[j] += top_logprobs[label]
                elif label.lower() in top_logprobs:
                    label_probs[j] += top_logprobs[label.lower()]
                else:
                    all_found = False
            if not all_found:
                position = (i, j) # (which test example, which label)
                all_missing_positions.append(position)
        all_label_probs.append(label_probs)
    all_label_probs = np.array(all_label_probs) # prob not normalized

    # Fill in the label probs that are NOT in top k probs, by asking the model to rate perplexity
    # This helps a lot in zero shot as most labels wil not be in Top 100 tokens returned by LM
    if (not approx) and (len(all_missing_positions) > 0):
        print(f"Missing probs: {len(all_missing_positions)}/{len(raw_resp) * num_classes}")
        all_additional_prompts = []
        num_prompts_each = []
        for position in all_missing_positions:
            which_sentence, which_label = position
            test_sentence = test_sentences[which_sentence]
            label_list = params['label_dict'][which_label]
            for label in label_list:
                prompt = construct_prompt(params, train_sentences, train_labels, test_sentence)
                prompt += " " + label
                all_additional_prompts.append(prompt)
            num_prompts_each.append(len(label_list))

        # chunk the prompts and feed into model
        chunked_prompts = list(chunks(all_additional_prompts, chunk_size_helper(params)))
        all_probs = []
        for chunk_id, chunk in enumerate(chunked_prompts):
            resp = complete(chunk, 0, params['model'], echo=True, num_log_probs=1)
            for ans in resp['choices']:
                prob = ans['logprobs']['token_logits'][-1]
                all_probs.append(prob)

        assert sum(num_prompts_each) == len(all_probs)
        assert len(num_prompts_each) == len(all_missing_positions)

        # fill in corresponding entries in all_label_probs
        for index, num in enumerate(num_prompts_each):
            probs = []
            while num > 0:
                probs.append(all_probs.pop(0))
                num -= 1
            prob = np.sum(probs)
            i, j = all_missing_positions[index]
            all_label_probs[i][j] = prob

        assert len(all_probs) == 0, "all should be popped"
        assert (all_label_probs > 0).all(), "all should be populated with non-zero value"

    return all_label_probs # NOT NORMALIZED


def get_label_probs(params, raw_resp, train_sentences, train_labels, test_sentences):
    """Obtain model's label probability for each of the test examples. The returned prob is NOT normalized"""
    num_classes = len(params['label_dict'])
    approx = params['approx']
    assert len(raw_resp) == len(test_sentences)

    # Fill in the labels that is in the top k prob
    all_label_probs = []
    all_missing_positions = []
    for i, ans in enumerate(raw_resp):
        top_logprobs = ans['logprobs']['top_logprobs'][0]  # [0] since we only ask for complete one more token
        if 'sst2' in params['dataset'] and ('llama' in params['model'] or 'vicuna' in params['model'] or 'alpaca' in params['model']  or 'Llama' in params['model']):
            labels_dict_items = {0: ['Neg'], 1: ['Pos']}.items()
        else:
            labels_dict_items = params['label_dict'].items()
        label_probs = [0] * len(params['label_dict'].keys())
        for j, label_list in labels_dict_items:
            all_found = True
            for label in label_list:  # each possible label correspond to the same class
                label = label  # notice prompt does not have space after 'A:'
                if " " + label in top_logprobs:
                    label_probs[j] += np.exp(top_logprobs[" " + label])
                elif label in top_logprobs:
                    label_probs[j] += np.exp(top_logprobs[label])
                elif label.lower() in top_logprobs:
                    label_probs[j] += np.exp(top_logprobs[label.lower()])
                else:
                    all_found = False
            if not all_found:
                position = (i, j) # (which test example, which label)
                all_missing_positions.append(position)
        all_label_probs.append(label_probs)
    all_label_probs = np.array(all_label_probs) # prob not normalized

    # Fill in the label probs that are NOT in top k probs, by asking the model to rate perplexity
    # This helps a lot in zero shot as most labels wil not be in Top 100 tokens returned by LM
    if (not approx) and (len(all_missing_positions) > 0):
        print(f"Missing probs: {len(all_missing_positions)}/{len(raw_resp) * num_classes}")
        all_additional_prompts = []
        num_prompts_each = []
        for position in all_missing_positions:
            which_sentence, which_label = position
            test_sentence = test_sentences[which_sentence]
            label_list = params['label_dict'][which_label]
            for label in label_list:
                prompt = construct_prompt(params, train_sentences, train_labels, test_sentence)
                prompt += " " + label
                all_additional_prompts.append(prompt)
            num_prompts_each.append(len(label_list))

        # chunk the prompts and feed into model
        chunked_prompts = list(chunks(all_additional_prompts, chunk_size_helper(params)))
        all_probs = []
        for chunk_id, chunk in enumerate(chunked_prompts):
            resp = complete(chunk, 0, params['model'], echo=True, num_log_probs=1)
            for ans in resp['choices']:
                prob = np.exp(ans['logprobs']['token_logprobs'][-1])
                all_probs.append(prob)

        assert sum(num_prompts_each) == len(all_probs)
        assert len(num_prompts_each) == len(all_missing_positions)

        # fill in corresponding entries in all_label_probs
        for index, num in enumerate(num_prompts_each):
            probs = []
            while num > 0:
                probs.append(all_probs.pop(0))
                num -= 1
            prob = np.sum(probs)
            i, j = all_missing_positions[index]
            all_label_probs[i][j] = prob

        assert len(all_probs) == 0, "all should be popped"
        assert (all_label_probs > 0).all(), "all should be populated with non-zero value"

    return all_label_probs # NOT NORMALIZED

def load_pickle(params):
    # load saved results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    assert os.path.isfile(file_name), f"file does not exist: {file_name}"
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    print(f"Loaded data from {file_name}")
    return data

def save_pickle(params, data):
    # save results from model
    if "llama" in params['expr_name']:
        model_names = params['expr_name'].split('/')
        model_name = model_names[0] + model_names[-2] + '_' + model_names[-1]
    else:
        model_name = params['expr_name']
        
    file_name = os.path.join(SAVE_DIR, f"{model_name}.pkl")
    if os.path.isfile(file_name):
        print("WARNING! overwriting existing saved files")
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    print(f"Saved to {file_name}")
    return data

def print_results(tree, names=('Original Accuracy  ','Calibrated Accuracy')):
    # print out all results
    root = deepcopy(tree)
    names_ece = ('Original ECE', )
    for dataset in root.keys():
        print(f"\n\nDataset: {dataset}")
        models_node = root[dataset]
        for model in models_node.keys():
            print(f"\nModel: {model}")
            num_shots_node = models_node[model]
            for num_shots in num_shots_node.keys():
                accuracies = np.array(list(num_shots_node[num_shots].values()))
                if len(accuracies) == 0:
                    continue
                accuracies_mean = np.mean(accuracies, axis=0)
                accuracies_low = np.min(accuracies, axis=0)
                accuracies_high = np.max(accuracies, axis=0)
                accuracies_std = np.std(accuracies, axis=0)

                if isinstance(num_shots, str)  and 'ece' in num_shots:
                    names = names_ece
                elif isinstance(num_shots, str)  and 'diff' in num_shots:
                    names = ('Conf diff', )
                elif isinstance(num_shots, str)  and 'norm' in num_shots:
                    names = ('Feature norm', )
                elif isinstance(num_shots, str)  and 'temp' in num_shots:
                    names = ('Best tempture', 'Specific')
                elif isinstance(num_shots, str)  and 'entropy' in num_shots:
                    names = ('Entropy',)
                elif isinstance(num_shots, str)  and 'conf' in num_shots:
                    names = ('Confidence',)
                else:
                    names = ('Original Accuracy  ','Calibrated Accuracy')
                    print(f"\n{num_shots}-shot, {len(accuracies)} seeds")
                for i, (m, l, h, s) in enumerate(zip(accuracies_mean, accuracies_low, accuracies_high, accuracies_std)):
                    print(f"{names[i]} | Mean: {m:.4f}, Low: {l:.4f}, High: {h:.4f}, Std: {s:.4f}")
                print()

def load_results(params_list):
    # load saved results from model
    result_tree = dict()
    for params in params_list:
        saved_result = load_pickle(params)
        keys = [params['dataset'], params['model'], params['num_shots']]
        node = result_tree # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]
        node[params['seed']] = saved_result['accuracies']
    print_results(result_tree)