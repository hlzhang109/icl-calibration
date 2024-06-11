import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from data_utils import load_dataset
from utils import *
from losses import ECELoss
from calibrate.tempture import *


SAVE_DIR_TMP = './raw_logits_cot'
os.makedirs(SAVE_DIR_TMP, exist_ok=True)

def save_pickle_tmp(params, data):
    # save results from model
    if "llama" in params['expr_name'] or 'mistralai' in params['expr_name'] or 'lmsys' in params['expr_name']:
        model_names = params['expr_name'].split('/')
        model_name = model_names[0] + model_names[-2] + '_' + model_names[-1]
    else:
        model_name = params['expr_name']
        
    file_name = os.path.join(SAVE_DIR_TMP, f"{model_name}.pkl")
    if os.path.isfile(file_name):
        print("WARNING! overwriting existing saved files")
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    print(f"Saved to {file_name}")
    return data


def construct_prompt(params, train_sentences, train_labels, test_sentence):
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
        prompt += s + " So, the answer is: "
        if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): # integer labels for classification
            assert params['task_format'] == 'classification'
            l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
        else:
            assert isinstance(l, str) # string labels
            assert params['task_format'] == 'qa'
            l_str = l
        prompt += l_str + "\n\n"

    prompt += q_prefix
    prompt += test_sentence + "\n"
    
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
    return prompt

def main(models, datasets, all_shots, num_seeds, subsample_test_set, api_num_log_prob, approx, use_saved_results, bs, half=False, **kwargs):
    """
    Run experiment or load past results, print accuracy
    """
    default_params = {
        'conditioned_on_correct_classes': True,
        'subsample_test_set': subsample_test_set,
        'api_num_log_prob': api_num_log_prob,
        'approx': approx,
        'bs': bs
    }

    # list of all experiment parameters to run
    all_params = []
    for model in models:
        for dataset in datasets:
            for num_shots in all_shots:
                for seed in range(num_seeds):
                    p = deepcopy(default_params)
                    p['model'] = model
                    p['dataset'] = dataset
                    p['seed'] = seed
                    p['half'] = half
                    p['num_shots'] = num_shots
                    p.update(kwargs)
                    p['expr_name'] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot_{repr(p['subsample_test_set'])}_subsample_seed{p['seed']}"
                    all_params.append(p)

    # query the model and save the responses
    if use_saved_results:
        load_results(all_params)
    else:
        save_results(all_params)

def softmax_entropy(raw_resp: list) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    all_entropy = []
    for i, ans in enumerate(raw_resp):
        x = ans['logprobs']['token_logits'][0]
        entropy = -(x.softmax(0) * x.log_softmax(0)).sum()
        all_entropy.append(entropy.cpu().numpy())
        
    all_entropy = np.array(all_entropy)
    return np.mean(all_entropy)

def save_results(params_list, freeze_test_set=True):
    """
    Run the model and save its responses and the rest of configs into a pickle file
    """
    result_tree = dict()
    ece_loss = ECELoss(10)
    zero_shot_represnetation = []
    temperature = None
    seed_num = 10
    for param_index, params in enumerate(params_list):
        print("\nExperiment name:", params['expr_name'])
        if params['num_shots'] == 0 and params['seed'] > 0:
            continue
        ### load data
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = load_dataset(params)
        # params_check(params)

        ### sample test set
        if params['subsample_test_set'] is None:
            test_sentences, test_labels = all_test_sentences, all_test_labels
            print(f"selecting full test set ({len(all_test_labels)} examples)")
        else:
            if freeze_test_set:
                np.random.seed(0) # always use seed 0 result if freeze
            else:
                np.random.seed(params['seed'])
            test_sentences, test_labels = random_sampling(all_test_sentences, all_test_labels, params['subsample_test_set'])
            print(f"selecting {len(test_labels)} subsample of test set")

        ### sample few-shot training examples
        np.random.seed(params['seed'])
        train_sentences, train_labels = random_sampling(all_train_sentences, all_train_labels, params['num_shots'])
   
        prompts = []
        for test_sentence in test_sentences:
            prompts.append(construct_prompt(params, train_sentences, train_labels, test_sentence))
        print(f"getting raw resp for {len(test_sentences)} test sentences")
        raw_resp_test = get_model_response(params, train_sentences, train_labels, test_sentences, override_prompt=prompts, num_tokens_to_predict_override=100)
        
        hidden_states = [raw_resp_test[i]['logprobs']['hidden_states'] for i in range(len(raw_resp_test))]
        token_logits = [raw_resp_test[i]['logprobs']['token_logits'] for i in range(len(raw_resp_test))]
        for i in range(len(raw_resp_test)):
            tokens = raw_resp_test[i]['logprobs']['tokens']
            index = None
            for j, item in enumerate(tokens):
                if params['dataset'] in ['strategy_qa_cot']:
                    keys = params['inv_label_dict'].keys()
                else:
                    keys = {'A': 0, 'B': 1, 'C':2, 'D':3, 'E':4}
                for key in keys:
                    index = None
                    if j <= 2: 
                        continue
                    if key in item or key.lower() in item or ' ' + key in item or ' ' + key.lower() in item:
                        if ':' in tokens[j-1] and 'is' in tokens[j-2] and 'answer' in tokens[j-3] and 'the' in tokens[j-4]:
                            index = j
                            break
                if index is not None:
                    break
            index = 0 if index is None else index
            for key in raw_resp_test[i]['logprobs'].keys():
                raw_resp_test[i]['logprobs'][key] = [raw_resp_test[i]['logprobs'][key][index]]

        cur_rep = [resp['logprobs']['hidden_states'][-1].numpy() for resp in raw_resp_test]
        diff = [0]
        norm = [np.linalg.norm(cur_rep[i]) for i in range(len(cur_rep))]
        diff = [np.mean(diff)]
        norm = [np.mean(norm)]
        # get prob for each label
        all_label_probs = get_label_probs(params, raw_resp_test, train_sentences, train_labels, test_sentences)
        all_label_raw_logits = get_label_logits(params, raw_resp_test, train_sentences, train_labels, test_sentences)

        # calculate P_cf
        content_free_inputs = ["N/A", "", "[MASK]"]
        acc_original, conf_ori = eval_accuracy(all_label_probs, test_labels)
        # entropy = softmax_entropy(raw_resp_test)
        ece_original = ece_loss(all_label_raw_logits, test_labels)
        accuracies = [acc_original, acc_original]
        eces = [ece_original.item()]
        confs = [conf_ori]
        p_cf = [0]
        print(f"Accuracies: {accuracies}")
        print(f"Ece      : {eces}")
        print(f"Diff      : {diff}")
        print(f"Norm      : {norm}")
        print(f"confidence      : {confs}")
        print(f"p_cf      : {p_cf}")

        # add to result_tree
        keys = [params['dataset'], params['model'], params['num_shots']]
        node = result_tree # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]
        seed = params['seed']
        node[seed] = accuracies
        if not f"{keys[2]}_ece" in result_tree[keys[0]][keys[1]].keys():
            result_tree[keys[0]][keys[1]][f"{keys[2]}_ece"] = dict()
        if not f"{keys[2]}_diff" in result_tree[keys[0]][keys[1]].keys():
            result_tree[keys[0]][keys[1]][f"{keys[2]}_diff"] = dict()
        if not f"{keys[2]}_norm" in result_tree[keys[0]][keys[1]].keys():
            result_tree[keys[0]][keys[1]][f"{keys[2]}_norm"] = dict()
        if not f"{keys[2]}_conf" in result_tree[keys[0]][keys[1]].keys():
            result_tree[keys[0]][keys[1]][f"{keys[2]}_conf"] = dict()
        result_tree[keys[0]][keys[1]][f"{keys[2]}_ece"][seed] = eces
        result_tree[keys[0]][keys[1]][f"{keys[2]}_conf"][seed] = confs
        result_tree[keys[0]][keys[1]][f"{keys[2]}_diff"][seed] = diff
        result_tree[keys[0]][keys[1]][f"{keys[2]}_norm"][seed] = norm
        # save to file
        result_to_save = dict()
        params_to_save = deepcopy(params)
        result_to_save['params'] = params_to_save
        result_to_save['test_labels'] = test_labels
        result_to_save['all_label_probs'] = all_label_probs
        result_to_save['p_cf'] = p_cf
        result_to_save['eces'] = eces
        result_to_save['diffs'] = diff
        result_to_save['norm'] = norm
        result_to_save['confs'] = confs
        result_to_save['accuracies'] = accuracies
        result_to_save['raw_logits'] = all_label_raw_logits
        if 'prompt_func' in result_to_save['params'].keys():
            params_to_save['prompt_func'] = None
        save_pickle_tmp(params, result_to_save)
        print_results(result_tree)

    print_results(result_tree)

def calibrate_ece(all_label_probs, test_labels, mode=None, p_cf=None, ece_loss=ECELoss(10)):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False

    correctness_list = []
    assert len(all_label_probs) == len(test_labels)
    for label_probs, true_label in zip(all_label_probs, test_labels):
        label_probs = label_probs / np.sum(label_probs) # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b

        loss = ece_loss(calibrate_label_probs.T, true_label)
        correctness_list.append(loss.item())
    return np.mean(correctness_list)

def eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False

    correctness_list, prob_list = [], []
    softmax = lambda x: np.exp(x)/np.exp(x).sum()
    assert len(all_label_probs) == len(test_labels)
    for label_probs, true_label in zip(all_label_probs, test_labels):
        ans_conf = np.max(label_probs)
        label_probs = label_probs / np.sum(label_probs) # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b
        ans_label = np.argmax(calibrate_label_probs)
        
        prob_list.append(ans_conf)
        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    return np.mean(correctness_list), np.mean(prob_list)

def get_p_content_free(params, train_sentences, train_labels, content_free_inputs=('N/A',)):
    """Query model with content free input, return its prediction probability for each label"""
    label_dict = params['label_dict']

    all_p_y = []
    for content_free_input in content_free_inputs:
        prompt = construct_prompt(params, train_sentences, train_labels, content_free_input)

        p_y = [0] * len(label_dict)
        for i, answers in label_dict.items():
            prob = 0
            for a in answers:
                prob += np.exp(complete(prompt + " " + a, 0, params['model'], echo=True, num_log_probs=1)['choices'][0]['logprobs']['token_logprobs'][-1])
            p_y[i] = prob
        all_p_y.append(p_y)

    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y) # normalize
    return p_y


def params_check(params):
    """sanity check the experiment params"""
    assert params['num_tokens_to_predict'] == 1
    if 'llama' in params['model']:
        return
    # for classification, make sure that all of the class names are one word.
    for key, label_names in params['label_dict'].items():
        for label_id, label_name in enumerate(label_names):
            first_token_of_label_name = complete(' ' + label_name, 1, params['model'], echo=True, num_log_probs=2, half=params['half'])['choices'][0]['logprobs']['tokens'][0]
            if first_token_of_label_name[1:] != label_name:
                print('label name is more than 1 token', label_name)
                assert False

    if not (params['dataset'] in ['cb', 'rte']):
        # formatting: there should be a space after question/answer prefix
        assert params["q_prefix"][-1] == " "
        assert params["a_prefix"][-1] == " "
        assert len(params["prompt_prefix"]) == 0 or params["prompt_prefix"][-2:] == '\n\n'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--models', dest='models', action='store', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset(s), e.g., agnews')
    parser.add_argument('--num_seeds', dest='num_seeds', action='store', required=True, help='num seeds for the training set', type=int)
    parser.add_argument('--all_shots', dest='all_shots', action='store', required=True, help='num training examples to use')
    # other arguments
    parser.add_argument('--subsample_test_set', dest='subsample_test_set', action='store', required=False, type=int,
                        default=None, help='size of test set to use to speed up eval. None means using all test set')
    parser.add_argument('--api_num_log_prob', dest='api_num_log_prob', action='store', required=False, type=int,
                        default=100, help='number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API')
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=None,
                        help='batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.')
    # flags
    parser.add_argument('--use_saved_results', dest='use_saved_results', action='store_const', const=True, default=False,
                        help='whether to load the results from pickle files and not run the model')
    parser.add_argument('--approx', dest='approx', action='store_const', const=True, default=False,
                        help='whether to set token prob to zero if not in top 100')
    
    parser.add_argument('--half', action='store_true', default=False,
                        help='Use float 16')

    args = parser.parse_args()
    args = vars(args)


    # simple processing
    def convert_to_list(items, is_int=False):
        if is_int:
            return [int(s.strip()) for s in items.split(",")]
        else:
            return [s.strip() for s in items.split(",")]

    print('num of gpus: ', torch.cuda.device_count())
    args['models'] = convert_to_list(args['models'])
    args['datasets'] = convert_to_list(args['datasets'])
    args['all_shots'] = convert_to_list(args['all_shots'], is_int=True)
    print(args)
    main(**args)