import argparse
import os
from data_utils import load_dataset
from utils import *
# from calibrate.tempture import *
from calibrate.verified_calibration import *

def print_results(tree, names=('Original Accuracy  ','Calibrated Accuracy')):
    # print out all results
    root = deepcopy(tree)
    names_ece = ('Original ECE  ', 'calibration', 'Shot calibration', 'Prompt specific calibration')
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
                    names = ('Temp', 'Shot specific', 'Prompt specific')
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
                    p['lr'] = kwargs['lr']
                    p['batch'] = kwargs['batch']
                    p['epochs'] = kwargs['epochs']
                    p['expr_name'] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot_{repr(p['subsample_test_set'])}_subsample_seed{p['seed']}"
                    all_params.append(p)

    # query the model and save the responses
    if use_saved_results:
        load_results(all_params)
    else:
        save_results(all_params)

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
        if params['num_shots'] == 0 and params['seed'] > 0:
            continue
        print("\nExperiment name:", params['expr_name'])

        ### load data
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = load_dataset(params)
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
        
        if temperature is None:
            # train the tempture from training instances
            calibrator = train_temperature(params, all_train_sentences, all_train_labels,  num_shots=0, batch=params['batch'], epochs=params['epochs'], lr=params['lr'])
        calibrator_shot =  train_temperature(params, all_train_sentences, all_train_labels,  num_shots=params['num_shots'], batch=params['batch'], epochs=params['epochs'], lr=params['lr'])
        calibrator_fix_prompt = train_temperature_specific_prompts(params, all_train_sentences, all_train_labels, train_sentences, train_labels, batch=params['batch'], epochs=params['epochs'], lr=params['lr'])
        test_sentences, test_labels = test_sentences[params['num_shots']:], test_labels[params['num_shots']:]

        ### Evaluate the performance and save all results
        # obtaining model's response on test examples
        print(f"getting raw resp for {len(test_sentences)} test sentences")
        raw_resp_test = get_model_response(params, train_sentences, train_labels, test_sentences)
            
        cur_rep = [resp['logprobs']['hidden_states'][-1].numpy() for resp in raw_resp_test]
        # get prob for each label
        all_label_probs = get_label_probs(params, raw_resp_test, train_sentences, train_labels, test_sentences)
        all_label_raw_logits = get_label_logits(params, raw_resp_test, train_sentences, train_labels, test_sentences)

        # calculate P_cf
        content_free_inputs = ["N/A", "", "[MASK]"]
        acc_original, conf_ori = eval_accuracy(all_label_raw_logits, test_labels)
        ece_original = ece_loss(all_label_raw_logits, test_labels)
        
        ece_tempture = ece_loss(calibrator.calibrate(all_label_raw_logits), test_labels).item()
        ece_tempture_shot = ece_loss(calibrator_shot.calibrate(all_label_raw_logits), test_labels).item()
        ece_tempture_fix_prompt = ece_loss(calibrator_fix_prompt.calibrate(all_label_raw_logits), test_labels).item()
        accuracies = [acc_original, acc_original]
        eces = [ece_original.item(),  ece_tempture, ece_tempture_shot, ece_tempture_fix_prompt]
        confs = [conf_ori]
        p_cf = [0]
        print(f"Accuracies: {accuracies}")
        print(f"Ece      : {eces}")
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
        if not f"{keys[2]}_conf" in result_tree[keys[0]][keys[1]].keys():
            result_tree[keys[0]][keys[1]][f"{keys[2]}_conf"] = dict()
        if not f"{keys[2]}_temp" in result_tree[keys[0]][keys[1]].keys():
            result_tree[keys[0]][keys[1]][f"{keys[2]}_temp"] = dict()
        result_tree[keys[0]][keys[1]][f"{keys[2]}_ece"][seed] = eces
        result_tree[keys[0]][keys[1]][f"{keys[2]}_conf"][seed] = confs
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
    
    # tempture scaling
    parser.add_argument('--lr', type=float, default=1., help='optimizer learning rate')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--repeat', type=int, default=1, help='batch size')
    parser.add_argument('--epochs', type=int, default=5, help='epochs for training')
    
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

    args['models'] = convert_to_list(args['models'])
    args['datasets'] = convert_to_list(args['datasets'])
    args['all_shots'] = convert_to_list(args['all_shots'], is_int=True)
    print(args)
    main(**args)
