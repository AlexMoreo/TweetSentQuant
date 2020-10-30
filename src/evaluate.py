import glob
import pickle
from scipy.stats import ttest_ind_from_stats, ttest_rel
from quapy.error import *
from collections import defaultdict
import numpy as np
import sys


def __clean_name(method_name, del_run=True):
    #method_name is <path>/dataset_method_learner_length_optim_run.pkl
    method_name = method_name.lower()
    method_name = method_name.replace('.pkl', '')
    if '/' in method_name:
        method_name = method_name[method_name.rfind('/') + 1:]
    if del_run and '-run' in method_name:
        method_name = method_name[:method_name.find('-run')]
    return method_name


def evaluate_directory(result_path_regex='../results/*.pkl', evaluation_measures=[mae]):
    """
    A method that pre-loads all results and evaluates them in terms of some evaluation measures
    :param result_path_regex: the regular expression accepting all methods to be evaluated
    :param evaluation_measures: the evaluation metrics (a list of callable functions) to apply (e.g., mae, mrae)
    :return: a dictionary with keys the names of the methods plus a suffix -eval, and values the score of the
            evaluation metric (eval)
    """
    result_dict = defaultdict(lambda: [])

    for result in glob.glob(result_path_regex):
        with open(result, 'rb') as fin:
            true_prevalences, estim_prevalences = pickle.load(fin)

        method_name = __clean_name(result)
        for eval in evaluation_measures:
            score = eval(true_prevalences, estim_prevalences)
            result_dict[method_name+'-'+eval.__name__].append(score)

    result_dict = {method: np.mean(scores) for method, scores in result_dict.items()}
    return result_dict


def statistical_significance(result_path_regex='../results/*.pkl', eval_measure=ae):
    """
    Performs a series of two-tailored t-tests comparing any method with the best one found for the metric-dataset pair.
    :param result_path_regex: a regex to search for all methods that have to be submitted to the test
    :param eval_measure: the evaluation metric (e.g., ae, or rae) that will be the object of study of the test
    :return: a dictionary with keys the names of the methods, and values a tuple (x,y), i, which:
        x takes on values (best, verydifferent, different, nondifferent) indicating the outcome of the test w.r.t. the
            best performing method, for confidences pval<0.001, 0.001<=p-val<0.05, >=0.05, respectively
        y the interpolated rank, with 1 being assigned to the best method, 0 to the worst
    """
    result_dict = defaultdict(lambda: [])

    for result in glob.glob(result_path_regex):
        with open(result, 'rb') as fin:
            true_prevalences, estim_prevalences = pickle.load(fin)

        method_name = __clean_name(result)

        scores = eval_measure(true_prevalences, estim_prevalences)
        result_dict[method_name].extend(scores)

    if len(result_dict)==0:
        print('no result submitted to t-test with '+result_path_regex)
        return {}

    method_score = [(method, np.mean(scores)) for method, scores in result_dict.items()]
    method_score = sorted(method_score, key=lambda x:x[1])
    best_method, mean1 = method_score[0]
    worst_method, meanworst = method_score[-1]
    sorted_methods = np.asarray([m for m,s in method_score])
    std1, nobs1 = np.mean(result_dict[best_method]), len(result_dict[best_method])

    stats = {}
    for method, scores in result_dict.items():
        if method == best_method:
            stats[method] = ('best', 1, 1)
        else:
            mean2 = np.mean(scores)
            std2  = np.std(scores)
            nobs2 = len(scores)
            _, pval = ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2)
            rel_rank = 1 - (mean2-mean1) / (meanworst-mean1)
            abs_rank = np.argwhere(method==sorted_methods).item()+1
            #_, pval = ttest_rel(best_scores, scores)
            if pval < 0.001:
                stats[method] = ('verydifferent', rel_rank, abs_rank)
            elif pval < 0.05:
                stats[method] = ('different', rel_rank, abs_rank)
            else:
                stats[method] = ('nondifferent', rel_rank, abs_rank)

    return stats


def load_Gao_Sebastiani_previous_results():
    nice = {
        'kld':'svmkld',
        'nkld':'svmnkld',
        'qbeta2':'svmq',
        'em':'emq'
    }
    gao_seb_results = {}
    with open('../Gao_Sebastiani_results.txt', 'rt') as fin:
        lines = fin.readlines()
        for line in lines[1:]:
            line = line.strip()
            parts = line.lower().split()
            print(parts)
            if len(parts) == 4:
                dataset, method, ae, rae = parts
            else:
                method, ae, rae = parts
            learner,method = method.split('-')
            method = nice.get(method, method)
            gao_seb_results[f'{dataset}-{method}-mae'] = float(ae)
            gao_seb_results[f'{dataset}-{method}-mrae'] = float(rae)
    return gao_seb_results


def get_ranks_from_Gao_Sebastiani():
    gao_seb_results = load_Gao_Sebastiani_previous_results()
    datasets = set([key.split('-')[0] for key in gao_seb_results.keys()])
    methods = np.asarray(sorted(list(set([key.split('-')[1] for key in gao_seb_results.keys()]))))
    ranks = {}
    for metric in ['mae', 'mrae']:
        for dataset in datasets:
            scores = [gao_seb_results[f'{dataset}-{method}-{metric}'] for method in methods]
            order = np.argsort(scores)
            sorted_methods = methods[order]
            for i,method in enumerate(sorted_methods):
                ranks[f'{dataset}-{method}-{metric}']=i+1
    return ranks, gao_seb_results


if __name__ == '__main__':
    ranks, results = get_ranks_from_Gao_Sebastiani()
    methods = sorted(list(set([key.split('-')[1] for key in ranks.keys()])))
    dataset = 'wb'
    print(dataset)
    for method in methods:
        r = ranks[f'{dataset}-{method}-mae']
        s = results[f'{dataset}-{method}-mae']
        print(f'\t{method}\trank={r}\tscore={s}')


    sys.exit(0)

    # testing
    result_dict = evaluate_directory('../results/*.pkl', [mae])
    for method, scores in result_dict.items():
        print(f'{method}:={scores}')

    sys.exit(0)

