import quapy as qp
import numpy as np
import settings
import glob
import pathlib
import pickle
import os
from collections import defaultdict


def load_trdataset(dataset_name):
    if dataset_name in {'semeval13', 'semeval14', 'semeval15'}:
        # these three datasets have the same training set, called "semeval"
        dataset_name = 'semeval'

    traindev_path = f'{settings.DATASET_HOME}/train/{dataset_name}.train+dev.feature.txt'
    return qp.dataset.text.LabelledCollection.from_sparse(traindev_path)


def load_dataset_prevalences(datasets, pickle_path='./dataset_prevs.dat'):
    if not os.path.exists(pickle_path):
        dataset_prevs = {dataset: load_trdataset(dataset).prevalence() for dataset in datasets}
        pickle.dump(dataset_prevs, open(pickle_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    else:
        dataset_prevs = pickle.load(open(pickle_path, 'rb'))
    return dataset_prevs




def nice(name):
    nice_names = {
        'svmkld': 'SVM(KLD)',
        'svmnkld': 'SVM(NKLD)',
        'svmq': 'SVM(Q)',
        'svmae': 'SVM(AE)',
        'svmnae': 'SVM(NAE)',
        'svmmae': 'SVM(AE)',
        'svmmrae': 'SVM(RAE)'
    }
    return nice_names.get(name, name.upper())

method_order = ['cc', 'acc', 'pcc', 'pacc', 'emq', 'svmq', 'svmkld', 'svmnkld']

if __name__ == '__main__':
    drift_measure_x = qp.error.ae

    for drift_measure_y in [qp.error.ae, qp.error.rae]:
        accept_regex = f'../results/*-m{drift_measure_y.__name__}-*.pkl'
        datasets = np.unique([pathlib.Path(result).name.split('-')[0] for result in glob.glob(accept_regex)])

        train_drifts = defaultdict(lambda:[])
        test_errors = defaultdict(lambda:[])
        dataset_prevs = load_dataset_prevalences(datasets)
        for result_path in glob.glob(accept_regex):
            dataset, method = pathlib.Path(result_path).name.split('-')[:2]
            true_prevs, estim_prevs = pickle.load(open(result_path, 'rb'))
            train_prev = dataset_prevs[dataset]
            n_samples = true_prevs.shape[0]
            train_drifts[method].extend([drift_measure_x(train_prev, true_prevs[i]) for i in range(n_samples)])
            test_errors[method].extend([drift_measure_y(true_prevs[i], estim_prevs[i]) for i in range(n_samples)])

        method_trdrift_tedrift = {
            nice(m): (train_drifts[m], test_errors[m]) for m in train_drifts.keys()
        }

        error_name = drift_measure_y.__name__.upper()
        qp.util.plot_error_by_drift(method_trdrift_tedrift,
                            n_bins=21,
                            path=f'../plots/drift{error_name}.pdf',
                            method_order=map(nice, method_order + ['svmm' + drift_measure_y.__name__]),
                            logscale_y=(error_name=='RAE'),
                            error_name=error_name)







