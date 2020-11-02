import itertools
import multiprocessing
from collections import defaultdict
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from quapy.error import rae
import pathlib, os


def get_parallel_slices(n_tasks, n_jobs=-1):
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    batch = int(n_tasks / n_jobs)
    remainder = n_tasks % n_jobs
    return [slice(job * batch, (job + 1) * batch + (remainder if job == n_jobs - 1 else 0)) for job in
            range(n_jobs)]


def parallelize(func, args, n_jobs):
    slices = get_parallel_slices(len(args), n_jobs)

    results = Parallel(n_jobs=n_jobs)(
        delayed(func)(args[slice_i]) for slice_i in slices
    )
    return list(itertools.chain.from_iterable(results))


def plot_diagonal(prevalences, methods_predictions, train_prev=None, test_prev=None,
                  title='Artificial Sampling Protocol', savedir=None, show=True):
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 200
    methodnames, method_predictions = zip(*list(methods_predictions.items()))
    x_ticks = np.sort(np.unique(prevalences))

    ave = np.array([[np.mean(method_i[prevalences == p]) for p in x_ticks] for method_i in method_predictions])
    std = np.array([[np.std(method_i[prevalences == p]) for p in x_ticks] for method_i in method_predictions])

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid()
    ax.plot([0,1], [0,1], '--k', label='ideal', zorder=1)
    for i,method in enumerate(ave):
        label = methodnames[i]
        ax.errorbar(x_ticks, method, fmt='-', marker='o', label=label, markersize=3, zorder=2)
        ax.fill_between(x_ticks, method-std[i], method+std[i], alpha=0.25)
    if train_prev is not None:
        ax.scatter(train_prev, train_prev, c='c', label='tr-prev', linewidth=2, edgecolor='k', s=100, zorder=3)
    if test_prev is not None:
        ax.scatter(test_prev, test_prev, c='y', label='te-prev', linewidth=2, edgecolor='k', s=100, zorder=3)

    ax.set(xlabel='true prevalence', ylabel='estimated prevalence', title=title)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if savedir is not None:
        plt.savefig(savedir)

    if show:
        plt.show()


def plot_error_histogram_(method_data_trainPrev_truePrevs_estimPrevs : dict,
                         drift_measure_x: callable,
                         drift_measure_y: callable,
                         n_bins=21,
                         nice={},
                         method_order=None,
                         logscale_y=False):
    #plt.rcParams['figure.figsize'] = [12, 8]
    #plt.rcParams['figure.dpi'] = 200
    fig, ax = plt.subplots()
    #ax.set_aspect('equal')
    ax.grid()

    bins = np.linspace(0, 1, n_bins)
    method_sample_drifts = {}
    method_errors = {}
    for method, dataset_vals in method_data_trainPrev_truePrevs_estimPrevs.items():
        for (dataset, train_prev, true_prevs, estim_prevs) in dataset_vals:
            n_samples = true_prevs.shape[0]
            sample_drift = [drift_measure_x(train_prev, true_prevs[i]) for i in range(n_samples)]
            method_error = [drift_measure_y(true_prevs[i], estim_prevs[i]) for i in range(n_samples)]

            if method not in method_errors:
                method_errors[method]=method_error
                method_sample_drifts[method]=sample_drift
            else:
                prev_drift, prev_error = method_sample_drifts[method], method_errors[method]
                prev_drift.extend(sample_drift)
                prev_error.extend(method_error)
                method_sample_drifts[method],method_errors[method] = prev_drift, prev_error

    if method_order is None:
        method_order = sorted(list(method_sample_drifts.keys()))
    for method in method_order:
        if method not in method_sample_drifts: continue
        indices = np.digitize(method_sample_drifts[method], bins=bins)
        x, y, std_low, std_high = [], [], [], []
        for idx, bin in enumerate(bins):
            scores = np.asarray(method_errors[method])[indices==idx]
            if logscale_y:
                print(method, np.any(np.isnan(scores)))
                scores = np.log(scores+1)
                print(method, np.any(np.isnan(scores)))

            mean = np.mean(scores)
            std = np.std(scores)
            if not np.isnan(mean):
                x.append(bin)
                y.append(mean)
                std_low.append(mean-std)
                std_high.append(mean + std)
        ax.errorbar(x, y, fmt='-', marker='o', label=nice.get(method, method.upper()), markersize=3, zorder=2)
        #ax.fill_between(x, std_low, std_high, alpha=0.25)
    drift_x_name = drift_measure_x.__name__.upper()
    drift_y_name = drift_measure_y.__name__.upper()
    ax.set(xlabel=f'Distribution shift between training set and test sample',
           ylabel=f'{drift_y_name} (true distribution, predicted distribution)',
           title=f'Quantification error as a function of distribution shift')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim(min(x), max(x))
    #plt.tight_layout()
    #plt.show()
    print(f'saving figure in ../drift{drift_y_name}.pdf')
    plt.savefig(f'../drift{drift_y_name}.pdf')

def plot_error_histogram(method_data_trainPrev_truePrevs_estimPrevs : dict,
                         drift_measure_x: callable,
                         drift_measure_y: callable,
                         n_bins=21,
                         nice={},
                         method_order=None,
                         logscale_y=False):

    train_drifts = defaultdict(lambda:[])
    test_errors = defaultdict(lambda:[])
    for method, dataset_vals in method_data_trainPrev_truePrevs_estimPrevs.items():
        for (dataset, train_prev, true_prevs, estim_prevs) in dataset_vals:
            n_samples = true_prevs.shape[0]
            train_drifts[method].append([drift_measure_x(train_prev, true_prevs[i]) for i in range(n_samples)])
            test_errors[method].append([drift_measure_y(true_prevs[i], estim_prevs[i]) for i in range(n_samples)])

    method_trdrift_tedrift = {method:(train_drifts[method], test_errors[method]) for method in train_drifts.keys()}

    plot_error_by_drift(method_trdrift_tedrift, n_bins, path, method_order, logscale_y, error_name)

def plot_error_by_drift(method_trdrift_tedrift:dict, n_bins=21, path='../plots/drift.pdf', method_order=None,
                        logscale_y=False, error_name='Error'):
    fig, ax = plt.subplots()
    #ax.set_aspect('equal')
    ax.grid()

    bins = np.linspace(0, 1, n_bins)

    if method_order is None:
        method_order = sorted(list(method_trdrift_tedrift.keys()))
    for method in method_order:
        if method not in method_trdrift_tedrift:
            continue
        sample_drifts, method_errors = method_trdrift_tedrift[method]
        indices = np.digitize(sample_drifts, bins=bins)
        x, y, std_low, std_high = [], [], [], []
        for idx, bin in enumerate(bins):
            scores = np.asarray(method_errors)[indices==idx]
            if logscale_y:
                scores = np.log(scores+1)

            mean = np.mean(scores)
            std = np.std(scores)
            if not np.isnan(mean):
                x.append(bin)
                y.append(mean)
                std_low.append(mean-std)
                std_high.append(mean + std)
        ax.errorbar(x, y, fmt='-', marker='o', label=method, markersize=3, zorder=2)
        #ax.fill_between(x, std_low, std_high, alpha=0.25)
    ax.set(xlabel=f'Distribution shift between training set and test sample',
           ylabel=f'{error_name} (true distribution, predicted distribution)',
           title=f'Quantification error as a function of distribution shift')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim(min(x), max(x))
    #plt.tight_layout()
    #plt.show()
    os.makedirs(pathlib.Path(path).parent, exist_ok=True)
    print(f'saving figure in {path}')
    plt.savefig(path)

