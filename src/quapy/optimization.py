import itertools
import numpy as np
from joblib import Parallel, delayed
from sklearn import clone
from tqdm import tqdm
from quapy.dataset.text import LabelledCollection
from quapy.method.aggregative import BaseQuantifier, OneVsAllELM


def optimize_for_quantification(method : BaseQuantifier,
                                training: LabelledCollection,
                                validation,  #: LabelledCollection or float (split-point),
                                error,
                                sample_size,
                                sample_prevalences,
                                param_grid,
                                refit=False,
                                n_jobs=-1):

    if isinstance(validation, float):
        training, validation = training.split_stratified(validation)
    elif not isinstance(validation, LabelledCollection):
        raise ValueError('unexpected type for valid_set; accepted are LabelledCollection or float (split-point)')

    params_keys = list(param_grid.keys())
    params_values = list(param_grid.values())

    # generate the indexes that extract samples at desires prevalences
    sampling_indexes = [validation.sampling_index(sample_size, *prev) for prev in sample_prevalences]

    # the true prevalences might slightly differ from the requested prevalences
    true_prevalences = np.array([validation.sampling_from_index(idx).prevalence() for idx in sampling_indexes])

    print(f'[starting optimization with n_jobs={n_jobs}]')
    scores_params=[]
    for values in itertools.product(*params_values):
        params = {k: values[i] for i, k in enumerate(params_keys)}

        # overrides default parameters with the parameters being explored at this iteration
        method.set_params(**params)
        method.fit(training)

        estim_prevalences = Parallel(n_jobs=n_jobs)(
            delayed(method.quantify)(
                validation.sampling_from_index(idx).documents
            ) for idx in tqdm(sampling_indexes, desc=f'validating hyperparameters {params}')
        )
        estim_prevalences = np.asarray(estim_prevalences)
        score = error(true_prevalences, estim_prevalences)
        print(f'checking hyperparams={params} got {error.__name__} score {score:.5f}')
        scores_params.append((score, params))
    scores, params = zip(*scores_params)
    best_pos = np.argmin(scores)
    best_params, best_score = params[best_pos], scores[best_pos]

    print(f'optimization finished: refitting for {best_params} (score={best_score:.5f}) on the whole development set')
    method.set_params(**best_params)
    if refit:
        method.fit(training+validation)
    return method


def optimize_for_quantification_ELM(method : BaseQuantifier,
                                training: LabelledCollection,
                                validation,  #: LabelledCollection or float (split-point),
                                error,
                                sample_size,
                                sample_prevalences,
                                param_grid,
                                refit=False,
                                n_jobs=-1):

    assert isinstance(method, OneVsAllELM), f'this hack works only for {OneVsAllELM.__name__} instances'

    if isinstance(validation, float):
        training, validation = training.split_stratified(validation)
    elif not isinstance(validation, LabelledCollection):
        raise ValueError('unexpected type for valid_set; accepted are LabelledCollection or float (split-point)')

    params_keys = list(param_grid.keys())
    params_values = list(param_grid.values())

    # generate the indexes that extract samples at desires prevalences
    sampling_indexes = [validation.sampling_index(sample_size, *prev) for prev in sample_prevalences]

    # the true prevalences might slightly differ from the requested prevalences
    true_prevalences = np.array([validation.sampling_from_index(idx).prevalence() for idx in sampling_indexes])

    print(f'[starting optimization with n_jobs={n_jobs}]')
    scores_params=[]
    for values in itertools.product(*params_values):
        params = {k: values[i] for i, k in enumerate(params_keys)}

        # overrides default parameters with the parameters being explored at this iteration
        method.set_params(**params)
        method.fit(training)

        print('precomputing classifications...')
        preclassif_val = method.preclassify_collection(validation)

        def task_quantify(sample):
            estim_prevalence = sample.mean(axis=0)
            estim_prevalence /= estim_prevalence.sum()
            return estim_prevalence

        estim_prevalences = Parallel(n_jobs=n_jobs)(
            delayed(task_quantify)(
                preclassif_val.sampling_from_index(idx).documents
            ) for idx in tqdm(sampling_indexes, desc=f'validating hyperparameters {params}')
        )
        estim_prevalences = np.asarray(estim_prevalences)
        score = error(true_prevalences, estim_prevalences)
        print(f'checking hyperparams={params} got {error.__name__} score {score:.5f}')
        scores_params.append((score, params))
    scores, params = zip(*scores_params)
    best_pos = np.argmin(scores)
    best_params, best_score = params[best_pos], scores[best_pos]

    print(f'optimization finished: refitting for {best_params} (score={best_score:.5f}) on the whole development set')
    method.set_params(**best_params)
    if refit:
        method.fit(training+validation)
    return method


def optimize_for_classification(method : BaseQuantifier,
                                training : LabelledCollection,
                                validation,  #: LabelledCollection or float,
                                error,
                                param_grid,
                                refit=False):

    if isinstance(validation, float):
        training, validation = training.split_stratified(validation)
    elif not isinstance(validation, LabelledCollection):
        raise ValueError('unexpected type for valid_set; accepted are LabelledCollection or float (split-point)')

    params_keys = list(param_grid.keys())
    params_values = list(param_grid.values())

    learner = method.learner
    best_p, best_error = None, None
    pbar = tqdm(list(itertools.product(*params_values)))
    for values_ in pbar:
        params_ = {k: values_[i] for i, k in enumerate(params_keys)}

        # overrides default parameters with the parameters being explored at this iteration
        learner.set_params(**params_)
        learner.fit(training.documents, training.labels)
        class_predictions = learner.predict(validation.documents)
        score = error(validation.labels, class_predictions)

        if best_error is None or score < best_error:
            best_error, best_p = score, params_
        pbar.set_description(
            f'checking hyperparams={params_} got got {error.__name__} score={score:.5f} [best params = {best_p} with score {best_error:.5f}]'
        )

    print(f'optimization finished: refitting for {best_p} on the whole development set')
    method.set_params(**best_p)
    if refit:
        method.fit(training+validation)
    return method