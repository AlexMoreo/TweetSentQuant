import os
import pickle
from pathlib import Path
from absl import flags, logging
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import settings
import quapy as qp
from quapy.dataset.text import TQDataset
from quapy.method.aggregative import *
from quapy.method.non_aggregative import *
from tqdm import tqdm


FLAGS = flags.FLAGS

# quantifiers:
# ----------------------------------------
# alias for quantifiers and default configurations
QUANTIFIER_ALIASES = {
    'cc': lambda learner: ClassifyAndCount(learner),
    'acc': lambda learner: AdjustedClassifyAndCount(learner),
    'pcc': lambda learner: ProbabilisticClassifyAndCount(learner),
    'pacc': lambda learner: ProbabilisticAdjustedClassifyAndCount(learner),
    'emq': lambda learner: ExpectationMaximizationQuantifier(learner),
    'svmq': lambda learner: OneVsAllELM(settings.SVM_PERF_HOME, loss='q'),
    'svmkld': lambda learner: OneVsAllELM(settings.SVM_PERF_HOME, loss='kld'),
    'svmnkld': lambda learner: OneVsAllELM(settings.SVM_PERF_HOME, loss='nkld'),
    'svmmae': lambda learner: OneVsAllELM(settings.SVM_PERF_HOME, loss='mae'),
    'svmmrae': lambda learner: OneVsAllELM(settings.SVM_PERF_HOME, loss='mrae'),
    'mlpe': lambda learner: MaximumLikelihoodPrevalenceEstimation(),
}


# learners:
# ----------------------------------------
TFIDF_BASED={'svm', 'lr','svmperf', 'none'}
DEEPLEARNING_BASED={'cnn'}

# alias for classifiers/regressors and default configurations
LEARNER_ALIASES = {
    'svm': lambda: LinearSVC(),
    'lr': lambda: LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1),
    'svmperf': lambda: SVMperf(settings.SVM_PERF_HOME),
    'none': lambda: None
}

# hyperparameter spaces for each classifier/regressor
__C_range = np.logspace(-4, 5, 10)

HYPERPARAMS = {
    'svm': {'C': __C_range, 'class_weight': [None, 'balanced']},
    'lr': {'C': __C_range, 'class_weight': [None, 'balanced']},
    'svmperf': {'C': __C_range},
    'none': {}
}


# apps' utils:
# ----------------------------------------
def set_random_seed():
    np.random.seed(FLAGS.seed)


def load_dataset_model_selection():
    logging.info(f'loading dataset {FLAGS.dataset}')
    dataset = FLAGS.dataset

    if dataset in {'semeval13', 'semeval14', 'semeval15'}:
        # these three datasets have the same training set, called "semeval"
        dataset = 'semeval'

    train_path = f'{settings.DATASET_HOME}/train/{dataset}.train.feature.txt'
    dev_path = f'{settings.DATASET_HOME}/test/{dataset}.dev.feature.txt'
    dataset_model_selection = TQDataset.from_sparse(train_path, dev_path)
    print(f'\ttraining-prevalence={strprev(dataset_model_selection.training.prevalence())}')
    print(f'\tdevel-prevalence={strprev(dataset_model_selection.test.prevalence())}')

    return dataset_model_selection


def load_dataset_model_evaluation(test_set_name):
    logging.info(f'loading dataset {test_set_name}')

    if test_set_name in {'semeval13', 'semeval14', 'semeval15'}:
        # these three datasets have the same training set, called "semeval"
        name_tr = 'semeval'
    else:
        name_tr = test_set_name

    traindev_path = f'{settings.DATASET_HOME}/train/{name_tr}.train+dev.feature.txt'
    test_path = f'{settings.DATASET_HOME}/test/{test_set_name}.test.feature.txt'
    if test_set_name == 'semeval16':
        # correct on the fly one corrupt name found for semeval16
        test_path = f'{settings.DATASET_HOME}/test/{test_set_name}.devtest.feature.txt'

    dataset_model_evaluation = TQDataset.from_sparse(traindev_path, test_path)
    print(f'\ttrain+dev-prevalence={strprev(dataset_model_evaluation.training.prevalence())}')
    print(f'\ttest-prevalence={strprev(dataset_model_evaluation.test.prevalence())}')

    return dataset_model_evaluation


def resample_training_prevalence(benchmark: TQDataset):
    prev = FLAGS.trainp
    if prev is None:
        return benchmark
    else:
        logging.info(f'resampling training set at p={100*FLAGS.trainp:.2f}%')
        assert 0 < prev < 1, f'error: trainp ({prev}) must be in (0,1)'
        new_training = benchmark.training.undersampling(prev)
        return TQDataset(training=new_training, test=benchmark.test)


def instantiate_learner():
    logging.info(f'instantiating classifier {FLAGS.learner}')

    learner = FLAGS.learner.lower()
    if learner not in LEARNER_ALIASES:
        raise ValueError(f'unknown learner {FLAGS.learner}')

    return LEARNER_ALIASES[learner]()


def instantiate_quantifier(learner):
    logging.info(f'instantiating quantifier {FLAGS.method}')

    method = FLAGS.method.lower()
    if method not in QUANTIFIER_ALIASES:
        raise ValueError(f'unknown quantification method {FLAGS.method}')

    return QUANTIFIER_ALIASES[method](learner)


def instantiate_error():
    logging.info(f'instantiating error {FLAGS.error}')
    return getattr(qp.error, FLAGS.error)


def model_selection(method, benchmark: TQDataset):
    if FLAGS.error != 'none':
        error = instantiate_error()
        method = optimization(method, error, benchmark)
    else:
        logging.info('using default classifier (no model selection will be performed)')
    return method


def run_name(test_name=None):
    dataset_name = Path(FLAGS.dataset).name if test_name is None else test_name
    suffix = f'-run{FLAGS.seed}'
    return f'{dataset_name}-{FLAGS.method}-{FLAGS.learner}-{FLAGS.sample_size}-{FLAGS.error}' + suffix


def optimization(method, error, benchmark):
    logging.info(f'exploring hyperparameters')

    learner = FLAGS.learner.lower()
    if error in qp.error.CLASSIFICATION_ERROR:
        logging.info(f'optimizing for classification [{error.__name__}]')
        method = qp.optimization.optimize_for_classification(
            method,
            benchmark.training,
            benchmark.test,
            error,
            param_grid=HYPERPARAMS[learner]
        )
    elif error in qp.error.QUANTIFICATION_ERROR:
        logging.info(f'optimizing for quantification [{error.__name__}]')
        optim_for_quantification = qp.optimization.optimize_for_quantification
        if isinstance(method, OneVsAllELM):
            logging.info(f'\tmethod is an instance of {OneVsAllELM.__name__}: applying the pre-computed version')
            optim_for_quantification = qp.optimization.optimize_for_quantification_ELM
        method = optim_for_quantification(
            method,
            benchmark.training,
            benchmark.test,
            error,
            FLAGS.sample_size,
            sample_prevalences=artificial_prevalence_sampling(
                benchmark.n_classes, n_prevalences=settings.DEVEL_PREVALENCES, repeat=settings.DEVEL_REPETITIONS
            ),
            param_grid=HYPERPARAMS[learner],
            n_jobs=-1
        )
    else:
        raise ValueError('unexpected value for parameter "error"')
    return method


def produce_predictions(method, test):
    if isinstance(method, OneVsAllELM):
        """
        OneVsAllELM relies on a committee of SVMperf-based quantifiers. This learner requires the data to be stored in
        disk before training a testing. For the artificial sampling protocol evaluation, creating a file for every 
        test sample constitutes a very cumbersome bottleneck. The function produce_predictions_ELM precomputes all
        classifications (i.e., it produces only one test file), and computes the sampling quantifications based on it.
        Although this workaround could be used for other methods (e.g., directly on CC, PCC, and with little 
        modifications on the others) it is not truly general. For example, a "transductive" method cannot be adapted
        to this strategy easily.
        """
        return produce_predictions_ELM(method, test)
    return produce_predictions_general(method, test)


def produce_predictions_general(method, test):
    logging.info(f'generating predictions for test')

    n_prevalences = settings.TEST_PREVALENCES
    repeats = settings.TEST_REPETITIONS

    def test_method(sample, method):
        true_prevalence = sample.prevalence()
        estim_prevalence = method.quantify(sample.documents)
        return true_prevalence, estim_prevalence

    results = Parallel(n_jobs=-1)(
        delayed(test_method)(sample, method) for sample in tqdm(
            test.artificial_sampling_generator(FLAGS.sample_size, n_prevalences=n_prevalences, repeats=repeats),
            total=int((n_prevalences*n_prevalences+1)*repeats/2),
            desc='testing'
        )
    )

    true_prevalences, estim_prevalences = zip(*results)
    true_prevalences = np.asarray(true_prevalences)
    estim_prevalences = np.asarray(estim_prevalences)

    return true_prevalences, estim_prevalences


def produce_predictions_ELM(method:OneVsAllELM, test):
    logging.info(f'precomputing all class predictions for svmperf-based quantifiers')
    assert isinstance(method, OneVsAllELM), f'this hack works only for {OneVsAllELM.__name__} instances'
    test = method.preclassify_collection(test)

    n_prevalences = settings.TEST_PREVALENCES
    repeats = settings.TEST_REPETITIONS

    def test_method(sample):
        true_prevalence = sample.prevalence()
        estim_prevalence = sample.documents.mean(axis=0)
        estim_prevalence /= estim_prevalence.sum()
        return true_prevalence, estim_prevalence

    results = Parallel(n_jobs=-1)(
        delayed(test_method)(sample) for sample in tqdm(
            test.artificial_sampling_generator(FLAGS.sample_size, n_prevalences=n_prevalences, repeats=repeats),
            total=int((n_prevalences*n_prevalences+1)*repeats/2),
            desc='testing'
        )
    )

    true_prevalences, estim_prevalences = zip(*results)
    true_prevalences = np.asarray(true_prevalences)
    estim_prevalences = np.asarray(estim_prevalences)

    return true_prevalences, estim_prevalences


def evaluate_experiment(true_prevalences, estim_prevalences, test_name=None):
    n_classes = true_prevalences.shape[1]
    repeats = settings.TEST_REPETITIONS
    true_ave = true_prevalences.reshape(-1, repeats, n_classes).mean(axis=1)
    estim_ave = estim_prevalences.reshape(-1, repeats, n_classes).mean(axis=1)
    estim_std = estim_prevalences.reshape(-1, repeats, n_classes).std(axis=1)
    print('\nTrueP->mean(Phat)(std(Phat))\n'+'='*22)
    for true, estim, std in zip(true_ave, estim_ave, estim_std):
        str_estim = ', '.join([f'{mean:.3f}+-{std:.4f}' for mean, std in zip(estim, std)])
        print(f'{strprev(true)}->[{str_estim}]')

    print('\nEvaluation Metrics:\n'+'='*22)
    for eval_measure in [qp.error.mae, qp.error.mrae]:
        err = eval_measure(true_prevalences, estim_prevalences)
        print(f'\t{eval_measure.__name__}={err:.4f}')
    print()
    save_arrays(FLAGS.results, true_prevalences, estim_prevalences, test_name)


def evaluate_method_point_test(method, test : LabelledCollection, test_name=None):
    estim_prev = method.quantify(test.documents)
    true_prev = prevalence_from_labels(test.labels, test.n_classes)
    print('\nPoint-Test evaluation:\n' + '=' * 22)
    print(f'true-prev={strprev(true_prev)}, estim-prev={strprev(estim_prev)}')
    for eval_measure in [qp.error.mae, qp.error.mrae]:
        err = eval_measure(true_prev, estim_prev)
        print(f'\t{eval_measure.__name__}={err:.4f}')
    save_arrays(FLAGS.results_point, true_prev, estim_prev, test_name)


def save_arrays(path, true_array, estim_array, test_name=None):
    os.makedirs(path, exist_ok=True)
    fout = f'{path}/{run_name(test_name)}.pkl'
    logging.info(f'saving results in {fout}')
    true_array = np.asarray(true_array)
    estim_array = np.asarray(estim_array)
    with open(fout, 'wb') as foo:
        pickle.dump(tuple((true_array, estim_array)), foo, pickle.HIGHEST_PROTOCOL)


