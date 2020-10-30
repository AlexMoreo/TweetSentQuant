import random
import subprocess
import tempfile
from os.path import join, exists
from os import remove
from subprocess import PIPE, STDOUT
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import dump_svmlight_file


class SVMperf(BaseEstimator, ClassifierMixin):

    # losses with their respective codes in svm_perf implementation
    valid_losses = {'01':0, 'f1':1, 'kld':12, 'nkld':13, 'q':22, 'qacc':23, 'qf1':24, 'qgm':25, 'mae':26, 'mrae':27}

    def __init__(self, svmperf_base, C=0.01, verbose=False, loss='01'):
        assert loss in self.valid_losses, f'unsupported loss {loss}, valid ones are {list(self.valid_losses.keys())}'
        self.tmpdir = None
        self.svmperf_learn = join(svmperf_base, 'svm_perf_learn')
        self.svmperf_classify = join(svmperf_base, 'svm_perf_classify')
        self.verbose = verbose
        self.loss = '-w 3 -l ' + str(self.valid_losses[loss])
        self.set_c(C)

    def set_c(self, C):
        self.param_C = '-c ' + str(C)

    def set_params(self, **parameters):
        assert list(parameters.keys()) == ['C'], 'currently, only the C parameter is supported'
        self.set_c(parameters['C'])

    def fit(self, X, y):
        self.classes_ = sorted(np.unique(y))
        self.n_classes_ = len(self.classes_)

        local_random = random.Random()
        # this would allow to run parallel instances of predict
        random_code = '-'.join(str(local_random.randint(0,1000000)) for _ in range(5))
        self.tmpdir = tempfile.TemporaryDirectory(suffix=random_code)

        self.model = join(self.tmpdir.name, 'model-'+random_code)
        traindat = join(self.tmpdir.name, f'train-{random_code}.dat')

        dump_svmlight_file(X, y, traindat, zero_based=False)

        cmd = ' '.join([self.svmperf_learn, self.param_C, self.loss, traindat, self.model])
        if self.verbose:
            print('[Running]', cmd)
        p = subprocess.run(cmd.split(), stdout=PIPE, stderr=STDOUT)
        remove(traindat)

        if self.verbose:
            print(p.stdout.decode('utf-8'))

        return self

    def predict(self, X, y=None):
        assert self.tmpdir is not None, 'predict called before fit, or model directory corrupted'
        assert exists(self.model), 'model not found'
        if y is None:
            y = np.zeros(X.shape[0])

        # in order to allow for parallel runs of predict, a random code is assigned
        local_random = random.Random()
        random_code = '-'.join(str(local_random.randint(0, 1000000)) for _ in range(5))
        predictions_path = join(self.tmpdir.name, 'predictions'+random_code+'.dat')
        testdat = join(self.tmpdir.name, 'test'+random_code+'.dat')
        dump_svmlight_file(X, y, testdat, zero_based=False)

        cmd = ' '.join([self.svmperf_classify, testdat, self.model, predictions_path])
        if self.verbose:
            print('[Running]', cmd)
        p = subprocess.run(cmd.split(), stdout=PIPE, stderr=STDOUT)

        if self.verbose:
            print(p.stdout.decode('utf-8'))

        predictions = (np.loadtxt(predictions_path) > 0) * 1
        remove(testdat)
        remove(predictions_path)

        return predictions


