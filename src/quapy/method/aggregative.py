from sklearn.metrics import confusion_matrix
from .base import *
from sklearn.calibration import CalibratedClassifierCV
from quapy.classification.svmperf import SVMperf
from quapy.dataset.text import LabelledCollection
from quapy.functional import *
from joblib import Parallel, delayed


# Abstract classes
# ------------------------------------
from ..error import mae


class AggregativeQuantifier(BaseQuantifier):

    @abstractmethod
    def fit(self, data: LabelledCollection, fit_learner=True, *args): ...

    def classify(self, documents):
        return self.learner.predict(documents)

    def get_params(self, deep=True):
        return self.learner.get_params()

    def set_params(self, **parameters):
        self.learner.set_params(**parameters)

    @property
    def n_classes(self):
        return len(self.classes_)

    @property
    def classes_(self):
        return self.learner.classes_


class AggregativeProbabilisticQuantifier(AggregativeQuantifier):

    def soft_classify(self, data):
        return self.learner.predict_proba(data)

    def set_params(self, **parameters):
        if isinstance(self.learner, CalibratedClassifierCV):
            parameters={'base_estimator__'+k:v for k,v in parameters.items()}
        self.learner.set_params(**parameters)


# Helper
# ------------------------------------
def training_helper(learner,
                    data: LabelledCollection,
                    fit_learner: bool = True,
                    ensure_probabilistic=False,
                    train_val_split=None):
    if fit_learner:
        if ensure_probabilistic:
            if not hasattr(learner, 'predict_proba'):
                print(f'The learner {learner.__class__.__name__} does not seem to be probabilistic. '
                      f'The learner will be calibrated.')
                learner = CalibratedClassifierCV(learner, cv=5)
        if train_val_split is not None:
            if not (0 < train_val_split < 1):
                raise ValueError(f'train/val split {train_val_split} out of range, must be in (0,1)')
            train, unused = data.split_stratified(train_size=train_val_split)
        else:
            train, unused = data, None
        learner.fit(train.documents, train.labels)
    else:
        if ensure_probabilistic:
            if not hasattr(learner, 'predict_proba'):
                raise AssertionError('error: the learner cannot be calibrated since fit_learner is set to False')
        unused = data

    return learner, unused


# Methods
# ------------------------------------
class ClassifyAndCount(AggregativeQuantifier):

    def __init__(self, learner):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True, *args):
        self.learner, _ = training_helper(self.learner, data, fit_learner)
        return self

    def quantify(self, documents, *args):
        classification = self.classify(documents)           # classify
        return prevalence_from_labels(classification, self.n_classes)  # & count


class AdjustedClassifyAndCount(AggregativeQuantifier):

    def __init__(self, learner):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True, train_val_split=0.6):
        self.learner, validation = training_helper(self.learner, data, fit_learner, train_val_split=train_val_split)
        self.cc = ClassifyAndCount(self.learner)
        y_ = self.cc.classify(validation.documents)
        y  = validation.labels
        # estimate the matrix with entry (i,j) being the estimate of P(yi|yj), that is, the probability that a
        # document that belongs to yj ends up being classified as belonging to yi
        self.Pte_cond_estim_ = confusion_matrix(y,y_).T / validation.counts()
        return self

    def quantify(self, documents, *args):
        prevs_estim = self.cc.quantify(documents)
        # solve for the linear system Ax = B with A=Pte_cond_estim and B = prevs_estim
        A = self.Pte_cond_estim_
        B = prevs_estim
        try:
            adjusted_prevs = np.linalg.solve(A, B)
            adjusted_prevs = np.clip(adjusted_prevs, 0, 1)
            adjusted_prevs /= adjusted_prevs.sum()
        except np.linalg.LinAlgError:
            adjusted_prevs = prevs_estim  # no way to adjust them!
        return adjusted_prevs

    def classify(self, data):
        return self.cc.classify(data)


class ProbabilisticClassifyAndCount(AggregativeProbabilisticQuantifier):
    def __init__(self, learner):
        self.learner = learner

    def fit(self, data : LabelledCollection, fit_learner=True, *args):
        self.learner, _ = training_helper(self.learner, data, fit_learner, ensure_probabilistic=True)
        return self

    def quantify(self, documents, *args):
        posteriors = self.soft_classify(documents)                        # classify
        prevalences = prevalence_from_probabilities(posteriors, binarize=False)  # & count
        return prevalences


class ProbabilisticAdjustedClassifyAndCount(AggregativeQuantifier):

    def __init__(self, learner):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True, train_val_split=0.6):
        self.learner, validation = training_helper(
            self.learner, data, fit_learner, ensure_probabilistic=True, train_val_split=train_val_split
        )
        self.pcc = ProbabilisticClassifyAndCount(self.learner)
        y_ = self.pcc.classify(validation.documents)
        y = validation.labels
        # estimate the matrix with entry (i,j) being the estimate of P(yi|yj), that is, the probability that a
        # document that belongs to yj ends up being classified as belonging to yi
        self.Pte_cond_estim_ = confusion_matrix(y, y_).T / validation.counts()
        return self

    def quantify(self, documents, *args):
        prevs_estim = self.pcc.quantify(documents)
        A = self.Pte_cond_estim_
        B = prevs_estim
        try:
            adjusted_prevs = np.linalg.solve(A, B)
            adjusted_prevs = np.clip(adjusted_prevs, 0, 1)
            adjusted_prevs /= adjusted_prevs.sum()
        except np.linalg.LinAlgError:
            adjusted_prevs = prevs_estim  # no way to adjust them!
        return adjusted_prevs

    def classify(self, data):
        return self.pcc.classify(data)


class ExpectationMaximizationQuantifier(AggregativeProbabilisticQuantifier):

    MAX_ITER = 1000

    def __init__(self, learner, verbose=False):
        self.learner = learner
        self.verbose = verbose

    def fit(self, data: LabelledCollection, fit_learner=True, *args):
        self.learner, _ = training_helper(self.learner, data, fit_learner, ensure_probabilistic=True)
        self.train_prevalence = prevalence_from_labels(data.labels, self.n_classes)
        return self

    def quantify(self, X, y=None, epsilon=1e-4):
        tr_prev=self.train_prevalence
        posteriors = self.soft_classify(X)
        return self.EM(tr_prev, posteriors, self.verbose, y, epsilon)

    @classmethod
    def EM(cls, tr_prev, posterior_probabilities, verbose=False, true_labels=None, epsilon=1e-4):
        Px = posterior_probabilities
        trueprev = prevalence_from_labels(true_labels, Px.shape[1]) if true_labels is not None else -1

        Ptr = np.copy(tr_prev)
        qs = np.copy(Ptr) # i.e., prevalence(ytr)

        s, converged = 0, False
        qs_prev_ = None
        while not converged and s < ExpectationMaximizationQuantifier.MAX_ITER:
            # E-step: ps is Ps(y=+1|xi)
            ps_unnormalized = (qs / Ptr) * Px
            ps = ps_unnormalized / ps_unnormalized.sum(axis=1).reshape(-1,1)

            # M-step: qs_pos is Ps+1(y=+1)
            qs = ps.mean(axis=0)

            # if verbose:
            #     print(('s={} qs_pos={:.6f}'+('' if y is None else ' true={:.6f}'.format(trueprev))).format(s,qs_pos))

            if qs_prev_ is not None and mae(qs, qs_prev_) < epsilon and s>10:
                converged = True

            qs_prev_ = qs
            s += 1

        if verbose:
            print('-'*80)

        if not converged:
            raise UserWarning('the method has reached the maximum number of iterations, it might have not converged')

        return qs


def train_task(c, learners, data):
    learners[c].fit(data.documents, data.labels == c)


def binary_quant_task(c, learners, X):
    predictions_ci = learners[c].predict(X)
    return predictions_ci.mean()  # since the predictions array is binary


class OneVsAllELM(AggregativeQuantifier):

    def __init__(self, svmperf_base, loss, n_jobs=-1, **kwargs):
        self.svmperf_base = svmperf_base
        self.loss = loss
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def fit(self, data: LabelledCollection, fit_learner=True, *args):
        assert fit_learner, 'the method requires that fit_learner=True'

        self.learners = {c: SVMperf(self.svmperf_base, loss=self.loss, **self.kwargs) for c in data.classes_}
        Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(train_task)(c, self.learners, data) for c in self.learners.keys()
        )
        return self

    def quantify(self, X, y=None):
        prevalences = np.asarray(
            Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(binary_quant_task)(c, self.learners, X) for c in self.learners.keys()
            )
        )
        prevalences /= prevalences.sum()
        return prevalences

    @property
    def classes_(self):
        return sorted(self.learners.keys())

    def preclassify_collection(self, data: LabelledCollection):
        classifications = []
        for class_ in data.classes_:
            classifications.append(self.learners[class_].predict(data.documents))
        classifications = np.vstack(classifications).T
        precomputed =  LabelledCollection(classifications, data.labels)
        return precomputed

    def set_params(self, **parameters):
        self.kwargs=parameters

    def get_params(self, deep=True):
        return self.kwargs


class ExplicitLossMinimisation(AggregativeQuantifier):

    def __init__(self, svmperf_base, loss, **kwargs):
        self.learner = SVMperf(svmperf_base, loss=loss, **kwargs)

    def fit(self, data: LabelledCollection, fit_learner=True, *args):
        assert fit_learner, 'the method requires that fit_learner=True'
        self.learner.fit(data.documents, data.labels)
        return self

    def quantify(self, X, y=None):
        predictions = self.learner.predict(X)
        return prevalence_from_labels(predictions, self.learner.n_classes_)

    def classify(self, X, y=None):
        return self.learner.predict(X)


class SVMQ(ExplicitLossMinimisation):
    def __init__(self, svmperf_base, **kwargs):
        super(SVMQ, self).__init__(svmperf_base, loss='q', **kwargs)


class SVMKLD(ExplicitLossMinimisation):
    def __init__(self, svmperf_base, **kwargs):
        super(SVMKLD, self).__init__(svmperf_base, loss='kld', **kwargs)


class SVMNKLD(ExplicitLossMinimisation):
    def __init__(self, svmperf_base, **kwargs):
        super(SVMNKLD, self).__init__(svmperf_base, loss='nkld', **kwargs)


class SVMAE(ExplicitLossMinimisation):
    def __init__(self, svmperf_base, **kwargs):
        super(SVMAE, self).__init__(svmperf_base, loss='mae', **kwargs)


class SVMRAE(ExplicitLossMinimisation):
    def __init__(self, svmperf_base, **kwargs):
        super(SVMRAE, self).__init__(svmperf_base, loss='mrae', **kwargs)

