from .aggregative import *
from .non_aggregative import *

AGGREGATIVE_METHODS = {
    ClassifyAndCount,
    AdjustedClassifyAndCount,
    ProbabilisticClassifyAndCount,
    ProbabilisticAdjustedClassifyAndCount,
    ExplicitLossMinimisation,
    ExpectationMaximizationQuantifier,
}

NON_AGGREGATIVE_METHODS = {
    MaximumLikelihoodPrevalenceEstimation
}

QUANTIFICATION_METHODS = AGGREGATIVE_METHODS | NON_AGGREGATIVE_METHODS


# common alisases
CC = ClassifyAndCount
ACC = AdjustedClassifyAndCount
PCC = ProbabilisticClassifyAndCount
PACC = ProbabilisticAdjustedClassifyAndCount
ELM = ExplicitLossMinimisation
EMQ = ExpectationMaximizationQuantifier
MLPE = MaximumLikelihoodPrevalenceEstimation


