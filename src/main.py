from absl import app
from app_helper import *

flags.DEFINE_string('dataset', None, 'the name of the dataset (e.g, sanders)')
flags.DEFINE_string('method', None, 'a quantificaton method '
                                    '(cc, acc, pcc, pacc, emq, svmq, svmkld, svmnkld, svmae, svmrae)')
flags.DEFINE_string('learner', None, 'a classification learner method (lr svmperf)')
flags.DEFINE_integer('sample_size', settings.SAMPLE_SIZE, 'sampling size')
flags.DEFINE_string('error', 'mae', 'error to optimize for in model selection (none acce f1e mae mrae)')
flags.DEFINE_string('results', '../results', 'where to pickle the results as a pickle containing the true prevalences '
                                         'and the estimated prevalences according to the artificial sampling protocol')
flags.DEFINE_string('results_point', '../results_point', 'where to pickle the results as a pickle containing the true '
                                      'prevalences and the estimated prevalences according to the natural prevalence')
flags.DEFINE_integer('seed', 0, 'a numeric seed for aligning random processes and a suffix to be used in the the '
                                'result file path, e.g., "run0"')
flags.mark_flags_as_required(['dataset', 'method', 'learner'])

FLAGS = flags.FLAGS


def main(_):
    set_random_seed()

    benchmark_ms = load_dataset_model_selection()

    learner = instantiate_learner()
    method = instantiate_quantifier(learner)
    method = model_selection(method, benchmark_ms)

    # decide the test to be performed (in the case of 'semeval', tests are 'semeval13', 'semeval14', 'semeval15')
    if FLAGS.dataset == 'semeval':
        test_sets = ['semeval13', 'semeval14', 'semeval15']
    else:
        test_sets = [FLAGS.dataset]

    for test_set in test_sets:
        benchmark_eval = load_dataset_model_evaluation(test_set)
        method.fit(benchmark_eval.training)
        true_prevalences, estim_prevalences = produce_predictions(method, benchmark_eval.test)
        evaluate_experiment(true_prevalences, estim_prevalences, test_name=test_set)
        evaluate_method_point_test(method, benchmark_eval.test, test_name=test_set)


if __name__ == '__main__':
    app.run(main)
