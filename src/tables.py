from os import makedirs
from evaluate import evaluate_directory, statistical_significance, get_ranks_from_Gao_Sebastiani
import settings
from quapy.error import *

result_path = '../results'
tables_path = '../tables'
MAXTONE = 50  # sets the intensity of the maximum color reached by the worst (red) and best (green) results

makedirs(tables_path, exist_ok=True)

sample_length=settings.SAMPLE_SIZE

datasets = ['gasp', 'hcr', 'omd', 'sanders', 'semeval13', 'semeval14', 'semeval15', 'semeval16', 'sst', 'wa', 'wb']
evaluation_measures = [mae, mrae]

results_dict = evaluate_directory('../results/*.pkl', evaluation_measures)
stats = {
    dataset : {
        'mae': statistical_significance(f'../results/{dataset}-*-mae-run?.pkl', ae),
        'mrae': statistical_significance(f'../results/{dataset}-*-mrae-run?.pkl', rae),
    } for dataset in datasets
}

nice = {
    'mae':'AE',
    'mrae':'RAE',
    'svmkld': 'SVM(KLD)',
    'svmnkld': 'SVM(NKLD)',
    'svmq': 'SVM(Q)',
    'svmae': 'SVM(AE)',
    'svmnae': 'SVM(NAE)',
    'svmmae': 'SVM(AE)',
    'svmmrae': 'SVM(RAE)',
    'quanet': 'QuaNet',
    'hdy': 'HDy',
    'dys': 'DyS',
    'svmperf':'',
    'sanders': 'Sanders',
    'semeval13': 'SemEval13',
    'semeval14': 'SemEval14',
    'semeval15': 'SemEval15',
    'semeval16': 'SemEval16'
}
#     }
# }


def nicerm(key):
    return '\mathrm{'+nice[key]+'}'

def color_from_rel_rank(rel_rank, maxtone=100):
    rel_rank = rel_rank*2-1
    if rel_rank < 0:
        color = 'red'
        tone = maxtone*(-rel_rank)
    else:
        color = 'green'
        tone = maxtone*rel_rank
    return '\cellcolor{' + color + f'!{int(tone)}' + '}'

def color_from_abs_rank(abs_rank, n_methods, maxtone=100):
    rel_rank = 1.-(abs_rank-1.)/(n_methods-1)
    return color_from_rel_rank(rel_rank, maxtone)


def save_table(path, table):
    print(f'saving results in {path}')
    with open(path, 'wt') as foo:
        foo.write(table)


# Tables evaluation scores for AE and RAE (two tables)
# ----------------------------------------------------
for i, eval_func in enumerate(evaluation_measures):
    optim = eval_func.__name__
    methods = ['cc', 'acc', 'pcc', 'pacc', 'emq', 'svmq', 'svmkld', 'svmnkld', 'svm'+optim] #, 'quanet', 'dys']
    table = """
    \\begin{table}[h]
    """
    if i == 0:
        caption = """
          \caption{Values of AE obtained in our experiments; each value is the
          average across 5775 values, each obtained on a different sample.
          \\textbf{Boldface} indicates the best method for a given dataset. 
          Superscripts $\dag$ and $\dag\dag$ denote the
          methods (if any) whose score are not statistically significantly
          different from the best one according to a paired sample, two-tailed 
          t-test at different confidence levels: symbol $\dag$ indicates 
          $0.001<p$-value $<0.05$ while symbol $\dag\dag$ indicates 
          $0.05\leq p$-value. The absence of any such symbol indicates
          $p$-value $\leq 0.001$.
          }
        """
    else:
        caption = "\caption{As Table~\\ref{tab:maeresults}, but with "+nice[optim]+" instead of AE.}"
    table += caption + """
            \\begin{center}
            \\resizebox{\\textwidth}{!}{
        """

    tabular = """
        \\begin{tabularx}{\\textwidth}{|c||Y|Y|Y|Y|Y|Y|Y|Y||Y|} \hline
          & \multicolumn{8}{c||}{Methods tested in~\cite{Gao:2016uq}} &  \\\\ \hline
    """

    for method in methods:
        tabular += ' & \side{'+nice.get(method, method.upper())+'$^{'+nicerm(optim)+'}$} '
    tabular += '\\\\\hline\n'

    for dataset in datasets:
        tabular += nice.get(dataset, dataset.upper()) + ' '
        for method in methods:
            learner = 'lr' if not method.startswith('svm') else 'svmperf'
            key = f'{dataset}-{method}-{learner}-{settings.SAMPLE_SIZE}-{optim}'
            if key+'-'+optim in results_dict:
                score = results_dict[key+'-'+optim]
                stat_sig, rel_rank, abs_rank = stats[dataset][optim][key]
                if stat_sig=='best':
                    tabular += '& \\textbf{'+ f'{score:.3f}'+'}' + '$\phantom{\dag}\phantom{\dag}$'
                elif stat_sig=='verydifferent':
                    tabular += f'& {score:.3f}' + '$\phantom{\dag}\phantom{\dag}$'
                elif stat_sig=='different':
                    tabular += f'& {score:.3f}'+'$\dag\phantom{\dag}$'
                elif stat_sig=='nondifferent':
                    tabular += f'& {score:.3f}'+'$\dag\dag$'
                else:
                    print('stat sig error: ' + stat_sig)
                print(key)
                try:
                    tabular += color_from_rel_rank(rel_rank, maxtone=MAXTONE)
                except ValueError:
                    tabular += ''
            else:
                tabular += ' & --- '
        tabular += '\\\\\hline\n'
    tabular += "\end{tabularx}"
    table += tabular + """
        }
      \end{center}
      \label{tab:"""+optim+"""results}
    \end{table}
    """
    save_table(f'../tables/tab_results_{optim}.tex', table)


gao_seb_ranks, gao_seb_results = get_ranks_from_Gao_Sebastiani()

# Tables ranks for AE and RAE (two tables)
# ----------------------------------------------------
for i, eval_func in enumerate(evaluation_measures):
    optim = eval_func.__name__
    methods = ['cc', 'acc', 'pcc', 'pacc', 'emq', 'svmq', 'svmkld', 'svmnkld']
    table = """
    \\begin{table}[h]
    """
    if i == 0:
        caption = """
          \caption{Rank positions of the quantification methods in the AE
          experiments, and (between parentheses) the rank positions 
          obtained in the evaluation of~\cite{Gao:2016uq}.}
        """
    else:
        caption = "\caption{Same as Table~\\ref{tab:maeranks}, but with "+nice[optim]+" instead of AE.}"
    table += caption + """
            \\begin{center}
            \\resizebox{\\textwidth}{!}{
        """
    tabular = """
        \\begin{tabularx}{\\textwidth}{|c||Y|Y|Y|Y|Y|Y|Y|Y|} \hline
          & \multicolumn{8}{c|}{Methods tested in~\cite{Gao:2016uq}}  \\\\ \hline
    """

    for method in methods:
        tabular += ' & \side{'+nice.get(method, method.upper())+'$^{'+nicerm(optim)+'}$} '
    tabular += '\\\\\hline\n'

    for dataset in datasets:
        tabular += nice.get(dataset, dataset.upper()) + ' '
        ranks_no_gap = []
        for method in methods:
            learner = 'lr' if not method.startswith('svm') else 'svmperf'
            key = f'{dataset}-{method}-{learner}-{settings.SAMPLE_SIZE}-{optim}'
            ranks_no_gap.append(stats[dataset][optim].get(key,(None,None,len(methods)))[2])
        ranks_no_gap = sorted(ranks_no_gap)
        ranks_no_gap = {rank:i+1 for i,rank in enumerate(ranks_no_gap)}
        for method in methods:
            learner = 'lr' if not method.startswith('svm') else 'svmperf'
            key = f'{dataset}-{method}-{learner}-{settings.SAMPLE_SIZE}-{optim}'
            if key in stats[dataset][optim]:
                _, _, abs_rank = stats[dataset][optim][key]
                real_rank = ranks_no_gap[abs_rank]
                tabular += f' & {real_rank}'
                tabular += color_from_abs_rank(real_rank, len(methods), maxtone=MAXTONE)
            else:
                tabular += ' & --- '
            old_rank = gao_seb_ranks.get(f'{dataset}-{method}-{optim}', 'error')
            tabular += f' ({old_rank})'
        tabular += '\\\\\hline\n'
    tabular += "\end{tabularx}"
    table += tabular + """
        }
      \end{center}
      \label{tab:"""+optim+"""ranks}
    \end{table}
    """
    save_table(f'../tables/tab_rank_{optim}.tex', table)


print("[Done]")