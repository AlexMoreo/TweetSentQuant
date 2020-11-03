# Tweet Sentiment Quantification: An Experimental Re-Evaluation
## ECIR2021: Reproducibility track

This repo contains the code to reproduce all experiments discussed 
in the paper entitled _Tweet Sentiment Quantification: An Experimental Re-Evaluation_
which is submitted for consideration to the _ECIR2021's track on Reproducibility_

## Requirements
* skicit-learn, numpy, scipy
* svmperf patched for quantification (see below)
* absl-py
* tqdm

A simple way to get started is to create a conda environment from the
configuration file [environment_q.yml](environment_q.yml).
At this point it is useful to run the scripts that prepare the
datasets and the svmperf package (explained below):

```
conda create ecir -f environment_cc.yml
conda activate ecir
git clone https://github.com/AlexMoreo/TweetSentQuant.git
cd TweetSentQuant
chmod +x *.sh
./prepare_datasets.sh
./prepare_svmperf.sh
```

Test that everything works by running:

```
cd src
python3 main.py --dataset hcr --method cc --learner lr
```

### SVM-perf with quantification-oriented losses
In order to run experiments involving SVM(Q), SVM(KLD), SVM(NKLD),
SVM(AE), or SVM(RAE), you have to first download the 
[svmperf](http://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html) 
package, apply the patch 
[svm-perf-quantification-ext.patch](./svm-perf-quantification-ext.patch), and compile the sources.
The script [prepare_svmperf.sh](prepare_svmperf.sh) does all the job. Simply run:

```
./prepare_svmperf.sh
```
The resulting directory [svm_perf_quantification](./svm_perf_quantification) contains the
patched version of _svmperf_ with quantification-oriented losses. Make sure that the variable
_SVM_PERF_HOME_ from [./src/settings.py](./src/settings.py) points to the right path if you
decide to move it somewhere else.

The [svm-perf-quantification-ext.patch](./svm-perf-quantification-ext.patch) is an extension of the patch made available by
[Esuli et al. 2015](https://dl.acm.org/doi/abs/10.1145/2700406?casa_token=8D2fHsGCVn0AAAAA:ZfThYOvrzWxMGfZYlQW_y8Cagg-o_l6X_PcF09mdETQ4Tu7jK98mxFbGSXp9ZSO14JkUIYuDGFG0) 
that allows SVMperf to optimize for
the _Q_ measure as proposed by [Barranquero et al. 2015](https://www.sciencedirect.com/science/article/abs/pii/S003132031400291X) 
and for the _KLD_ and _NKLD_ as proposed by [Esuli et al. 2015](https://dl.acm.org/doi/abs/10.1145/2700406?casa_token=8D2fHsGCVn0AAAAA:ZfThYOvrzWxMGfZYlQW_y8Cagg-o_l6X_PcF09mdETQ4Tu7jK98mxFbGSXp9ZSO14JkUIYuDGFG0)
for quantification.
This patch extends the former by also allowing SVMperf to optimize for 
_AE_ and _RAE_.

## Datasets
The 11 datasets used in this work can be downloaded from 
[here](alt.qcri.org/~wgao/data/SNAM/tweet_sentiment_quantification.zip).
The datasets are in vector form, and in sparse format.

The file [semeval15.test.feature.txt](./datasets/test/semeval15.test.feature.txt)
is corrupted in the zip file (all documents have the 0 label). The
script [repair_semeval15_test.py](repair_semeval15_test.py) replaces
the wrong labels with the correct ones in 
[semeval15.test.labels.npy](.semeval15.test.labels.npy).

In order to prepare the datasets (download and patch the file), simply
run the script:
```
./prepare_datasets.sh
```


## Reproduce Experiments
All experiments and tables reported in the paper can be reproduced by running the script in 
[./src](./src) folder:

```
./experiments.sh
``` 
Each of the experiments runs the [main.py](src/main.py) file with different arguments. 
Run the command:
```
python main.py --help
```
to display the arguments and options:

```
       USAGE: main.py [flags]
flags:

main.py:
  --dataset: the name of the dataset (e.g, sanders)
  --error: error to optimize for in model selection (none acce f1e mae mrae)
    (default: 'mae')
  --learner: a classification learner method (lr svmperf)
  --method: a quantificaton method (cc, acc, pcc, pacc, emq, svmq, svmkld,
    svmnkld, svmae, svmrae)
  --results: where to pickle the results as a pickle containing the true
    prevalences and the estimated prevalences according to the artificial
    sampling protocol
    (default: '../results')
  --results_point: where to pickle the results as a pickle containing the true
    prevalences and the estimated prevalences according to the natural
    prevalence
    (default: '../results_point')
  --sample_size: sampling size
    (default: '100')
    (an integer)
  --seed: a numeric seed for aligning random processes and a suffix to be used
    in the the result file path, e.g., "run0"
    (default: '0')
    (an integer)
``` 
For example, the following command will train and test the _Adjusted Classify & Count_ variant 
with a _LR_ as the learner device for classification, and will perform a grid-search
optimization of hyperparameters in terms of _MAE_ for the dataset Sanders. 
```
python main.py --dataset sanders --method acc --learner lr --error mae
```
The program will produce a pickle file in _../results/sanders-acc-lr-100-mae-run0.pkl_ that contains 
the true prevalences of the sampled used during test (a _np.array_ of 5775 prevalences, 
21x22/2 prevalences x 25 repetitions, according to the _artificial sampling protocol_ with
three classes) and the estimated prevalences 
(a _np.array_ with the 5775 estimations delivered by the ACC method for each of the test 
samples). 

The resulting pickles are used for evaluating and comparing the different runs.
The evaluation of the current run is shown before exiting. In this example:

```
optimization finished: refitting for {'C': 1000.0, 'class_weight': 'balanced'} (score=0.06271) on the whole development set

TrueP->mean(Phat)(std(Phat))
======================
0.000, 0.000, 1.000->[0.023+-0.0162, 0.139+-0.0553, 0.838+-0.0514]
0.000, 0.050, 0.950->[0.016+-0.0191, 0.189+-0.0687, 0.795+-0.0647]
0.000, 0.100, 0.900->[0.016+-0.0210, 0.230+-0.0695, 0.753+-0.0642]
0.000, 0.150, 0.850->[0.025+-0.0283, 0.244+-0.0676, 0.731+-0.0589]
0.000, 0.200, 0.800->[0.016+-0.0193, 0.308+-0.0638, 0.675+-0.0637]
0.000, 0.250, 0.750->[0.019+-0.0256, 0.330+-0.0797, 0.652+-0.0752]
...
0.900, 0.000, 0.100->[0.896+-0.0506, 0.030+-0.0386, 0.074+-0.0352]
0.900, 0.050, 0.050->[0.894+-0.0601, 0.066+-0.0692, 0.040+-0.0259]
0.900, 0.100, 0.000->[0.871+-0.0757, 0.117+-0.0799, 0.012+-0.0222]
0.950, 0.000, 0.050->[0.936+-0.0562, 0.043+-0.0613, 0.021+-0.0233]
0.950, 0.050, 0.000->[0.928+-0.0506, 0.064+-0.0509, 0.008+-0.0139]
1.000, 0.000, 0.000->[0.978+-0.0240, 0.013+-0.0213, 0.010+-0.0143]

Evaluation Metrics:
======================
	mae=0.0642
	mrae=0.9731

I1030 19:53:29.857306 4762131904 app_helper.py:294] saving results in ../results/sanders-acc-lr-100-mae-run0.pkl

Point-Test evaluation:
======================
true-prev=0.164, 0.688, 0.148, estim-prev=0.163, 0.708, 0.129
	mae=0.0136
	mrae=0.0541
```

Note that the first evaluation corresponds to the artificial sampling
protocol, in which a grid of prevalences is explored.
The second evaluation is a single evaluation, carried out in the
test set with _natural prevalence_, i.e., without performing
sampling (as was done in past literature).
