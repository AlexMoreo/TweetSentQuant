#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:.

PY='python3 main.py'
learner=lr
# semeval={semeval13 semeval14 and semeval15(fixed)}
datasets="sanders hcr wb gasp omd wa sst semeval16 semeval" #

error=mae
for dataset in $datasets ; do
  #$PY --dataset $dataset --method mlpe --learner none --error none --seed 0
  $PY --dataset $dataset --method cc --learner $learner --error $error --seed 0 &
  $PY --dataset $dataset --method acc --learner $learner --error $error --seed 0 &
  $PY --dataset $dataset --method pcc --learner $learner --error $error --seed 0 &
  $PY --dataset $dataset --method pacc --learner $learner --error $error --seed 0 &
  $PY --dataset $dataset --method emq --learner $learner --error $error --seed 0 &
  wait
  $PY --dataset $dataset --method svmq --learner svmperf --error $error --seed 0 &
  $PY --dataset $dataset --method svmkld --learner svmperf --error $error --seed 0 &
  $PY --dataset $dataset --method svmnkld --learner svmperf --error $error --seed 0 &
  $PY --dataset $dataset --method svmmae --learner svmperf --error $error --seed 0 &
  wait
done

exit

error=mrae
for dataset in $datasets ; do
  #$PY --dataset $dataset --method mlpe --learner none --error none --seed 0
#  $PY --dataset $dataset --method cc --learner $learner --error $error --seed 0 &
#  $PY --dataset $dataset --method acc --learner $learner --error $error --seed 0 &
#  $PY --dataset $dataset --method pcc --learner $learner --error $error --seed 0 &
#  $PY --dataset $dataset --method pacc --learner $learner --error $error --seed 0 &
#  $PY --dataset $dataset --method emq --learner $learner --error $error --seed 0 &
  $PY --dataset $dataset --method svmq --learner svmperf --error $error --seed 0 &
  $PY --dataset $dataset --method svmkld --learner svmperf --error $error --seed 0 &
  $PY --dataset $dataset --method svmnkld --learner svmperf --error $error --seed 0 &
  $PY --dataset $dataset --method svmmrae --learner svmperf --error $error --seed 0 &
  wait
done

python3 tables.py



