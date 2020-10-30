#!/bin/bash
set -x

mkdir -p ./datasets

wget alt.qcri.org/~wgao/data/SNAM/tweet_sentiment_quantification.zip
unzip tweet_sentiment_quantification.zip -d datasets
rm tweet_sentiment_quantification.zip

python3 repair_semeval15_test.py











