import numpy as np
from tqdm import tqdm
import os

labels = np.load('semeval15.test.labels.npy')

assert not os.path.exists('./datasets/test/semeval15.test.feature.bu.txt'), 'file already patched!'

semeval15_path = './datasets/test/semeval15.test.feature.txt'
semeval15_bu_path = './datasets/test/semeval15.test.feature.bu.txt'
os.rename(semeval15_path, semeval15_bu_path)

with open(semeval15_bu_path, 'rt') as fin, open(semeval15_path, 'wt') as foo:
    for i,line in tqdm(enumerate(fin.readlines()), desc='patching semeval15.test labels'):
        foo.write(f'{labels[i]} {line[2:]}')
print("[Done]")


