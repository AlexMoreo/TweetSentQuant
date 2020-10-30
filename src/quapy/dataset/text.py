import numpy as np
from scipy.sparse import issparse, dok_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from quapy.functional import artificial_prevalence_sampling
from scipy.sparse import csr_matrix, vstack


class LabelledCollection:

    def __init__(self, documents, labels, n_classes=None):
        self.documents = documents if issparse(documents) else np.asarray(documents)
        self.labels = np.asarray(labels, dtype=int)
        n_docs = len(self)
        if n_classes is None:
            self.classes_ = np.unique(self.labels)
            self.classes_.sort()
        else:
            self.classes_ = np.arange(n_classes)
        self.index = {class_i: np.arange(n_docs)[self.labels == class_i] for class_i in self.classes_}

    @classmethod
    # File fomart <0 or 1>\t<document>\n
    def from_file(cls, path):
        all_sentences, all_labels = [], []
        for line in tqdm(open(path, 'rt').readlines(), f'loading {path}'):
            line = line.strip()
            if line:
                label, sentence = line.split('\t')
                sentence = sentence.strip()
                label = int(label)
                if sentence:
                    all_sentences.append(sentence)
                    all_labels.append(label)
        return LabelledCollection(all_sentences, all_labels)

    @classmethod
    def from_sparse(cls, path):
        def split_col_val(col_val):
            col, val = col_val.split(':')
            col, val = int(col) - 1, float(val)
            return col, val

        all_documents, all_labels = [], []
        max_col = 0
        for line in tqdm(open(path, 'rt').readlines(), f'loading {path}'):
            parts = line.strip().split()
            if parts:
                all_labels.append(int(parts[0]))
                cols, vals = zip(*[split_col_val(col_val) for col_val in parts[1:]])
                cols, vals = np.asarray(cols), np.asarray(vals)
                max_col = max(max_col, cols.max())
                all_documents.append((cols, vals))
        n_docs = len(all_labels)
        X = dok_matrix((n_docs, max_col + 1), dtype=float)
        for i, (cols, vals) in tqdm(enumerate(all_documents), total=len(all_documents),
                                    desc=f'converting matrix of shape {X.shape}'):
            X[i, cols] = vals
        X = X.tocsr()
        y = np.asarray(all_labels) + 1
        return LabelledCollection(X, y)

    def __len__(self):
        return self.documents.shape[0]

    def prevalence(self):
        return self.counts()/len(self)

    def counts(self):
        return np.asarray([len(self.index[ci]) for ci in self.classes_])

    @property
    def n_classes(self):
        return len(self.classes_)

    def sampling_index(self, size, *prevs, shuffle=True):
        if len(prevs) == self.n_classes-1:
            prevs = prevs + (1-sum(prevs),)
        assert len(prevs) == self.n_classes, 'unexpected number of prevalences'
        assert sum(prevs) == 1, f'prevalences ({prevs}) out of range (sum={sum(prevs)})'

        taken = 0
        indexes_sample = []
        for i, class_i in enumerate(self.classes_):
            if i == self.n_classes-1:
                n_requested = size - taken
            else:
                n_requested = int(size * prevs[i])

            n_candidates = len(self.index[class_i])
            index_sample = self.index[class_i][
                np.random.choice(n_candidates, size=n_requested, replace=(n_requested > n_candidates))
            ] if n_requested > 0 else []

            indexes_sample.append(index_sample)
            taken += n_requested

        indexes_sample = np.concatenate(indexes_sample).astype(int)

        if shuffle:
            indexes_sample = np.random.permutation(indexes_sample)

        return indexes_sample

    def sampling(self, size, *prevs, shuffle=True):
        index = self.sampling_index(size, *prevs, shuffle=shuffle)
        return self.sampling_from_index(index)

    def sampling_from_index(self, index):
        documents = self.documents[index]
        labels = self.labels[index]
        return LabelledCollection(documents, labels, n_classes=self.n_classes)

    def split_stratified(self, train_size=0.6):
        tr_docs, te_docs, tr_labels, te_labels = \
            train_test_split(self.documents, self.labels, train_size=train_size, stratify=self.labels)
        return LabelledCollection(tr_docs, tr_labels), LabelledCollection(te_docs, te_labels)

    def artificial_sampling_generator(self, sample_size, n_prevalences=101, repeats=1):
        dimensions=self.n_classes
        for prevs in artificial_prevalence_sampling(dimensions, n_prevalences, repeats):
            yield self.sampling(sample_size, *prevs)

    def __add__(self, other):
        if issparse(self.documents) and issparse(other.documents):
            docs = vstack([self.documents, other.documents])
        elif isinstance(self.documents, list) and isinstance(other.documents, list):
            docs = self.documents + other.documents
        else:
            raise NotImplementedError('unsupported operation for collection types')
        labels = np.concatenate([self.labels, other.labels])
        return LabelledCollection(docs, labels)


class TQDataset:

    def __init__(self, training: LabelledCollection, test: LabelledCollection):
        self.training = training
        self.test = test

    def tfidfvectorize(self, min_freq=3):
        vectorizer = TfidfVectorizer(min_df=min_freq, sublinear_tf=True)
        self.training.documents = vectorizer.fit_transform(self.training.documents)
        self.test.documents = vectorizer.transform(self.test.documents)
        self.vocabulary_ = vectorizer.vocabulary_

    @classmethod
    def from_files(cls, train_path, test_path):
        training = LabelledCollection.from_file(train_path)
        test = LabelledCollection.from_file(test_path)
        return TQDataset(training, test)

    @classmethod
    def from_sparse(cls, train_path, test_path, min_word_freq=5):
        training = LabelledCollection.from_sparse(train_path)
        test = LabelledCollection.from_sparse(test_path)
        assert training.documents.shape[1] == test.documents.shape[1], 'unaligned vector spaces'
        if min_word_freq>1:
            training.documents, test.documents = filter_by_occurrences(training.documents, test.documents, min_word_freq)
        return TQDataset(training, test)

    @property
    def n_classes(self):
        return self.training.n_classes


def filter_by_occurrences(X, W, min_word_freq):
    word_presence_count = np.asarray((X>0).sum(axis=0)).flatten()
    take_columns = word_presence_count>=min_word_freq
    former_dim = X.shape[1]
    X = X[:, take_columns]
    W = W[:, take_columns]
    later_dim = X.shape[1]
    print(f'reducing from {former_dim} -> {later_dim} columns with at least {min_word_freq} occurrences')
    return X, W


