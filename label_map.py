# Copyright (c) Microsoft. All rights reserved.

from .vocab import Vocabulary
from .metrics import compute_acc, compute_f1, compute_mcc, compute_pearson, compute_spearman
# scitail
ScitailLabelMapper = Vocabulary(True)
ScitailLabelMapper.add('neutral')
ScitailLabelMapper.add('entails')

# label map
SNLI_LabelMapper = Vocabulary(True)
SNLI_LabelMapper.add('contradiction')
SNLI_LabelMapper.add('neutral')
SNLI_LabelMapper.add('entailment')

# qnli
QNLILabelMapper = Vocabulary(True)
QNLILabelMapper.add('not_entailment')
QNLILabelMapper.add('entailment')

# rqe
RQE_LabelMapper = Vocabulary(True)
RQE_LabelMapper.add('false') #not entailment
RQE_LabelMapper.add('true') #entailment
# Add false first since index 0 is assigned to label false. Index 1 will be assigned to label true.
# IMDB_LabelMapper = Vocabulary(True)
# IMDB_LabelMapper.add('neg')
# IMDB_LabelMapper.add('pos')
# done
GLOBAL_MAP = {
 'scitail': ScitailLabelMapper,
 'mednli': SNLI_LabelMapper, # BioNLP
 'rqe': RQE_LabelMapper, # BioNLP
 'mnli': SNLI_LabelMapper,
 'snli': SNLI_LabelMapper,
 'qnli': QNLILabelMapper,
 'qnnli': QNLILabelMapper,
 'rte': QNLILabelMapper,
 'diag': SNLI_LabelMapper
 # 'yelp':YELP_LabelMapper,
 # 'imdb':IMDB_LabelMapper,
 # 'nnlp':NNLP_LabelMapper
}

# done
# number of class/labels. NLI has 3 (entailment, neutral, contradiction). Entailment has 2 (entailment,non-entailment)
DATA_META = {
 'mnli': 3,
 'snli': 3,
 'mednli':3,
 'rqe':2,
 'scitail': 2,
 'qqp': 2,
 'qnli': 2,
 'qnnli': 1,
 'wnli': 2,
 'rte': 2,
 'mrpc': 2,
 'diag': 3,
 'sst': 2,
 'stsb': 1,
 'cola': 2,
 'yelp':5,
 'imdb':2,
 'nnlp':2
}

# done
# Whether input data is a sentence pair or just one sentence. 0 signifies sentence pair. 1 is single sentence. 
DATA_TYPE = {
 'mnli': 0,
 'snli': 0,
 'scitail': 0,
 'mednli': 0,
 'rqe': 0,
 'qqp': 0,
 'qnli': 0,
 'qnnli': 0,
 'wnli': 0,
 'rte': 0,
 'mrpc': 0,
 'diag': 0,
 'sst': 1,
 'stsb': 0,
 'cola': 1,
 'yelp':1,
 'imdb':1,
 'nnlp':1,
}

# TODO understand this
DATA_SWAP = {
 'mnli': 0,
 'snli': 0,
 'scitail': 0,
 'qqp': 1,
 'qnli': 0,
 'qnnli': 0,
 'wnli': 0,
 'rte': 0,
 'mrpc': 0,
 'diag': 0,
 'sst': 0,
 'stsb': 0,
 'cola': 0,
 'yelp':0,
 'imdb':0,
 'nnlp':0,
}

#done
# classification/regression
TASK_TYPE = {
 'mnli': 0,
 'snli': 0,
 'scitail': 0,
 'mednli':0,
 'rqe':0,
 'qqp': 0,
 'qnli': 0,
 'qnnli': 0,
 'wnli': 0,
 'rte': 0,
 'mrpc': 0,
 'diag': 0,
 'sst': 0,
 'stsb':1,
 'cola': 0,
 'yelp':0,
 'imdb':0,
 'nnlp':0,
}

#done
METRIC_META = {
 'mnli': [0],
 'snli': [0],
 'mednli':[0],
 'rqe':[0],
 'scitail': [0],
 'qqp': [0, 1],
 'qnli':[0],
 'qnnli': [0],
 'wnli': [0],
 'rte': [0],
 'mrpc': [0, 1],
 'diag': [0],
 'sst': [0],
 'stsb': [3, 4],
 'cola': [0, 2],
 'yelp':[0, 1],
 'imdb':[0, 1],
 'nnlp':[0, 1],
}

METRIC_NAME = {
 0: 'ACC',
 1: 'F1',
 2: 'MCC',
 3: 'Pearson',
 4: 'Spearman',
}

METRIC_FUNC = {
 0: compute_acc,
 1: compute_f1,
 2: compute_mcc,
 3: compute_pearson,
 4: compute_spearman,
}

SAN_META = {
    'mnli': 1,
    'snli': 1,
    'scitail': 1,
    'mednli': 1,
    'rqe': 1,
    'qqp': 1,
    'qnli': 1,
    'qnnli': 1,
    'wnli': 1,
    'rte': 1,
    'mrpc': 1,
    'diag': 0,
    'sst': 0,
    'stsb': 0,
    'cola': 0,
    'yelp':0,
    'imdb':0,
    'nnlp':0,
}

def generate_decoder_opt(task, max_opt):
    assert task in SAN_META
    opt_v = 0
    if SAN_META[task] and max_opt < 3:
        opt_v = max_opt
    return opt_v