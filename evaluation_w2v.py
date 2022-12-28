from cmath import nan
from typing import List
from sudachipy import tokenizer
from sudachipy import dictionary
from gensim.models import KeyedVectors
import sys
import numpy as np
import logging
import argparse
from prettytable import PrettyTable


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)




class Tokenizer(object):
    """
    Sudachiによる単語の分割
    """

    def __init__(self, hinshi_list: List[str] = None, split_mode: str = "C"):
        """
        :param hinshi_list: 使用する品詞のリスト. example) hinshi_list=["動詞", "名詞", "形容詞"]
        :param split_mode:
        """
        split_mode_list = ["A", "B", "C"]
        assert split_mode in split_mode, f"{split_mode} is a non-existent split_mode {split_mode_list}"
        split_dic = {
            "A": tokenizer.Tokenizer.SplitMode.A,
            "B": tokenizer.Tokenizer.SplitMode.B,
            "C": tokenizer.Tokenizer.SplitMode.C,
        }
        self.tokenizer_obj = dictionary.Dictionary().create()
        self.mode = split_dic[split_mode]
        self.hinshi_list = hinshi_list

    def __call__(self, text: str) -> str:
        if self.hinshi_list:
            return " ".join([m.normalized_form() for m in self.tokenizer_obj.tokenize(text, self.mode) if
                             m.part_of_speech()[0] in self.hinshi_list and m.normalized_form() != " "])
        return " ".join(
            m.normalized_form() for m in self.tokenizer_obj.tokenize(text, self.mode) if m.normalized_form() != " ")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_set", type=str, 
            choices=['sts', 'transfer', 'full', 'na'],
            default='sts',
            help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+', 
            default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                     'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                     'SICKRelatedness', 'STSBenchmark', 'JSTS', 'JACSTS'], 
            help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")
    
    args = parser.parse_args()
    
    # Load transformers' model checkpoint
    model = KeyedVectors.load_word2vec_format("/home/chen/cc.ja.300.vec.gz")
    #model = KeyedVectors.load("/home/chen/SlackBotPastRelevantQuestions/data/model/jawiki.word_vectors.300d.bin")
    tokenizer = Tokenizer(hinshi_list=["動詞", "名詞", "形容詞"], split_mode="A")
    
    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness', 'JSTS']
    #elif args.task_set == 'transfer':
        #args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'transfer':
        args.tasks = ['JACSTS']    
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness', 'JSTS']
        #args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
        args.tasks += ['JACSTS']

    # Set params for SentEval
    params = {'task_path': PATH_TO_DATA,  'kfold': 10}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}


    # SentEval prepare and batcher
    def prepare(params, samples):
        return
    
    def batcher(params, batch, max_length=None):
        embeddings = []
        for sent in batch:
            sentvec = []
            if sent != None:
                word_list = tokenizer(str(sent)).split()
                for word in word_list:
                    if word in model.key_to_index:
                       sentvec.append(model.get_vector(word))
                    else:
                       sentvec.append(np.random.uniform(-0.01, 0.01, model.vector_size))
                if not sentvec:
                    sentvec.append(np.random.uniform(-0.01, 0.01, model.vector_size))
                sentvec = np.mean(sentvec, 0)
                embeddings.append(sentvec)
            else:
                sentvec= np.random.uniform(-0.01, 0.01, model.vector_size)   
                embeddings.append(sentvec)
        embeddings = np.vstack(embeddings)

        return embeddings


    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result
    
    # Print evaluation results
    
     
    task_names = []
    scores = []

    gold_task_names = []
    gold_scores = []

    for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness', 'JSTS']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")

    for task in ['SICKRelatedness', 'JSTS']:
            gold_task_names.append(task)
            if task in results:
                if task in ['SICKRelatedness', 'JSTS']:
                    gold_scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                gold_scores.append("0.00")       

    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    gold_task_names.append("Avg_gold.")
    gold_scores.append("%.2f" % (sum([float(gold_score) for gold_score in gold_scores]) / len(gold_scores)))


    print_table(task_names, scores)
    print_table(gold_task_names, gold_scores)

    task_names = []
    scores = []
    for task in ['JACSTS']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))    
            else:
                scores.append("0.00")
    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    print_table(task_names, scores)    


if __name__ == "__main__":
    main()            