import numpy as np
import pandas as pd
from typing import Union, List
import evaluate
from gensim.models import KeyedVectors
from sudachipy import tokenizer
from sudachipy import dictionary


class Tokenizer(object):
    """
    Sudachiによる単語の分割
    """

    def __init__(self, hinshi_list: List[str] = None, split_mode: str = "A"):
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


class SwemEncoder():
    """
    SWEMを計算するエンコーダー
    """

    def __init__(self, model_path: str) -> None:
        self.model = KeyedVectors.load_word2vec_format(model_path)
        self.tokenizer = Tokenizer(hinshi_list=["動詞", "名詞", "形容詞"], split_mode="A")
    
    def _get_word_embeddings(self, sentence: str) -> np.array:
        """
        単語の分散表現のリストを取得する
        :param sentence: 単語リスト
        :return: np.array, shape(len(word_list), self.vector_size)
        """
        vectors = []
        if sentence != None:
            word_list = self.tokenizer(sentence).split()

            for word in word_list:
                if word in self.model.key_to_index:
                    vectors.append(self.model.get_vector(word))
                else:
                    vectors.append(np.random.uniform(-0.01, 0.01, self.model.vector_size))

            if not vectors:
                vectors.append(np.random.uniform(-0.01, 0.01, self.model.vector_size))

            vectors = np.mean(vectors, 0)    
        else:
            vectors = np.random.uniform(-0.01, 0.01, self.model.vector_size)
 
        return  vectors      

    def get_vector(self, singlesentence: str) -> np.array:
        swemvector=(self._get_word_embeddings(singlesentence))
        
        return np.array(swemvector)

    def get_matrix(self, sentences: Union[List[str], pd.Series]) -> np.array:
        swems = []
        for sentence in sentences:
            swems.append(self._get_word_embeddings(sentence))

        return np.concatenate(swems)            


def main():

    model_path = "./cc.ja.300.vec.gz" 

    
    model_encoder = SwemEncoder(model_path=model_path)
    evaluate.metric_ensemble(model_encoder)

if __name__ == "__main__":
    main()        