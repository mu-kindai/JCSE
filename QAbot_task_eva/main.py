from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from typing import Union, List
from tqdm import tqdm
import evaluate
import argparse



class JapaneseModel():
    def __init__(self, model_path, pooling_type, input_length):  
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling_type = pooling_type
        self.input_length = input_length
        self.model.eval()
        self.model.to(self.device)      

    def pooling_output(self, encoded_input):

        outputs = self.model(**encoded_input, output_hidden_states = True, return_dict=True)
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states


        if self.pooling_type == 'cls':
            # There is a linear+activation layer after CLS representation
            return pooler_output.cpu()
        elif self.pooling_type == 'cls_before_pooler':
            return last_hidden[:, 0].cpu()
        elif self.pooling_type == "avg":
            return ((last_hidden * encoded_input['attention_mask'].unsqueeze(-1)).sum(1) / encoded_input['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
        elif self.pooling_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * encoded_input['attention_mask'].unsqueeze(-1)).sum(1) / encoded_input['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        elif self.pooling_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * encoded_input['attention_mask'].unsqueeze(-1)).sum(1) / encoded_input['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        else:
            raise NotImplementedError

    def get_vector(self, sentence: str) -> np.array:
        encoded_input = self.tokenizer.encode_plus(sentence, padding=True, max_length = self.input_length,
                                       truncation=True, add_special_tokens = True, return_tensors="pt").to(self.device)
        
        sentence_embedding = self.pooling_output(encoded_input)
        return sentence_embedding.detach().clone().numpy()


    def get_matrix(self, sentences: Union[List[str], pd.Series], batch_size=64) -> np.array:
        embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]
            encoded_input = self.tokenizer.batch_encode_plus(batch, padding=True, max_length = self.input_length,
                                       truncation=True, add_special_tokens = True, return_tensors="pt").to(self.device)
            sentence_embeddings = self.pooling_output(encoded_input)
            embeddings.append(sentence_embeddings.detach().clone().numpy())
        return np.concatenate(embeddings, axis=0)    




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help="Transformers' model name or path")
    args = parser.parse_args() 
    model_path = args.model_name_or_path 

    
    model_encoder = JapaneseModel(model_path=model_path, pooling_type='avg', input_length=64)
    mrr, map, p1, p5 = evaluate.metric_ensemble(model_encoder)

"""     f_out = f"{model_path}/qabot_results.txt"
    with open(f_out, 'w', encoding = 'utf-8') as f:
             f.write(f"MRR: {mrr:.4f}" + "\n" +
                  f"MAP: {map:.4f}" + "\n" +
                  f"Precision@1: {p1:.4f}" + "\n" +
                  f"Precision@5: {p5:.4f}")
    f.close() """

if __name__ == "__main__":
    main()
