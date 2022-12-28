import spacy
from tqdm import tqdm

import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

nlp = spacy.load('ja_ginza')

sentences = []
for line in open('/home/chen/T5_mask_infilling/clinic_corpus.txt', 'r'):
    if line is not None:
        sentences.append(line.strip())


T5_PATH = '/home/chen/T5_mask_infilling/torch_model_clinic' 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
t5_config = T5Config.from_pretrained(T5_PATH)
t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE) 


def _filter(output, end_token='<extra_id_1>'):
    # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)
    _txt = t5_tokenizer.decode(output[2:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if end_token in _txt:
        _end_token_index = _txt.index(end_token)
        return _result_prefix + _txt[:_end_token_index] + _result_suffix
    else:
        return _result_prefix + _txt + _result_suffix

gen_list = []
for i in tqdm(range(len(sentences))):
    noun_tokens = []
    input_text = sentences[i]
    re_input_text = input_text
    doc = nlp(input_text)
    for np in doc.noun_chunks:
        noun_tokens.append(np.text)
                    
    for i in range(len(noun_tokens)):
            new_text = re_input_text.replace(noun_tokens[i], '<extra_id_0>', 1)
            encoded = t5_tokenizer(new_text, add_special_tokens=True, return_tensors='pt', 
                       padding = 'max_length', truncation = True, max_length = 64)
            input_ids = encoded['input_ids'].to(DEVICE)
        
            outputs = t5_mlm.generate(input_ids = input_ids, 
                          num_beams=200, num_return_sequences=2,
                          max_length=15)

            _0_index = new_text.index('<extra_id_0>')
            _result_prefix = new_text[:_0_index]
            _result_suffix = new_text[_0_index+12:]  # 12 is the length of <extra_id_0>

            results = list(map(_filter, outputs))
            re_input_text = results[1]
            
    gen_list.append(re_input_text)



file_out = 'gen_contradict_forsup_v1.1_top2.txt'
with open(file_out, 'w', encoding = 'utf-8') as f:
    for i in range(len(gen_list)):
        f.write(gen_list[i].strip() + '\n')
     
    
f.close() 