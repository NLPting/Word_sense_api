import torch
from collections import defaultdict
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy
import numpy as np
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from sklearn.metrics.pairwise import cosine_similarity
import json
import jsonlines
import re
from pprint import pprint
import os


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')
nlp = spacy.load(os.environ.get('SPACY_MODEL', 'en'))

print('Loading.......ok~')


def alignment_bert_spacy_tokens(spacy_tokens, bert_tokens):
    # rules for bert
    
    # rule (1)
    # "##" tokens starts with ## means continue words
    # like christmas will be split into ['ch','##rist','##mas' ]
    
    # rule (2)
    # bert unknown words is [UNK]
    
    #lower both tokens
    spacy_tokens = [token.lower() for token in spacy_tokens]
    bert_tokens = [token.lower() for token in bert_tokens]
    
    max_idx = len(spacy_tokens)
    spacy_idx = 0
    alignments = []
    
    spacy_word_start = 0
    
    for idx in range(len(bert_tokens)):
        
        # UNK case
        if bert_tokens[idx] == "[unk]":
            if spacy_word_start == len(spacy_tokens[spacy_idx]):
                spacy_word_start = 0
                spacy_idx += 1
            alignments.append(spacy_idx)
            continue
        
        if spacy_idx == max_idx:
            alignments.append(spacy_idx)
            continue
        
        # continue word case
        if bert_tokens[idx].startswith("##"):
            token = bert_tokens[idx][2:]
            tgt_idx = spacy_tokens[spacy_idx][spacy_word_start:].find(token)
            spacy_word_start += tgt_idx
            alignments.append(spacy_idx)
        else:
            token = bert_tokens[idx]
            tgt_idx = spacy_tokens[spacy_idx][spacy_word_start:].find(token)
            while tgt_idx == -1 and spacy_idx != max_idx:
                spacy_word_start = 0
                spacy_idx += 1
                tgt_idx = spacy_tokens[spacy_idx][spacy_word_start:].find(token)
            
            if tgt_idx != -1:
                spacy_word_start += tgt_idx + len(token)
            alignments.append(spacy_idx)
                
    return alignments

def load_senses(filename):
    senses = defaultdict(lambda :defaultdict(lambda :defaultdict(lambda: dict())))
    
    for line in open(filename):
        try:
            word, pos, example, level, vector = line.strip().split("\t")
            vector = np.array([float(value) for value in vector.split()])

            senses[word][pos][example]['vector'] = vector
            senses[word][pos][example]['level'] = level
        except Exception as e:
            print(e)
            print(line[:100])
            print(len(line.strip().split("\t")))

    
    return senses

senses = load_senses("api/bert_wsd_lv/output_lv.tsv")
print('sense OK!')

def wsd_bert(words, tgt_words_index, pos_tags, lemmatizer, bert):
    sentence = " ".join(words).lower()
    
    tgt_pos_tags = [pos_tags[idx] for idx in tgt_words_index]
#     print(tgt_pos_tags)
    lemmas = [lemmatizer(words[idx], pos_tag)[0] 
              for idx, pos_tag in zip(tgt_words_index, tgt_pos_tags)]
    
    model, tokenizer = bert
    tokenized_text = tokenizer.tokenize(" ".join(words))
    
    bertID2spacyID = alignment_bert_spacy_tokens(words, tokenized_text)
    spacyID2bertID = {}
    for idx, ID in enumerate(bertID2spacyID):
        if ID not in spacyID2bertID:
            spacyID2bertID[ID] = idx
    
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0 for _ in range(len(indexed_tokens) )]
    
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')
    model.to('cuda')
    
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    
    
    result = []
    
    for spacyID, lemma, pos_tag in zip(tgt_words_index, lemmas, tgt_pos_tags):
        bertID = spacyID2bertID[spacyID]
        
        embedding = []
        for i in range(-1, -5, -1):
            embedding.append(encoded_layers[i][0,bertID,])
        
        embedding = torch.cat(embedding).cpu().numpy()
        cosine_values = [ (sentence, sentence_info['level'], cosine_similarity([sentence_info['vector'] ], [embedding])[0][0])
                        for sentence, sentence_info in senses[lemma][pos_tag].items()]
        
        result.append(cosine_values)
    
    return result




def wsd_level(sentence, word):

    lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
    bert = (model, tokenizer)
    doc = nlp(sentence.lower())
    words = []
    pos_tags = []
    for token in doc:
        words.append(token.text)
        pos_tags.append(token.pos_)
    
    idx = [i for i, w in enumerate(words) if w == word]
    
    wsd_results = wsd_bert(words ,idx, pos_tags, lemmatizer=lemmatizer, bert=bert)
#     print(wsd_results)
    wsd_scores = [score for example_info in wsd_results for example, level, score in example_info]
    wsd_levels = [level for example_info in wsd_results for example, level, score in example_info]

#     print(wsd_scores.index(max(wsd_scores)))
    return wsd_levels[wsd_scores.index(max(wsd_scores))]