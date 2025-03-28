from collections import Counter
from itertools import chain 
import pandas as pd
from sklearn.cluster import HDBSCAN

from .utils import chunk

class SapBERT(): 
    '''
    Compute semantic similarity of sentence embeddings with SapBERT.

    Based on code from https://github.com/idekerlab/llm_evaluation_for_gene_set_interpretation/blob/fa7c98705c68a14e5bb236c511f682fed1e58f99/semanticSimFunctions.py
    '''

    def __init__(self, llm="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(llm)
        self.model = AutoModel.from_pretrained(llm)
        self.model.eval()

    def get_embedding(self, text):
        import torch
        encoded_input = self.tokenizer(
            text, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embedding


class SynonymExtractor(object):
    '''
    Identify synonym sets for a set of tokens.
    '''
    def __init__(
        self, prefix="~ ", min_count=0,
        max_syn_len=50
    ):
        self.prefix = prefix
        self.min_count = min_count
        self.max_syn_len = max_syn_len

    def embed_with_bert(self, tokens):
        '''
        Use SapBERT to embed tokens.
        '''
        if self.min_count > 0:
            token_counts = Counter(chain.from_iterable(tokens))
            tokens = [x for x, count in token_counts.items() if count >= self.min_count]
        ss = SapBERT()
        token_emb = pd.concat([
            pd.DataFrame(ss.get_embedding(c).numpy(), index=c)
            for c in chunk(list(tokens), 1000)
        ])
        print("embedding shape", token_emb.shape)
        return token_emb

    def identify_synonyms(self, embedding, counts, metric="euclidean"):
        '''
        Use HDBSCAN to find sets of terms and label them.
        '''
        assert embedding.index.is_unique, "embedding index is not unique"
        #run HDBSCAN on embedding and get clusters
        self.hdbscan = HDBSCAN(min_samples=2, min_cluster_size=2, metric=metric, n_jobs=-1)
        cluster_labels = pd.Series(self.hdbscan.fit_predict(embedding), index=embedding.index)
        #throw out catch-all cluster -1 then make list of terms per cluster
        synonyms = cluster_labels.loc[cluster_labels >= 0].groupby(cluster_labels).apply(lambda x: list(x.index))
        #rename with most frequent term
        synonym_names = self.prefix+synonyms.apply(lambda x: counts[x].idxmax())
        synonyms.rename(synonym_names, inplace=True)
        #remove sets that are too long
        synonyms = synonyms.loc[synonyms.apply(len) < self.max_syn_len]
        return synonyms
    
    def run(self, tokens, counts):
        '''
        Run on tokens with counds.
        '''
        embedding = self.embed_with_bert(tokens)
        self.synonyms = self.identify_synonyms(embedding, counts)

        return embedding, self.synonyms