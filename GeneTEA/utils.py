import re
import string
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class GeneSymbolMapper():
    '''
    Use the same HGNC gene mapping as portal to detemine official gene symbol.  
    
    Use the following lookup pattern...
        For search_symbol, first check if in HGNC's unique symbol mapping.  
        Then check previous symbols.
        Then check aliases.  
        If multiple symbols found or no symbols found, return NaN.
    '''
    
    SUCCESS = 0
    MULTIPLE = 1
    NONE = 2
    
    def __init__(self, hgnc_file="taiga"):
        if hgnc_file == "taiga":
            hgnc_file = self._fetch_from_taiga()
        self.hgnc_mapping = pd.read_csv(hgnc_file, low_memory=False, sep="\t").set_index("symbol", drop=False)
        
        assert self.hgnc_mapping.index.is_unique, "HGNC mapping is no longer unique"
        self._convert_int_to_str()
        self._setup_search_dicts()
        self.test()
        self.available_columns = self.hgnc_mapping.columns

    def _fetch_from_taiga(self):
        return download_to_cache_ext("genetea-manuscript-bb10.18/hgnc_complete_set_2024-01-01", ext="txt")

    def _convert_int_to_str(self):
        float_cols = self.hgnc_mapping.loc[:,self.hgnc_mapping.dtypes == "float64"]
        self.hgnc_mapping.loc[:,self.hgnc_mapping.dtypes == "float64"] = float_cols.astype('Int64').astype('str').replace("<NA>", np.nan)    
    
    def _setup_search_dicts(self):
        self.symbols = set(self.hgnc_mapping.index)
        prev_symbols = self.hgnc_mapping["prev_symbol"].str.split("|").explode().dropna().reset_index()
        self.prev_symbols = dict(prev_symbols.groupby("prev_symbol")["symbol"].apply(list))
        alias_symbols = self.hgnc_mapping["alias_symbol"].str.split("|").explode().dropna().reset_index()
        self.alias_symbols = dict(alias_symbols.groupby("alias_symbol")["symbol"].apply(list))

    def test(self):
        assert self.map_gene("BRAF", return_error=True) == (self.SUCCESS, "BRAF")
        assert self.map_gene("SKIV2L", return_error=True) == (self.SUCCESS, "SKIC2")
        assert self.map_gene("MT1", return_error=True)[0] == self.MULTIPLE
        assert self.map_gene("MT1", return_error=True)[0] == self.MULTIPLE
        assert self.map_gene("doesn't exist", return_error=True)[0] == self.NONE
        assert self.map_gene_to("CAD", "entrez_id") == "790"

    def get_protein_coding(self):
        return self.hgnc_mapping.loc[lambda x: x["locus_group"] == "protein-coding gene", "symbol"].to_list()

    def map_gene(self, search_symbol, return_error=False):
        #check HGNC's symbols
        if search_symbol in self.symbols:
            if return_error: return self.SUCCESS, search_symbol
            return search_symbol
            
        #check previous symbols
        if search_symbol in self.prev_symbols:
            matching_prev_symbols = self.prev_symbols[search_symbol]
            if len(matching_prev_symbols) == 1:
                if return_error: return self.SUCCESS, matching_prev_symbols[0]
                return matching_prev_symbols[0]
            else:
                if return_error: 
                    message = f"'{search_symbol}' matches multiple gene symbols {matching_prev_symbols},"\
                                    +" disambiguate to include."
                    return self.MULTIPLE, message
                return np.nan
                
        #check aliases
        if search_symbol in self.alias_symbols:
            matching_aliases = self.alias_symbols[search_symbol]
            if len(matching_aliases) == 1:
                if return_error: return self.SUCCESS, matching_aliases[0]
                return matching_aliases[0]
            else:
                if return_error: 
                    message = f"'{search_symbol}' matches multiple gene symbols {matching_aliases},"\
                                    +" disambiguate to include."
                    return self.MULTIPLE, message
                return np.nan
        
        #nothing found  
        if return_error: 
            message = f"failed to find matching gene symbol for '{search_symbol}' in HGNC,"\
                            +" will be ignored."
            return self.NONE, message
        return np.nan
        
    def map_genes(self, search_symbols, **kws):
        map_lambda = lambda g: self.map_gene(g, **kws)
        return pd.Series(np.array(list(map(map_lambda, search_symbols)), dtype="object"), index=search_symbols)

    def map_gene_to(self, search_symbol, column, **kws):
        symbol = self.map_gene(search_symbol, **kws)
        if pd.isnull(symbol):
            return np.nan
        return self.hgnc_mapping.loc[symbol, column]
        
    def map_genes_to(self, search_symbols, column, **kws):
        map_lambda = lambda g: self.map_gene_to(g, column=column, **kws)
        return pd.Series(np.array(list(map(map_lambda, search_symbols)), dtype="object"), index=search_symbols)
    
    def map_genes_from(self, ids, column, to="symbol"):
        return pd.DataFrame(ids.values, index=ids.values).rename_axis(column).merge(
            self.hgnc_mapping, how="left", on=column
        ).set_index(column)[to]

def clean_word_tokenize(sentence, remove_punct=False):
    '''
    Tokenize and remove punkt while preserving tokens in all caps.
    '''
    clean_caps = lambda t: t if bool(re.match(".*[A-Z].*[A-Z].*", t)) else t.lower()
    tokens = [clean_caps(t) for t in re.findall(f'[{string.punctuation}]|\w+', sentence)]
    if remove_punct:
        return [t for t in tokens if t not in set(string.punctuation)]
    return tokens

def clean_tokenize(sentences):
    '''
    Tokenize and remove punkt while preserving tokens in all caps.
    '''
    return [clean_word_tokenize(sentence) for sentence in sentences]

def balanced_paranthesis(string):
    '''
    Check if parantheses are balanced
    '''
    parens_map ={'(':')','{':'}','[':']'}
    stack = []
    for paren in string:
        #if open add to stack
        if paren in parens_map:
            stack.append(paren)
        #if closed pop pair from stack if possible
        elif paren in parens_map.values():
            if (not stack) or (paren != parens_map[stack.pop()]):
                return False
    return not stack

def check_token_valid(token, stopwords=[], min_length=2, remove_no_alpha_chars=True, remove_start_end_punct=False):
    '''
    Check if a token is valid.
    '''
    if len(token) < min_length:
        return False
    if all([tok in stopwords for tok in token.split()]):
        return False
    if not balanced_paranthesis(token):
        return False
    if remove_no_alpha_chars and all([not t.isalpha() for t in token]):
        return False
    if remove_start_end_punct and (token[0] in string.punctuation or token[-1] in string.punctuation):
        return False
    return True

def subterm_id(x, y, bounded=True, sub_with=False):
    '''
    Determine if y is a subterm of x.
    '''
    #replace space/punct with option for 1 or more space/punct
    regex_term = re.sub(r"(\\ |\\-|\/)+", r"(\\s|\-|\/)+?", re.escape(x))
    #handle word bounds if desired
    if bounded:
        regex_term = r"\b"+regex_term+r"\b"
    compiled = re.compile(regex_term, flags=re.IGNORECASE)
    #substitute string
    if sub_with:
        return compiled.sub(sub_with, y)
    #or report match present
    return compiled.search(y) is not None
    
def chunk(list_like, chunk_size):
    '''
    Split a list-llike into chunks.
    '''
    length = len(list_like)
    return [
        list_like[i:min(i+1*chunk_size, length)] 
        for i in list(range(0, length, chunk_size))
    ]

def tril_stack(x):
    '''
    Get the lower triangle of a matrix as a series.
    '''
    triu_mask = np.tril(np.ones_like(x), k=-1)
    mean_x = pd.DataFrame(x).mask(triu_mask == 0).stack().dropna()
    return mean_x

def jaccard_index(n_matching_in_list, n_matching_overall, query_size):
    '''
    Compute jaccard index for two sets.
    '''
    return n_matching_in_list / ((n_matching_overall - n_matching_in_list) + query_size)

def read_gmt(name):
    '''
    Read a gmt file.
    '''
    genesets_dict = {}
    with open(name) as genesets:
        for line in genesets:
            try:
                entries = line.strip().split("\t")
                key = entries[0]
                if len(entries) < 3:
                    genesets_dict[key] = None
                else:
                    genesets_dict[key] = pd.Series({
                        "Name": entries[1],
                        "Genes":entries[2:]
                    })
            except:
                print(entries)
                raise Exception("failed")
    return pd.DataFrame(genesets_dict).T

def assess_obj_size(x, xname):
    '''
    Measure the sizes of python objects.
    '''
    import sys
    sizes = pd.Series()
    for var in dir(x):
        sizes[xname+"."+var] = sys.getsizeof(eval(xname+"."+var)) * 1e-9
    return sizes.sort_values().round(3)

def download_to_cache_ext(id, ext="raw", format="raw"):
    '''
    Helper that wraps internal CDS tool taigapy, essentially outputting a local path.
    '''
    from taigapy import default_tc3 as tc
    from taigapy.client_v3 import LocalFormat
    import uuid
    import os
    if format == "raw":
        format = LocalFormat.RAW
    elif format == "table":
        format = LocalFormat.CSV_TABLE
    elif format == "matrix":
        format = LocalFormat.CSV_MATRIX
    else:
        raise Exception(f"invalid taiga format {format}")   
    fn = tc.download_to_cache(id, format)
    new_name = f"/tmp/{str(uuid.uuid4())}.{ext}"
    os.symlink(fn, new_name)
    return new_name