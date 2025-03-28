import pandas as pd     
import numpy as np

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk import everygrams

from collections import Counter

import re

from .utils import check_token_valid, clean_word_tokenize, GeneSymbolMapper, subterm_id, download_to_cache_ext
from .synonyms import SynonymExtractor

class UMLSPhraseMatcher():
    '''
    Get phrases matching concepts in the UMLS.
    '''
    def __init__(self, mrconso_path="taiga"):
        '''
        Intialize.

        Parameters:
            mrconso_path (str) - path to MRCONSO file from UMLS
        '''
        #use internal taiga path
        if mrconso_path == "taiga":
            mrconso_path = self._get_taiga_mrconso_path()
        #read in file
        mrconso = pd.read_csv(
            mrconso_path, sep='|', encoding='utf-8',
            names=[
                "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI",
                "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF", "filler"
            ],
            header=None, low_memory=False
        )
        #get terms in english that are part of multiple vocabs
        mrconso = mrconso.loc[lambda x: (x["LAT"] == "ENG")]
        multi_vocab = mrconso.groupby("CUI")['SAB'].nunique()
        filtered = mrconso.set_index("CUI").loc[multi_vocab.loc[lambda x: x > 1].index]
        #only keep phrases which have been cleaned
        self.vocab = set(
            filtered["STR"].dropna().apply(
                lambda x: " ".join(clean_word_tokenize(x, remove_punct=True))
            ).loc[
                lambda x: x.str.contains(" ").fillna(False) & x.apply(check_token_valid)
            ]
        )
        print("UMLS vocab size", len(self.vocab))

    def _get_taiga_mrconso_path(self):
        '''
        Fetch the internal taiga path.
        '''
        return download_to_cache_ext("genetea-manuscript-bb10.10/MRCONSO")

    def extract_matches(self, sentence):
        '''
        Split a tokenized sentence into everygrams
        '''
        tokens = clean_word_tokenize(sentence, remove_punct=True)
        eg = pd.Series([" ".join(eg) for eg in everygrams(tokens)])
        matches = list(set(eg) & self.vocab)
        return eg.loc[eg.isin(matches)].to_list()
    
    def run(self, sentences):
        '''
        Extract matches on a set of sentences.
        '''
        return [self.extract_matches(s) for s in sentences]


class DescriptionCorpus():
    '''
    Data structure for a corpus of descriptions.
    '''
    def __init__(
        self, descriptions, sources, entity_type="Gene",
        id_mapper="default",
        map_id_to= "default",
        stopwords="nltk", 
        phraser=None, 
        find_synonyms=True, blacklist_synonyms=True,
        synonyms=None,
        verbose=True,
        hgnc_table="taiga"
    ):
        self.ENTITY_TYPE = entity_type
        #add ID mapper 
        if id_mapper == "default":
            id_mapper = GeneSymbolMapper(hgnc_table).map_genes
        self.id_mapper = id_mapper
        if map_id_to == "default":
            map_id_to = GeneSymbolMapper(hgnc_table).map_gene_to
        self.map_id_to = map_id_to
        #sources contain citations
        if verbose: print("Checking sources...")
        self.sources = self._check_sources(sources)
        #description has a column for a set of source
        if verbose: print("Checking descriptions...")
        self.descriptions = self._check_descriptions(descriptions)
        #setup stopwords and phrase creater
        self.stopwords = nltk_stopwords.words("english") if stopwords == "nltk" else stopwords
        self.phraser = phraser
        #parse into sentences and terms
        if verbose: print("Parsing all descriptions...")
        self.sentences, self.terms = self.parse_all_descriptions()
        #look for coincident or synonymous terms if desired
        if verbose: print("Blacklisting subterms and removing coincident terms...")
        self.get_coincident_and_blacklist_subterms()
        if verbose and find_synonyms: print("Getting synonyms terms...")
        # self.include_synonymous_phrases = include_synonymous_phrases
        self.embedding, self.synonyms = None, synonyms
        if find_synonyms:
            self.embedding, self.synonyms = self.get_synonyms(blacklist_synonyms=blacklist_synonyms)
            
    def _check_sources(self, sources):
        '''
        Validate sources input.
        '''
        assert isinstance(sources, pd.DataFrame), "sources must be a pandas dataframe"
        expected_cols = ["Name", "Link", "Description"]
        assert all([c in sources.columns for c in expected_cols]), f"sources must have {expected_cols}"
        return sources

    def _check_descriptions(self, descriptions):
        '''
        Validate descriptions input.
        '''
        assert descriptions.shape[0] > 0, "no rows in descriptions"
        assert descriptions.shape[0] > 0, "no columns in descriptions"
        mismatches = set(descriptions.columns) - set(self.sources["Name"])
        assert len(mismatches) == 0, f"sources' Names must match descriptions columns, the following mismatch: {list(mismatches)}"
        if self.id_mapper is not None:
            descriptions = self._remap_descriptions(descriptions)
        descriptions.index.name = self.ENTITY_TYPE
        descriptions.columns.name = "Source"
        return descriptions
    
    def _remap_descriptions(self, descriptions):
        id_mapping = self.id_mapper(descriptions.index)
        clean = {}
        for id, des in descriptions.groupby(id_mapping):
            if des.shape[0] == 1:
                clean[id] = des.iloc[0]
            else:
                clean[id] = des.apply(lambda x: ' '.join(x.dropna()))
        clean = pd.concat(clean, axis=1).T
        return clean

    def get_descriptions(self, ids=None, sources=None):
        '''
        Get descriptions for a set of sources.
        '''
        if ids is None:
            ids = self.descriptions.index
        if sources is None:
            sources = self.descriptions.columns
        return self.descriptions.loc[ids, sources]
    
    def get_source_href(self, source, entity=None):
        '''
        Return the source as a link.
        '''
        link = self.sources.set_index("Name").loc[source, "Link"]
        page = self.sources.set_index("Name").loc[source, f"{self.ENTITY_TYPE}Page"]
        if page is not None and entity is not None:
            column_name = re.findall("\[(\w+)\]", page)[0]
            if self.map_id_to is not None:
                entity = self.map_id_to(entity, column_name)
            link = page.replace(f"[{column_name}]", str(entity))
        if link is None:
            return source
        return "<a href=\""+link+f"\">{source}</a>"

    def get_combined_description_as_text(self, ids, sep=" | ", html=False):
        '''
        Get descriptions as a single text with source links per id.
        '''
        texts = self.sentences.loc[self.sentences[self.ENTITY_TYPE].isin(ids)].groupby(self.ENTITY_TYPE)
        combined_des = {}
        for id, t in texts:
            t = t.groupby("Source")["Text"].apply(lambda x: " ".join(x.dropna()))
            if html:
                t = t.index.to_series().apply(lambda x: self.get_source_href(x, entity=id))+": "+t.values
            else:
                t = t.index.to_series()+": "+t.values
            combined_des[id] = sep.join(t.dropna())
        return pd.Series(combined_des)

    def parse_into_sentences(self, id):
        '''
        Read through all the sources and tokenize sentences for a id, then tokens and phrases.
        '''
        sentences = []
        for source, text in self.descriptions.loc[id].items():
            if pd.isnull(text): continue
            curr_sentences = sent_tokenize(text)
            for i, sent in enumerate(curr_sentences):
                tokens = clean_word_tokenize(sent)
                sentences.append({
                    self.ENTITY_TYPE:id,
                    "Source":source,
                    "SentenceNum":i,
                    "Text":sent,
                    "Tokens":[t for t in tokens if check_token_valid(t, self.stopwords)],
                })
        return pd.DataFrame(sentences)

    def _create_fingerprint(self, terms):
        '''
        Produce a fingerprint based on sentences and appearances.
        '''
        return (
            terms["Sentence Index"].apply(lambda x: " ".join([str(i) for i in x]))
            +" #"+terms["Total #"].astype(str)
        )

    def map_terms_to_sents(self, sentences):
        '''
        Map tokens and phrases to source sentences, creating a term table with fingerprints.
        '''
        tokens = sentences.explode("Tokens").groupby("Tokens").apply(
            lambda x: pd.Series({"Sentence Index":list(x.index), "Total #":x.shape[0]})
        )
        tokens["Phrase"] = False
        phrases = sentences.explode("Phrases").groupby("Phrases").apply(
            lambda x: pd.Series({"Sentence Index":list(x.index), "Total #":x.shape[0]})
        )
        phrases["Phrase"] = True
        terms = pd.concat([tokens, phrases])
        terms["Sentence Fingerprint"] = self._create_fingerprint(terms)
        terms["Blacklisted"] = False
        terms.index.name = "Term"
        return terms
    
    def parse_all_descriptions(self):
        '''
        Parse all descriptions into sentences then terms.
        '''
        sentences = pd.concat([
            self.parse_into_sentences(id)
            for id in self.descriptions.index
        ]).reset_index(drop=True)
        if self.phraser is not None:
            sentences["Phrases"] = self.phraser.run(sentences["Text"])
        else:
            sentences["Phrases"] = np.nan
        terms = self.map_terms_to_sents(sentences)
        sentences.drop(columns=["Tokens", "Phrases"], inplace=True)
        return sentences, terms 

    def find_subterms(self, terms):
        '''
        Identify terms which are a subset of tokens within a larger term.
        '''
        return pd.Series({
            term:[t for t in terms if t != term and subterm_id(term, t)]
            for term in terms
        })
    
    def get_coincident_and_blacklist_subterms(self):
        '''
        Use fingerprints to find terms that only ever appear in the same sentences.
        Any terms which are subterms of a different member of a coincident set are blacklisted to reduce redundancy.
        '''
        blacklisted = []
        for f, coincident_for_f in self.terms.reset_index().groupby("Sentence Fingerprint"):
            if coincident_for_f.shape[0] > 1:
                subterms = self.find_subterms(coincident_for_f["Term"].tolist()).explode()
                blacklisted.extend(subterms.index[subterms.notnull()].unique().tolist())
        self.terms.loc[blacklisted, "Blacklisted"] = True
        self.terms.index.name = "Term"
    

    def identify_synonyms(self, exclude_sources=['Chromosome Arm', 'Gene Location']):
        '''
        Identify synonyms with a synonym extractor.
        '''
        self.synonym_extractor = SynonymExtractor(min_count=0)
        #identify terms only found in certain sources that should be excluded (ie. chromosome arm)
        if exclude_sources is not None:
            exclude_from_syns = [
                term for term in self.terms.index
                if self.sentences.loc[
                    self.terms.loc[term, 'Sentence Index'], 'Source'
                ].isin(exclude_sources).all()
            ]
        else:
            exclude_from_syns = []
        #filter tokens by count, blacklist, and exclude terms only in certain sources
        filtered_terms = self.terms.loc[
            (self.terms['Total #'] >= 2) & (~self.terms.index.isin(exclude_from_syns))
        ]
        tokens = filtered_terms.index
        counts = filtered_terms["Total #"].astype(int)
        #run the extractor
        embedding, synonyms = self.synonym_extractor.run(tokens, counts=counts)
        return embedding, synonyms
    
    def get_synonyms(self, blacklist_synonyms=False):
        '''
        Get synonyms identified with synonym extractor.
        '''
        if self.synonyms is None:
            embedding, synonyms = self.identify_synonyms()
        else:
            embedding, synonyms = self.embedding, self.synonyms
        synonyms_exploded = synonyms.explode().loc[lambda x: x.isin(self.terms.index)]
        syn_map = pd.Series(synonyms_exploded.index, synonyms_exploded.values)
        syn_terms = synonyms_exploded.groupby(synonyms_exploded.index).apply(list).to_frame("Term Set")
        syn_terms["Sentence Index"] = self.terms.groupby(syn_map)["Sentence Index"].sum().apply(lambda x: sorted(x))
        syn_terms["Total #"] = self.terms.groupby(syn_map)["Total #"].sum()
        syn_terms["Sentence Fingerprint"] = self._create_fingerprint(syn_terms)
        syn_terms["Phrase"] = False
        syn_terms["Blacklisted"] = syn_terms["Term Set"].apply(lambda x: self.terms.loc[x, "Blacklisted"].all())
        self.terms = pd.concat([self.terms, syn_terms]).convert_dtypes()
        self.terms["Synonym"] = np.where(self.terms.index.isin(synonyms.index), True, False)
        self.terms.index.name = "Term"
        if blacklist_synonyms:
            self.terms.loc[synonyms_exploded.values, "Blacklisted"] = True
        return embedding, synonyms
    
    def preprocess(self, sources=None):
        '''
        Return a dataframe mapping non-blacklisted entities to a list of tokens.
        '''
        sources = self.sources['Name'] if sources is None else sources
        valid_terms = self.terms.loc[
            (self.terms['Total #'] > 1) & (~self.terms['Blacklisted']), 'Sentence Index'
        ].explode()
        valid_terms = valid_terms.reset_index().merge(
            self.sentences.loc[self.sentences['Source'].isin(sources), [self.ENTITY_TYPE]],
            left_on="Sentence Index", right_index=True, how='left'
        )
        term_to_entity = valid_terms.groupby(self.ENTITY_TYPE)["Term"].apply(list)
        return term_to_entity.to_frame('Terms')
    
    def nominate_stopwords(self, seed):
        '''
        Given a seed list of stopwords, use term embedding to find any highly correlated terms
        with the average of the seed list.
        '''
        mean_stopwords = self.embedding.loc[self.embedding.index.isin(seed)].mean(axis=0)
        corr_w_stopwords = self.embedding.T.corrwith(mean_stopwords).sort_values()
        return corr_w_stopwords.index[corr_w_stopwords > 0.65].to_list()
    
    def save_embedding(self, path):
        if self.embedding is not None:
            self.embedding.to_csv(path)

    def del_embedding(self):
        del self.embedding