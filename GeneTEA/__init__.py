import pandas as pd
import numpy as np
from itertools import cycle, chain
import textwrap
import dill
import sys
import difflib

import networkx as nx
from nltk import everygrams

import scipy.stats
import scipy.spatial
import scipy.cluster

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import f_regression, r_regression
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from multiprocessing import Pool
from functools import partial as functools_partial

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .utils import subterm_id, clean_word_tokenize
from .preprocessing import DescriptionCorpus

class xTEA(object):
    '''
    Creates TFIDF embedding of a set of [entity] summaries, identifying stopwords and
    allowing for search for frequent terms for a set of [entities].
    '''
    
    def __init__(self, corpus, sources=None, entity_type="Entity", custom_stopwords=None, vec_kwargs=dict(max_df=0.5, min_df=3, binary=False)):
        '''
        Initialize xTEA.

        Parameters:
            corpus (DescriptionCorpus) - a description corpus
            sources (None or list of str) - optional list of sources to restrict corpus to
            entity_type (str) - name of entity (ie. "Gene" or "Drug")
            custom_stopwords (list) - list of custom stopwords
            vec_kwargs (dict) - parameters for sklearn TfidfVectorizer
        '''
        self.ENTITY_TYPE = entity_type
        assert isinstance(corpus, DescriptionCorpus), "must provide a DescriptionCorpus as corpus"
        self.corpus = corpus
        self.sources = sources
        self.synonyms = self.corpus.synonyms
        self.vectorizer = TfidfVectorizer(analyzer=lambda x: x, **vec_kwargs)
        self.custom_stopwords = custom_stopwords
    
    def fit(self):
        '''
        Fit an sklearn TFIDF vectorizer to a set of summaries.
        '''
        #preprocess corpus
        self.preprocessed = self.corpus.preprocess(sources=self.sources) 
        #fit the vectorizer
        self.tfidfs = self.vectorizer.fit_transform(self.preprocessed["Terms"])
        #get indices and IDFs
        self.entities = self.preprocessed.index
        self.terms = pd.Index(self.vectorizer.get_feature_names_out())
        self.idfs = pd.Series(self.vectorizer.idf_, index=self.terms)
        self.n_matching_overall = self.get_n_entities_matching_term(None)
        #pick a min number of entities to filter on by default
        if self.idfs.skew() > 0:
            #if right skewed, do not filter
            self.max_n = 0
        else:
            #otherwise treat terms appearing in many entities as stopwords
            self.max_n = np.ceil(0.15 * self.entities.shape[0]).astype(int)
        #get stopwords
        self.stopwords = self.get_stopwords(return_reason=False)
        #create mapping of terms to synonyms
        if self.synonyms is not None:
            self.term_to_synonym = self.map_term_set(self.synonyms)
        syns_exploded = self.synonyms.explode() if self.synonyms is not None else pd.Series()
        self.all_syns = pd.Series(syns_exploded.index, syns_exploded.values)
        #get mapping of terms to subterms
        self.subterms = self.terms.to_series().apply(self.get_subterms)
        #run latent semantic analysis
        self.run_lsa(n_components=500)
    
    def run_lsa(self, entities=None, n_components=500):
        '''
        Run latent semantic analysis (LSA) with sklearn.

        Parameters:
            entities (None, str, list of str) - list of entities
            n_components (int) - dimensionality of LSA matrix
        '''
        tfidfs, _, entities = self.get_subset_tfidfs(entities=entities)
        self.lsa = make_pipeline(TruncatedSVD(n_components=n_components, algorithm="arpack"), Normalizer(copy=False))
        self.lsa_emb = pd.DataFrame(self.lsa.fit_transform(tfidfs), index=entities)
        
    def get_stopwords(
        self, max_n="default", remove_stopword_ngrams=None, 
        remove_stopword_synonyms=True, special_cases="chr(?:\d+|X|Y|MT)|(?:\d+|X|Y|MT)(?:p|q)(?:\d+)?(?:.\d+)?", 
        keep_all_caps=True, return_reason=False
    ):
        '''
        Get stopwords used by model. 

        Parameters:
            max_n (int or "default") - max size of term
            remove_stopword_ngrams (None or str) - rules to filter n-grams, either None, "all", "any", or "start/end"
            remove_stopword_synonyms (boolean) - whether to filter terms synonymous with other stopwords
            special_cases (str) - regex pattern for terms to exclude from stopword list
            keep_all_caps (boolean) - whether to keep terms that are all capital letters
            return_reason (boolean) - whether to return a verbose reason for inclusion in stopword list
        Return:
            list of stopwords (or pandas Series mapping term to reason included if return_reason is True)
        '''
        stopwords = []
        #add custom stopwords
        if self.custom_stopwords is not None:
            stopwords.append(pd.Series("custom stopword", index=set(self.custom_stopwords)))
        #remove ngrams containing stopwords
        if remove_stopword_ngrams is not None:
            #get ngram tokens
            ngram_tokens = self.terms[self.terms.str.contains(" ")]
            #find subtokens for all ngrams
            ngram_subtokens = pd.DataFrame({
                "nGram":ngram_tokens, 
                "Subtokens":ngram_tokens.str.split(" "),
            }).explode("Subtokens")
            #check if in stopwords
            temp_stopwords = set(pd.concat(stopwords).index)
            ngram_subtokens["Stopword"] = ngram_subtokens["Subtokens"].isin(temp_stopwords)
            #remove based on provided criteria
            if "all" in remove_stopword_ngrams:
                ngram_only_stopwords = ngram_subtokens.groupby("nGram")["Stopword"].all().loc[lambda x: x].index
                stopwords.append(pd.Series(
                    f"ngram made up of all stopwords", 
                    index=set(ngram_only_stopwords)
                ))

            if "any" in remove_stopword_ngrams:
                ngram_any_stopwords = ngram_subtokens.groupby("nGram")["Stopword"].any().loc[lambda x: x].index
                stopwords.append(pd.Series(
                    f"ngram with a stopword", 
                    index=set(ngram_any_stopwords)
                ))

            if "start/end" in remove_stopword_ngrams:
                ngram_start_stopwords = ngram_subtokens.groupby("nGram")["Stopword"].first().loc[lambda x: x].index
                ngram_end_stopwords = ngram_subtokens.groupby("nGram")["Stopword"].last().loc[lambda x: x].index
                stopwords.append(pd.Series(
                    f"ngram starts/ends with stopword", 
                    index=set(ngram_start_stopwords) | set(ngram_end_stopwords)
                ))

        #get members of synonym sets that are stopwords
        stopwords_from_synonyms = self.synonyms[self.synonyms.index.isin(pd.concat(stopwords).index)].explode()
        stopwords.append(pd.Series("in stopword synonym set", index=set(stopwords_from_synonyms)))
        #remove terms which are too frequent
        if max_n is None:
            pass
        elif max_n == "default":
            many_entities_stopwords = self.n_matching_overall.index[self.n_matching_overall > self.max_n].sort_values().to_list()
            stopwords.append(pd.Series(f"in many entities (>{self.max_n})", index=set(many_entities_stopwords)))
        else:
            many_entities_stopwords = self.n_matching_overall.index[self.n_matching_overall > max_n].sort_values().to_list()
            stopwords.append(pd.Series(f"in many entities (> {max_n})", index=set(many_entities_stopwords)))
        #remove synonyms that are made up mainly of stopwords
        if remove_stopword_synonyms and self.synonyms is not None:
            temp_stopwords = set(pd.concat(stopwords).index)
            frac_stopwords = self.synonyms.apply(lambda x: len(set(x) & set(temp_stopwords)) / len(x))
            stopword_synonyms = frac_stopwords.index[frac_stopwords >= 0.50]
            stopwords.append(pd.Series(f"synonym set made up of >50% stopwords", index=set(s for s in stopword_synonyms if s in self.terms)))
        #combine all stopwords into series and concatenate reasons
        if return_reason:
            stopwords = pd.concat(stopwords)
            #don't get rid terms that match certain patterns
            if special_cases is not None:
                stopwords = stopwords.loc[
                    ~stopwords.index.to_series().str.match(special_cases)
                ]
            if keep_all_caps:
                stopwords = stopwords.loc[~stopwords.index.to_series().str.isupper()]
            return stopwords.groupby(stopwords.index).apply(list).apply(";".join)
        else:
            stopwords_set = set()
            for s in stopwords:
                stopwords_set |= set(s.index)
            if special_cases is not None:
                keep = self.terms[self.terms.str.match(special_cases)]
                stopwords_set -= set(keep)
            if keep_all_caps:
                keep = self.terms[self.terms.str.isupper()]
                stopwords_set -= set(keep)
        return sorted(list(stopwords_set & set(self.terms)))
    
    def validate_entities(self, entities, verbose=False, background=None):
        '''
        Validate a list of entities.

        Parameters:
            entities (None, str, list of str) - entity or list of entities
            verbose (boolean) - whether to print info
            background (None or list of str) - entities to use as background
        Returns:
            valid (list), invalid (list), and remapped (dict of original to mapped) entities
        '''
        #handle None
        if entities is None:
            return None, None, None
        #if single entity convert to list
        if isinstance(entities, str):
            entities = [entities]
        #apply id mapping if present
        if self.corpus.id_mapper is not None:
            remapped = self.corpus.id_mapper(entities)
        else:
            remapped = pd.Series(list(entities), index=list(entities)).astype(str)
            remapped.index = remapped.index.astype(str)
        #handle background
        if background is None:
            background = self.entities
        #find valid, invalid, and remapped entities
        valid = sorted(list(set(remapped.dropna().values) & set(pd.Series(background).astype(str))))
        invalid = sorted(remapped.index[remapped.isnull()].to_list())
        remapped = remapped[remapped.index != remapped.values].dropna().to_dict()
        #error if no valid entities
        if len(valid) == 0:
            raise RuntimeError("no valid entities")
        #print info
        if verbose:
            if len(invalid) > 0:
                print(f"Invalid query entities: {invalid}")
            if len(remapped) > 0:
                print(f"Remapped query entities: {remapped}")
        return valid, invalid, remapped
    
    def validate_terms(self, terms):
        '''
        Validate a list of terms.

        Parameters:
            terms (None, str, list of str) - term or list of terms
        Returns:
            valid (list), invalid (list)
        '''
        #handle None
        if terms is None:
            return None, None
        #if single term convert to list
        if isinstance(terms, str):
            terms = [terms]
        #find valid, invalid, and remapped terms
        valid = sorted(list(set(terms) & set(self.terms)))
        invalid = sorted(list(set(terms) - set(self.terms)))
        #print info
        if len(valid) == 0:
            if len(invalid) > 0:
                most_sim = " Most similar terms: "+"; ".join([self.get_most_similar_terms(i, n=1).values[0] for i in invalid])
            else:
                most_sim = ""
            raise RuntimeError("no valid terms."+most_sim)
        return valid, invalid
    
    def get_subset_tfidfs(self, terms=None, entities=None):
        '''
        Get a subset of the tf-idf matrix.

        Parameters:
            terms (None, str, list of str) - list of terms
            entities (None, str, list of str) - list of entities
        Returns:
            tfidf csr sparse matrix, list of terms (columns), list of entities (rows)
        '''
        #filter input list of terms
        if terms is not None:
            terms, invalid_terms = self.validate_terms(terms) 
            if len(invalid_terms) > 0:
                print(f"Invalid query terms: {invalid_terms}")

            #find indices of terms and subset matrix
            terms_indices = pd.Series(range(self.terms.shape[0]), index=self.terms)
            subset = self.tfidfs[:, terms_indices[terms].values]
        # or use all terms
        else:
            terms = self.terms.values
            subset = self.tfidfs
        #repeat with entities provided
        if entities is not None:
            entities, _, _ = self.validate_entities(entities, verbose=True) 
            entities_indices = pd.Series(range(self.entities.shape[0]), index=self.entities)
            subset = subset[entities_indices[entities].values, :]
        #or use all entities
        else:
            entities = self.entities.values
        return subset, terms, entities
    
    def get_tfidf_series(self, term, entities=None):
        '''
        Get tfidf values as a pandas series.

        Parameters:
            term (str) - a single term
            entities (None, str, list of str) - list of entities
        Returns:
            a Series of tfidf values
        '''
        tfidfs, terms, entities = self.get_subset_tfidfs(terms=term, entities=entities)
        return pd.DataFrame(tfidfs.todense(), index=entities, columns=terms)[term]

    def get_entities_matching_term(self, terms, entities=None):
        '''
        Get the entities matching a term.

        Parameters:
            terms (None, str, list of str) - list of terms
            entities (None, str, list of str) - list of entities
        Returns:
            a Series with terms as index and space separated matching entities as values
        '''
        subset, terms, entities = self.get_subset_tfidfs(terms=terms, entities=entities)
        #get the indices of nonzero values
        entities_idx, terms_idx = subset.nonzero()
        #compile the series of terms-entities
        entities_for_terms = pd.Series(
            pd.Series(entities)[entities_idx].values
        ).groupby(
            pd.Series(terms)[terms_idx].values
        ).apply(
            lambda x: " ".join(sorted(x))
        )
        return entities_for_terms
    
    def get_terms_matching_entities(self, entities, terms=None):
        '''
        Get the terms matching an entity.

        Parameters:
            entities (None, str, list of str) - list of entities
            terms (None, str, list of str) - list of terms
        Returns:
            a Series with entites as index and ;-separated matching entities as values
        '''
        subset, terms, entities = self.get_subset_tfidfs(terms=terms, entities=entities)
        #get the indices of nonzero values
        entities_idx, terms_idx = subset.nonzero()
        #compile the series of terms-entities
        terms_for_entities = pd.Series(
            pd.Series(terms)[terms_idx].values
        ).groupby(
            pd.Series(entities)[entities_idx].values
        ).apply(
            lambda x: ";".join(sorted(x))
        )
        return terms_for_entities

    def get_n_entities_matching_term(self, terms, entities=None):
        '''
        Get the number of entities matching a term.

        Parameters:
            terms (None, str, list of str) - list of terms
            entities (None, str, list of str) - list of entities
        Returns:
            a Series with terms as index and number of matching entities as values
        '''
        subset, terms, entities = self.get_subset_tfidfs(terms=terms, entities=entities)
        #get the indices of nonzero values
        entities_idx, terms_idx = subset.nonzero()
        #compile the series of terms-counts
        entities_count = pd.Series(
            entities_idx
        ).groupby(
            pd.Series(terms)[terms_idx].values
        ).count().astype(int)
        return entities_count

    def get_terms_for_synonym(self, synonym):
        '''
        Get the terms in a synonym set.

        Parameters:
            synonym (str) - synonym set name
        Returns:
            list of synonymous terms or None
        '''
        if synonym in self.synonyms:
            return self.synonyms.loc[synonym].copy()
        return None
    
    def get_hypergeom_sig(self, entities, method='bh', background=None, exclude_from_fdr=[]):
        '''
        Compute a FDR corrected p-value with a hypergeometric test.

        Parameters:
            entities (None, str, list of str) - list of entities
            method (str) - method for false discovery control (scipy.stats.false_discovery_control parameter)
            background (None or list of str) - entities to use as background
            exclude_from_fdr (list of str) - terms to exclude from FDR calculation
        Returns:
            pandas DataFrame indexed by terms with "n Matching ENTITIES Overall", "n Matching ENTITIES in List", "p-val" and "FDR"
        '''
        #select background
        if background is None:
            n_overall = self.n_matching_overall
            background = self.entities
        else:
            background, _, _ = self.validate_entities(background)
            n_overall = self.get_n_entities_matching_term(None, entities=background)
        #check entities
        entities, _, _ = self.validate_entities(entities, verbose=True, background=background)
        #get n matches overall and n matches in query for all terms (fill NAs with 0)
        pval_inputs = pd.DataFrame({
            f"n Matching {self.ENTITY_TYPE}s Overall":n_overall, 
            f"n Matching {self.ENTITY_TYPE}s in List":self.get_n_entities_matching_term(None, entities),
        }).fillna(0).rename_axis("Term")
        #only compute p-vals for unique inputs
        unique_pval_inputs = pval_inputs.drop_duplicates()
        pvals_by_inputs = pd.concat([
            unique_pval_inputs.reset_index(drop=True),
            pd.Series(
                scipy.stats.hypergeom.sf(
                    unique_pval_inputs[f"n Matching {self.ENTITY_TYPE}s in List"], 
                    len(background), 
                    unique_pval_inputs[f"n Matching {self.ENTITY_TYPE}s Overall"], 
                    len(entities), 
                    loc=1
                )
            ).rename("p-val")
        ], axis=1)
        #map pvals inputs to terms and run FDR correction
        result = pval_inputs.reset_index().merge(pvals_by_inputs, on=pval_inputs.columns.to_list(), how="left").set_index("Term")
        pvals = result.loc[~result.index.isin(exclude_from_fdr), "p-val"]
        fdr = pd.Series(scipy.stats.false_discovery_control(pvals, method=method), index=pvals.index)
        result["FDR"] = fdr
        return result
    
    def get_total_info(self, terms=None, entities=None):
        '''
        Get the total info (sum of tfidfs) for each term over a set of entities.

        Parameters:
            terms (None, str, list of str) - list of terms
            entities (None, str, list of str) - list of entities
        Returns:
            pandas Series with sum of tfidfs along term axis
        '''
        tf, t, _ = self.get_subset_tfidfs(terms=terms, entities=entities)
        return pd.DataFrame(tf.sum(axis=0), columns=t).iloc[0]
    
    def get_multiterm_matches(self, terms, entities=None):
        '''
        Get list of entities matching multiple terms.

        Parameters:
            terms (None, str, list of str) - list of terms
            entities (None, str, list of str) - list of entities
        Returns:
            list of entities
        '''
        tfidfs, terms, genes = self.get_subset_tfidfs(terms=terms, entities=entities)
        #must have non-zero in all terms
        entities = pd.DataFrame((tfidfs > 0).sum(axis=1), index=genes)[0]
        return entities.loc[lambda x: x == len(terms)].index.to_list()

    def get_multiterm_pval(self, terms, entities, background=None):
        '''
        Get hypergeometric p-value for entities matching multiple terms.

        Parameters:
            terms (None, str, list of str) - list of terms
            entities (None, str, list of str) - list of entities
            background (None or list of str) - entities to use as background
        Returns:
            p-value
        '''
        #pick background
        if background is None:
            background = self.entities
        else:
            background, _, _ = self.validate_entities(background)
        #get all multi-term matches and cleaned query
        matches = self.get_multiterm_matches(terms, background)
        entities, _, _ = self.validate_entities(entities)
        return scipy.stats.hypergeom.sf(
            len(set(matches) & set(entities)), 
            len(background), 
            len(matches), 
            len(entities), 
            loc=1
        )
    
    def get_context(self, term, entities, html=False, highlight_with=('<mark>','</mark>')):
        '''
        Get the context sentences for a term in list of entities.

        Parameters:
            terms (str) - single term
            entities (None, str, list of str) - list of entities
            html (boolean) - whether to include html source link and highlights
            highlight_with (None or tuple of str) - html tags to highlight with
        Returns:
            pandas Series of entities to context sentences for term
        '''
        #restrict to single term
        assert isinstance(term, str), "term must be a string"
        terms, _ = self.validate_terms(term)
        if len(terms) != 1:
            print(f"Invalid term: {term}")
        term = terms[0]
        #validate entitites
        entities, _, _ = self.validate_entities(entities, verbose=True) 
        #find context sentences
        excerpts = pd.Series(dtype=str)
        for g in entities:
            #get sentences containing token
            entity_excerpts = self.corpus.sentences.loc[self.corpus.terms.loc[term, "Sentence Index"]].loc[
                lambda x: x[self.ENTITY_TYPE] == g, ["Source", "Text"]
            ]
            #concatenate sentences
            if len(entity_excerpts) > 0:
                entity_excerpts = entity_excerpts.groupby("Source")["Text"].apply(lambda x: "...".join(x))
                #highlight if desired
                if html:
                    entity_excerpts = entity_excerpts.index.to_series().apply(
                        lambda x: self.corpus.get_source_href(x, entity=g)
                    )+": "+entity_excerpts.values
                    #find terms to highlight in excerpts
                    if self.check_if_synonym(term):
                        terms_to_highlight = self.synonyms[term]
                    else: 
                        terms_to_highlight = [term]
                    for t in sorted(terms_to_highlight, key=len):
                        entity_excerpts = entity_excerpts.apply(
                            lambda x: subterm_id(t, x, sub_with=fr"{highlight_with[0]}\g<0>{highlight_with[1]}")
                        )
                else:
                    entity_excerpts = entity_excerpts.index.to_series()+": "+entity_excerpts.values 
                excerpts[g] = " | ".join(entity_excerpts)
        #add index name
        excerpts.index.name = self.ENTITY_TYPE
        if excerpts.shape[0] == 0:
            raise Exception("Something went wrong getting excerpts :(")
        return excerpts
    
    def get_text(self, entities, html=False):
        '''
        Get the full raw text for a set of entities.

        Parameters:
            entities (None, str, list of str) - list of entities
        Return:
            pandas Series of texts
        '''
        entities, _, _ = self.validate_entities(entities, verbose=True)
        #get the text
        contexts = self.corpus.get_combined_description_as_text(entities, html=html)
        contexts.index.name = self.ENTITY_TYPE
        contexts.name = "Text"
        return contexts
    
    def gen_cosine_similarity(self, subset=None, entities=None, terms=None, axis="terms"):
        '''
        Get cosine sim over tf-idf subset.

        Parameters:
            subset (None or scipy sparse matrix) - subset of tfidfs
            terms (None, str, list of str) - list of terms
            entities (None, str, list of str) - list of entities
            axis (str) - whether to compute for "terms" or ENTITIES
        Returns:
            pandas DataFrame of pairwise cosine sim
        '''
        if subset is None:
            subset, terms, entities = self.get_subset_tfidfs(terms=terms, entities=entities)

        if axis == "terms":
            subset = subset.transpose()
            labels = terms
            name = "Term"
        elif axis == self.ENTITY_TYPE:
            labels = entities
            name = self.ENTITY_TYPE
        else:
            raise RuntimeError("invalid axis")
        
        #compute the cosine similarity
        cosine = pd.DataFrame(cosine_similarity(subset, subset), index=labels, columns=labels)
        cosine.index.name = name + " Index"
        cosine.columns.name = name + " Columns"
        return cosine
    
    def gen_clusters(self, terms, requested, term_groups=None):
        '''
        Heirarchically cluster the terms, term groups and entities based on the cosine similarity.

        Parameters:
            terms (None, str, list of str) - list of terms
            requested (None, str, list of str) - list of requested entities
            term_groups (None or pandas Series) - mapping of term to term group
        Returns:
            clusterings with Cluster number and Order
        '''
        #get subset
        subset, terms, entities =  self.get_subset_tfidfs(entities=requested, terms=terms)

        #find expected number of clusters
        n_clusters = round(len(terms) * len(entities) /  subset.count_nonzero())

        #get cosine sim
        term_cosine = self.gen_cosine_similarity(subset=subset, terms=terms, axis="terms")

        #heirarchically cluster
        def cluster(df, index_name):
            if df.shape[0] < 3:
                return pd.DataFrame({
                    index_name:df.index, 
                    "Cluster":1, 
                    "Order":range(df.shape[0])
                })
            dist = scipy.spatial.distance.pdist(df)
            linkage = scipy.cluster.hierarchy.linkage(dist)
            cluster_df = pd.DataFrame(index=df.index)
            cluster_df["Cluster"] = scipy.cluster.hierarchy.fcluster(linkage, n_clusters, criterion="maxclust")
            ordered_linkage = scipy.cluster.hierarchy.optimal_leaf_ordering(linkage, dist)
            ordered_leaves = scipy.cluster.hierarchy.leaves_list(ordered_linkage)
            cluster_df['Order'] = pd.Series(range(len(ordered_leaves)), index=df.index[ordered_leaves])
            cluster_df.index.name = index_name
            return cluster_df.reset_index()
        
        #find clusters for each of terms, term groups and entities
        term_cluster = cluster(term_cosine, "Term")
        term_group_cluster = cluster(term_cosine.groupby(term_groups).mean(), "Term Group") if term_groups is not None else None
        if requested is not None:
            entity_cosine = self.gen_cosine_similarity(subset=subset, entities=entities, axis=self.ENTITY_TYPE)
            entities_cluster = cluster(entity_cosine, self.ENTITY_TYPE) 
        else:
            entities_cluster = None
        return term_cluster, entities_cluster, term_group_cluster
    
    def check_if_synonym(self, term):
        '''
        Check if a term is a synonym.

        Parameters:
            term (str) - a single term
        Returns:
            boolean indicating if synonym
        '''
        if self.synonyms is None:
            return False
        return term in self.synonyms.index
    
    def map_term_set(self, term_set):
        '''
        Map terms in a term set.

        Parameters:
            term_set (pandas Series) - index is set names and values are list of terms
        Returns:
            pandas Series mapping of terms to term sets and vice versa
        '''
        #combine term set mapping with mapping of terms to set names
        term_to_set = pd.concat([
            term_set, 
            pd.Series(term_set.explode().index, index=term_set.explode().values).apply(lambda x: [x])
        ])
        return pd.Series({
            t:";".join(term_to_set[t]) if t in term_to_set.index 
            else None for t in self.terms
        })

    def map_synonyms(self, terms):
        '''
        Map set of terms to synonyms.

        Parameters:
            terms (None, str, list of str) - list of terms
        Returns:
            pandas Series mapping terms to synonyms and vice versa or None
        '''
        terms, _ = self.validate_terms(terms) 
        if self.synonyms is None:
            return None
        return self.term_to_synonym.loc[terms]
        
    def get_frequent_terms(self, entities, background=None, plot=False, max_fdr=None, **plot_kws):
        '''
        Find minimally filtered frequent terms for a query of entities.

        Parameters:
            entities (list of str) - query of entities
            background (list or None) - list of entities to use as background or None for all entities in corpus
            plot (boolean) - whether to produce interactive plotly visual
        Returns:
            pandas DataFrame with Term, p-value, FDR, etc. or None if no terms are found
        '''
        #get FDR, stopword annotation, etc.
        frequent_terms = self.get_hypergeom_sig(entities, background=background).sort_values(["FDR", "p-val"])
        #if max FDR provided, filter in order to reduce overhead
        if max_fdr is not None:
            frequent_terms = frequent_terms.loc[frequent_terms["FDR"] < max_fdr]
            if frequent_terms.shape[0] == 0:
                return None
        #get other info
        frequent_terms["Stopword"] = frequent_terms.index.isin(self.stopwords)
        frequent_terms[f"Matching {self.ENTITY_TYPE}s in List"] = self.get_entities_matching_term(frequent_terms.index, entities=entities)
        frequent_terms["Synonyms"] = self.map_synonyms(frequent_terms.index)
        frequent_terms["Total Info"] = self.get_total_info(frequent_terms.index, entities=background)
        frequent_terms["Effect Size"] = self.get_total_info(frequent_terms.index, entities=entities)
        frequent_terms.sort_values("FDR", ascending=True, inplace=True)
        frequent_terms.reset_index(inplace=True)
        #plot if desired
        if plot:
            fig = self.plot_volc(frequent_terms, **plot_kws)
            fig.show()
        return frequent_terms.dropna(subset=[f"Matching {self.ENTITY_TYPE}s in List"])
    
    def get_subterms(self, term):
        '''
        Get list of subterms for a term.

        Parameters:
            term (str) - single term
        Returns:
            list of subterms
        '''
        #if synonym, apply to all synonyms
        if self.check_if_synonym(term):
            return self.synonyms[term]+list(chain.from_iterable([self.get_subterms(t) for t in self.synonyms[term]]))
        #tokenize and look for subterms
        split_term = clean_word_tokenize(term, remove_punct=False)
        subterms = [" ".join(x) for x in everygrams(split_term, max_len=len(split_term)-1)]
        subterms.extend(self.all_syns.reindex(subterms).dropna().values)
        return subterms
    
    def get_most_similar_terms(self, query_string, n=10):
        '''
        Get n most similar terms to a query.

        Parameters:
            query (str) - string to search for
            n (int) - number of terms to return
        Returns:
            pandas Series mapping most similar terms to self/synonym
        '''
        #get all possible terms
        all_terms = pd.concat([
            self.synonyms.explode(), 
            self.terms[~self.terms.isin(self.synonyms.index)].to_series()
        ])
        all_terms = pd.Series(all_terms.index, all_terms.values)
        #look for close matches
        most_sim = difflib.get_close_matches(query_string, all_terms.index, n=n)
        return all_terms.loc[most_sim]
    
    def rank_enriched_terms(self, enriched, continuous):
        '''
        Rank a set of enriched terms.

        Parameters:
            enriched (pandas DataFrame) - enriched term table
            continuous (boolean) - whether query is continuous
        Returns:
            a pandas Series with terms as index and ranks as values
        '''
        enriched_w_extras = enriched.copy()
        if not continuous:
            enriched_w_extras.sort_values(
                [
                    "Effect Size",
                    "FDR",
                    "Total Info",
                    "Term"
                ], ascending=[
                    False,
                    True,
                    False,
                    True
                ], inplace=True
            )
        else:
            enriched_w_extras["Abs Correlation"] = enriched_w_extras["Correlation"].abs()
            enriched_w_extras.sort_values(
                [
                    "Effect Size",
                    "Weighted Query Info",
                    "Abs Correlation",
                    "FDR",
                    "Total Info",
                    "Term"
                ], ascending=[
                    False,
                    False,
                    True,
                    False,
                    False,
                    True
                ], inplace=True
            )
        return pd.Series(range(len(enriched_w_extras)), enriched_w_extras["Term"])

    def graph_filter(
        self, enriched, continuous=False, sort_by=["FDR", "Effect Size"], ascending=[True, False], verbose=False
    ):
        '''
        Use graph-based strategy to filter and produce term groupings.

        Parameters:
            enriched (pandas DataFrame) - enriched term table
            continuous (boolean) - whether query is continuous
            sort_by (list of str) - columns to sort on
            ascending (list of booleans) - how to sort columns
            verbose - whether to print info
        Returns:
            a filtered and sorted version of enriched table
        '''
        if not continuous:
            entities = enriched[f'Matching {self.ENTITY_TYPE}s in List'].str.split().explode().unique()
            #construct a bipartite graph, where terms are connected to entities
            tfidf, terms, entities = self.get_subset_tfidfs(enriched['Term'], entities)
            B = nx.bipartite.from_biadjacency_matrix(tfidf)
            if verbose: nx.draw(B)

            #get the projected graph for terms, relabel them
            bottom_node_to_term = pd.Series(terms, range(len(entities), len(entities)+len(terms)))
            P_full = nx.bipartite.overlap_weighted_projected_graph(B, bottom_node_to_term.index, jaccard=False)
            nx.relabel_nodes(P_full, bottom_node_to_term.to_dict(), copy=False)

            #copy and prune the graph's edges to only connect subterms
            P = P_full.copy()
            S = nx.from_dict_of_lists(self.subterms.loc[terms].to_dict())
            P.remove_edges_from(
                (e for e, w in nx.get_edge_attributes(P,'weight').items() if w < 1 or not S.has_edge(*e))
            )
            if verbose: 
                nx.draw(P, with_labels=True)
                plt.show()
                print(list(nx.connected_components(P)))
        else:
            S = nx.from_dict_of_lists(self.subterms.loc[enriched["Term"]].to_dict())
            P = S.subgraph(enriched["Term"])
            entities = None

        #only keep nodes with highest ranking out of their direct neighbors
        ranking = self.rank_enriched_terms(enriched, continuous)
        enriched = enriched.set_index("Term")
        if verbose: print("ranking", ranking)
        to_keep = [
            n for n in P.nodes if (ranking.loc[list(P.neighbors(n))] >= ranking.loc[n]).all()
        ]
        if verbose: print("keep", to_keep)
        enriched_filtered = enriched.loc[to_keep]

        #create new graph where edges are cosine similarity in query versus across all genes
        if entities is None:
            C = nx.from_pandas_adjacency(self.gen_cosine_similarity(terms=to_keep))
        else:
            C = nx.from_pandas_adjacency(
                self.gen_cosine_similarity(terms=to_keep) *
                self.gen_cosine_similarity(terms=to_keep, entities=entities) 
            )
        #and from it, prune then identify communities
        C.remove_edges_from((e for e, w in nx.get_edge_attributes(C,'weight').items() if w < 0.15 or e[0] == e[1]))
        if verbose: 
            nx.draw(C, with_labels=True, pos=nx.spring_layout(C))
            plt.show()
        try:
            communities = nx.community.greedy_modularity_communities(C, weight="weight", resolution=1)
        #if error is raised, resort to connected components
        except:
            communities = nx.connected_components(C)

        #label term group with highest info term, and add as column
        groups, to_drop = {}, []
        for g in communities:
            g = list(g)
            if verbose: print("group", g)
            #if single term group, continue
            if len(g) == 1: 
                groups[g[0]] = g
            else:
                groups["|".join(ranking[g].nsmallest(3).index)+"++"] = g
        
        #if only one group, uncollapse it
        if verbose: print("groups", groups)
        if len(groups) == 1:
            groups = {t:t for t in groups[list(groups.keys())[0]]}

        #map term groups to terms
        groups = pd.Series(groups).explode()
        enriched_filtered["Term Group"] = pd.Series(groups.index, groups.values)
        enriched_filtered.drop(to_drop, inplace=True)

        #sort and return
        return enriched_filtered.reset_index().sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    
    def _sort_and_get_top_n(self, table, sort_by, n, group_subterms, filter=False, **kws):
        '''
        Internal sort and filter method.

        Parameters:
            table (pandas DataFrame) - enriched term table
            sort_by (str) - order to return terms in, either "Effect Size" or "Significance"
            n (int) - max number of terms to return
            group_subterms (boolean) - whether to group and filter subterms
            filter (boolean) - whether to apply graph filtering
        Returns:
            a filtered and sorted version of enriched table
        '''
        #determine sort order
        if sort_by == "Significance":
            sort_order, ascending = ["FDR", "Effect Size"], [True, False]
        elif sort_by == "Effect Size":
            sort_order, ascending = ["Effect Size", "FDR"], [False, True]
        else:
            raise RuntimeError("unrecognized sort_by")

        #filter and select top n if desired 
        if filter:
            table = self.graph_filter(table, sort_by=sort_order, ascending=ascending, **kws)

        if n is not None:
            #get top n groups rather then terms
            if group_subterms:
                top_n_groups = table["Term Group"].drop_duplicates()[:n]
                table = table.loc[table["Term Group"].isin(top_n_groups)].reset_index(drop=True)
            else:
                table = table[:n]
        return table.copy()

    def get_enriched_terms(
        self, entities=None, frequent_terms=None, 
        max_fdr=0.05, n_range=[None, "default"], min_n_list=2, n=10, 
        sort_by="Effect Size", plot=False, group_subterms=True, 
        background=None, exclude_terms=[], **plot_kws
    ):
        '''
        Get a table of enriched terms in a query of entites.

        Parameters:
            entities (list of str) - query of entities (can be None if frequent_terms is provided)
            frequent_terms (pandas DataFrame) - output from get_frequent_terms
            max_fdr (float) - FDR threshold
            n_range (list of int, None, or "default") - list of allowed number of entities [min, max] in returned terms (can be None, max can be "default")
            min_n_list (int) - minimum number of entiteis in query
            n (int) - max number of terms to return
            sort_by (str) - order to return terms in, either "Effect Size" or "Significance"
            plot (boolean) - whether to produce interactive plotly visual
            group_subterms (boolean) - whether to group and filter subterms
            background (list or None) - list of entities to use as background or None for all entities in corpus
            exclude_terms (list) - list of terms to drop from table
        Returns:
            pandas DataFrame with Term, p-value, FDR, etc. or None if no enriched terms are found

        '''
        #get terms
        if entities is not None:
            assert frequent_terms is None, "cannot provide entities and frequent_terms"
            frequent_terms = self.get_frequent_terms(entities, max_fdr=max_fdr, background=background)
        else:
            assert frequent_terms is not None, "if entities not provided, must provide frequent_terms"

        #return if no frequent terms
        if frequent_terms is None:
            return None

        #set thresholds
        min_n = 0 if n_range[0] is None else n_range[0]
        if n_range[1] is None:
            max_n = frequent_terms["IDF"].min()
        elif n_range[1] == "default":
            max_n = self.max_n
        else:
            max_n = n_range[1]

        #filter based on thresholds
        enriched = frequent_terms.loc[
            (~frequent_terms["Stopword"]) &
            (frequent_terms["FDR"] < max_fdr) & 
            (frequent_terms[f"n Matching {self.ENTITY_TYPE}s Overall"] >= min_n) &
            (frequent_terms[f"n Matching {self.ENTITY_TYPE}s Overall"] < max_n) &
            (frequent_terms[f"n Matching {self.ENTITY_TYPE}s in List"] >= min_n_list)
        ].copy()

        if len(exclude_terms) > 0:
            enriched = enriched.loc[~enriched["Term"].isin(exclude_terms)]

        if enriched.shape[0] == 0:
            return None
        
        enriched = self._sort_and_get_top_n(
            enriched, sort_by, n, group_subterms, filter=True
        )
        
        #plot if desired
        if plot:
            fig = self.plot_enriched(
                enriched, sort_by=sort_by, n=n, group_subterms=group_subterms, **plot_kws
            )
            fig.show()

        return enriched
    
    def _prep_for_plot(self, to_plot, groupby, split_on, no_term_to_entity=False):
        '''
        Create columns to aid with plotting.

        Paramters:
            to_plot (pandas DataFrame) - table to prep
            groupby (str) - column to group on
            split_on (str) - separator
            no_term_to_entity - whether to map terms to entities
        Returns:
            to_plot with more columns and optionally mapping of terms to entities
        '''
        if to_plot.shape[0] == 0:
            raise RuntimeError("nothing to plot")
        
        #add -log10 FDR values
        to_plot["-log10 FDR"] = -np.log10(to_plot["FDR"]+sys.float_info.min)

        #clip long text
        to_clip = {
            groupby:dict(width=25, break_long_words=True, max_lines=3, placeholder="..." ), 
            "Synonyms":dict(width=50, break_long_words=True, max_lines=3, placeholder="..." ), 
            f"Matching {self.ENTITY_TYPE}s in List":dict(width=50, break_long_words=True, max_lines=3, placeholder="..." )
        }
        for col, kws in to_clip.items():
            if col not in to_plot.columns:
                continue
            to_plot["Clipped "+col] = pd.Series(to_plot[col].fillna("").apply(
                lambda x: "<br>".join(textwrap.wrap(x, **kws))
            ).values, index=to_plot.index)

        if no_term_to_entity:
            return to_plot

        #map terms to entities using groupby column
        term_to_entity = pd.get_dummies(to_plot.set_index(groupby)[f"Matching {self.ENTITY_TYPE}s in List"].str.split(split_on).explode())
        term_to_entity.columns.name = self.ENTITY_TYPE
        term_to_entity = term_to_entity.groupby(term_to_entity.index).sum().stack().to_frame("Count").reset_index()
        term_to_entity["n Terms"] = term_to_entity[groupby].replace(to_plot.groupby(groupby)[f"Matching {self.ENTITY_TYPE}s in List"].count())
        term_to_entity["Fraction"] = (term_to_entity["Count"] / term_to_entity["n Terms"])

        return to_plot, term_to_entity
    
    def _customdata(self, to_plot, entities=None, continuous=False):
        '''
        Helper to format customdata for plotly.
        '''
        if not continuous:
            custom = to_plot[[
                "Term", "Clipped Synonyms", "FDR", "Effect Size", f"n Matching {self.ENTITY_TYPE}s Overall", 
                f"Clipped Matching {self.ENTITY_TYPE}s in List", f"n Matching {self.ENTITY_TYPE}s in List"
            ]].fillna('')
            custom[self.ENTITY_TYPE] = entities
        else:
            custom = to_plot[[
                "Term", "Clipped Synonyms", "FDR", "Effect Size", f"n Matching {self.ENTITY_TYPE}s Overall",
                "Weighted Query Info", "Correlation"
            ]].fillna('')
            custom[self.ENTITY_TYPE] = entities
        return custom

    def _hovertemplate(self, continuous=False):
        if not continuous:
            hover = "<b>%{customdata[0]}</b><br>%{customdata[1]}<br><br>" +\
                    "FDR: %{customdata[2]:.4e}<br>"+\
                    "Effect Size: %{customdata[3]:.4f}<br>" +\
                    "n Matching "+self.ENTITY_TYPE+"s Overall: %{customdata[4]}<br>" +\
                    "Matching:<br>%{customdata[5]} (n=%{customdata[6]})<br>" +\
                    "<extra></extra>"
        else:
            hover = "<b>%{customdata[0]}</b><br>%{customdata[1]}<br><br>" +\
                    "FDR: %{customdata[2]:.4e}<br>"+\
                    "Effect Size: %{customdata[3]:.4e}<br>" +\
                    "n Matching"+self.ENTITY_TYPE+"s Overall: %{customdata[4]}<br>" +\
                    "Weighted Query Info: %{customdata[5]:.4e}<br>" +\
                    "Correlation: %{customdata[6]:.4f}<br>" +\
                    "<extra></extra>"
        return hover

    def pyplot_enriched(
        self, to_plot, axs=None, group_subterms=True, show_entities=True, cluster=False, heatmap=False, return_fig=False,
        figsize=(3.5,2), dpi=300, width_ratios=[4,2], bbox_to_anchor=(-0.5, -0.5, 1, 1), rotate_x=False, color="#79c99e",
        to_color={}, clip_len=40, entities=None
    ):
        '''
        Plot a static visualization of enriched terms.
        '''
        if axs is None:
            fig, axs = plt.subplots(1, 2, width_ratios=width_ratios, figsize=figsize, layout="constrained", dpi=dpi)
            
        groupby = "Term Group" if group_subterms else "Term"

        if show_entities:
            to_plot, term_to_entity = self._prep_for_plot(to_plot, groupby, " ")
            to_plot["Clipped Term"] = to_plot["Term"].apply(lambda x: x[:clip_len]+"..." if len(x) > clip_len  else x)
            if cluster:
                term_cluster, entity_cluster, term_group_cluster = self.gen_clusters(
                    to_plot["Term"], term_to_entity[self.ENTITY_TYPE] if entities is None else entities, 
                    term_groups=to_plot.set_index("Term")["Term Group"] if group_subterms else None
                )
                clusters = term_cluster.set_index("Term")["Order"] if groupby == "Term" else term_group_cluster.set_index("Term Group")["Order"]
                entity_clusters = entity_cluster.set_index(self.ENTITY_TYPE)["Order"]
            else:
                grouped = to_plot[groupby].drop_duplicates()
                clusters = pd.Series(range(len(grouped), 0, -1), index=grouped)
                unique_ents = term_to_entity[self.ENTITY_TYPE].drop_duplicates() if entities is None else entities
                entity_clusters = pd.Series(range(len(unique_ents)), index=unique_ents)
            term_to_entity["Term Order"] = term_to_entity[groupby].replace(clusters)
            term_to_entity["Entity Order"] = term_to_entity[self.ENTITY_TYPE].replace(entity_clusters)
            bar_order = to_plot.set_index(groupby).loc[clusters.index, "Clipped Term"].loc[lambda x: ~x.index.duplicated()]
            
            few_entities = entity_clusters.shape[0] < 20
            if few_entities and not heatmap:
                sns.scatterplot(
                    data=term_to_entity.loc[lambda x: x["Count"] != 0], x="Entity Order", y="Term Order", size=0, ax=axs[0], 
                    s=20, color="#4D5359"
                )
                axs[0].grid(alpha=0.2, zorder=-1)
                axs[0].set_xticks(entity_clusters.values, rotation=0)
                axs[0].set_xticklabels(entity_clusters.index, rotation=0)
                axs[0].set_yticks(clusters.values, rotation=0)
                axs[0].set_yticklabels(bar_order.values, rotation=0)
                axs[0].legend().remove()
            else:
                cax = inset_axes(
                    axs[0], width="20%", height="5%", 
                    loc='lower left', borderpad=0,
                    bbox_to_anchor=bbox_to_anchor,
                    bbox_transform=axs[0].transAxes,
                )
                sns.heatmap(
                    data=term_to_entity.set_index(["Term Order", "Entity Order"])["Fraction"].unstack(), 
                    xticklabels=entity_cluster[self.ENTITY_TYPE] if few_entities else False,
                    cbar=True, cmap="Greens", linecolor="k", linewidths=1 if few_entities else 0,
                    cbar_kws=dict(label="Fraction\nMatching", orientation="horizontal"),
                    cbar_ax=cax, ax=axs[0], vmin=0, vmax=1
                )
                axs[0].set_xticks(entity_clusters.values, rotation=0)
                axs[0].set_xticklabels(entity_clusters.index, rotation=0)
                axs[0].set_yticks(clusters.values, rotation=0)
                axs[0].set_yticklabels(bar_order.values, rotation=0)
                cax.xaxis.set_ticks_position('top')
                for _, spine in axs[0].spines.items():
                    spine.set_visible(True)
            if rotate_x:
                axs[0].set_xticklabels(labels=axs[0].get_xticklabels(), rotation=90, ha="center")
        else:
            to_plot = to_plot.reset_index()
            to_plot["Clipped Term"] = to_plot["Term"].apply(lambda x: x[:clip_len]+"..." if len(x) > clip_len  else x)
            to_plot["-log10 FDR"] = -np.log10(to_plot["FDR"]+sys.float_info.min)
            bar_order = to_plot.set_index(groupby)["Clipped Term"]
            to_plot.rename(columns={"n Matching Genes in List":"# Genes"}, inplace=True)
            sns.heatmap(
                data=to_plot.groupby(groupby)[["# Genes"]].mean(), 
                yticklabels=bar_order,
                cbar=False, cmap="Greens", linecolor="k", linewidths=1,
                ax=axs[0], annot=True
            )
        axs[0].set_ylabel(None)
        axs[0].set_xlabel(None)
        sns.barplot(data=to_plot, y=groupby, x="-log10 FDR", ax=axs[1], color=color, order=bar_order.index)
        axs[1].set_xlabel(r"-log$_{10}$(FDR)")
        axs[1].set_ylabel(None)
        for pattern, c  in to_color.items():
            idxs = np.arange(len(bar_order))[bar_order.index.str.contains(pattern, case=False)]
            for i in idxs:
                axs[0].get_yticklabels()[i].set_color(c)
        axs[1].set_yticks([])
        axs[1].set_yticklabels([])
        if return_fig: return fig
    
    def plot_enriched(
        self, to_plot, filter=False, sort_by="Effect Size", n=None, group_subterms=True,
        title="Top TEA Terms", cluster_terms=True, 
        entity_order=None, split_on=" ", show_terms=None, **kws
    ):
        '''
        Plot an interactive upset style plot of enriched terms.
        '''
        #check that table is not none
        if to_plot is None:
            raise RuntimeError("No enriched terms")

        #sort and trim
        if show_terms is not None:
            to_plot = to_plot.set_index("Term").loc[show_terms].reset_index()
            if len(to_plot) == 0:
                raise RuntimeError("Could not find terms listed in show_terms.")
        else:
            to_plot = self._sort_and_get_top_n(
                to_plot, sort_by, n, group_subterms, filter=filter, 
                **kws
            )

        #add columns for plotting and get term to entity mapping
        groupby = "Term Group" if group_subterms else "Term"
        to_plot, term_to_entity = self._prep_for_plot(to_plot, groupby, split_on)
        #generate clusters
        term_cluster, entity_cluster, term_group_cluster = self.gen_clusters(
            to_plot["Term"], term_to_entity[self.ENTITY_TYPE], term_groups=to_plot.set_index("Term")["Term Group"]
        )
        to_plot = to_plot.merge(
            term_group_cluster if group_subterms else term_cluster, how="left", on=groupby
        )
        
        #create figure
        height = min(800, 350+(entity_cluster.shape[0]*25))
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            row_heights=[250,height-250],
            vertical_spacing=0
        )
        fig.append_trace(
            go.Bar(
                y=to_plot["-log10 FDR"],
                x=to_plot[groupby], 
                marker=dict(color="#0aa18f"),
                orientation='v',
                customdata=self._customdata(to_plot),
                hovertemplate=self._hovertemplate(),
            ), row=1, col=1
        )
        if entity_cluster.shape[0] > 40:
            fig.append_trace(
                go.Heatmap(
                    z=term_to_entity["Fraction"],
                    x=term_to_entity[groupby],
                    y=term_to_entity[self.ENTITY_TYPE],
                    colorscale="dense",
                    colorbar=dict(title='Fraction<br>Matching', len=0.25),
                    customdata=term_to_entity[[self.ENTITY_TYPE, groupby, "Count", "n Terms"]].fillna('N/A'),
                    hovertemplate=self.ENTITY_TYPE+": %{customdata[0]}<br>" +\
                                    "Term: %{customdata[1]}<br>" +\
                                    "(matches %{customdata[2]}/%{customdata[3]} subterms)" +\
                                    "<extra></extra>",
                ), row=2, col=1
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            fig.update_layout(
                yaxis2_title=f"{self.ENTITY_TYPE}s (n={entity_cluster.shape[0]})",
                yaxis2_showticklabels=False
            )
        else:
            term_to_entity = term_to_entity[term_to_entity["Count"] > 0]
            fig.append_trace(
                go.Scatter(
                    x=term_to_entity[groupby],
                    y=term_to_entity[self.ENTITY_TYPE],
                    textposition="bottom right",
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        size=term_to_entity["Fraction"].clip(lower=0.5)*15,
                        opacity=1,
                        color="#2b2c2c"
                    ),
                    customdata=term_to_entity[[self.ENTITY_TYPE, groupby, "Count", "n Terms"]].fillna('N/A'),
                    hovertemplate=self.ENTITY_TYPE+": %{customdata[0]}<br>" +\
                                    "Term: %{customdata[1]}<br>" +\
                                    "(matches %{customdata[2]}/%{customdata[3]} subterms)" +\
                                    "<extra></extra>",
                ), row=2, col=1
            )
        fig.update_layout(
            barmode='overlay',
            autosize=True,
            height=height,
            showlegend=False,
            xaxis_type='category',
            yaxis_title="-log10(FDR)",
            yaxis2_type='category',
            yaxis2_dtick=1, 
            title=title
        )

        #determine term and entity order
        if cluster_terms:
            term_order = to_plot.sort_values("Order").groupby(groupby).first().sort_values("Order")["Clipped "+groupby]
        else:
            term_order = to_plot.sort_values("FDR").groupby(groupby).first().sort_values("FDR")["Clipped "+groupby]
        fig.update_xaxes(
            categoryorder='array', 
            categoryarray=term_order.index,
            tickmode='array',
            tickvals=term_order.index,
            ticktext=term_order.values
        )

        if entity_order is None:
            entity_order = entity_cluster.sort_values('Order')[self.ENTITY_TYPE]
        else:
            entity_order = entity_order.loc[entity_order.isin(entity_cluster[self.ENTITY_TYPE])].iloc[::-1]
        fig.update_layout(
            yaxis2_categoryorder='array', 
            yaxis2_categoryarray=entity_order,
            yaxis2_range=[-1, entity_cluster[self.ENTITY_TYPE].nunique()],
        )
        return fig
    
    def plot_subterms(
        self, enriched, term_group, continuous=False, entity=None, color="#5F4690", 
        title="Sub-terms", split_on=" ", **kwargs
    ):
        '''
        Plot interactive visualization for sub-terms.
        '''
        #check that table is not none
        if enriched is None:
            raise RuntimeError("No enriched terms")
        
        #sort and trim
        to_plot = enriched.loc[enriched["Term Group"] == term_group]
        to_plot = self._sort_and_get_top_n(
            to_plot, "Effect Size", None, False
        )

        #if no subterms, return None
        if to_plot.shape[0] == 1:
            return None
        
        if not continuous:
            to_plot, term_to_entity = self._prep_for_plot(to_plot, "Term", split_on)

            #if only one entity selected, bar or waterfall
            if entity is not None:
                #get relevant terms
                to_plot = to_plot.loc[
                    to_plot["Term"].isin(term_to_entity.loc[(term_to_entity[self.ENTITY_TYPE] == entity) & (term_to_entity["Count"] > 0), "Term"])
                ]
                if to_plot.shape[0] == 0:
                    raise RuntimeError(f"No subterm terms for {entity}")
                #if few enough, plot top terms as bar
                elif to_plot.shape[0] < 25:
                    fig = go.Figure([
                        go.Bar(
                            y=to_plot["-log10 FDR"],
                            x=to_plot["Term"],
                            marker=dict(
                                color=color
                            ),
                            customdata=self._customdata(to_plot, entities=entity),
                            hovertemplate=self._hovertemplate(),
                        )])
                    fig.update_layout(
                        autosize=True,
                        height=350,
                        showlegend=False,
                        xaxis_type='category',
                        yaxis_title="-log10(FDR)",
                        title=title
                    )
                    fig.update_xaxes(
                        categoryorder='array', 
                        categoryarray=to_plot["Term"].values,
                        tickmode='array',
                        tickvals=to_plot["Term"].values,
                        ticktext=to_plot["Clipped Term"].values
                    )
                    return fig
                #otherwise do waterfall
                else:
                    fig = go.Figure([
                        go.Scattergl(
                            y=to_plot["-log10 FDR"],
                            x=to_plot["-log10 FDR"].rank(ascending=False),
                            mode="markers",
                            marker=dict(
                                color=color
                            ),
                            customdata=self._customdata(to_plot, entity=entity),
                            hovertemplate=self._hovertemplate(),
                        )
                    ])
                    fig.update_layout(
                        autosize=True,
                        height=350,
                        showlegend=False,
                        xaxis_title="Rank",
                        yaxis_title="-log10(FDR)",
                        title=title
                    )
                    return fig

        #if all entities, either do normal upset plot or do scatter of FDR vs Effect Size
        if to_plot.shape[0] < 25:
            enriched_plot = self.plot_enriched if not continuous else self.plot_enriched_continuous
            return enriched_plot(
                to_plot, group_subterms=False, n=None, cluster_terms=True,
                title=title, **kwargs
            )
        else:
            return self.plot_volc(to_plot, title=title, continuous=continuous)

    def plot_volc(self, to_plot, title="All Matching TEA Terms", continuous=False, enriched=None, enrich_kws=None, selected_terms=None):
        '''
        Plot an interactive volcano plot.
        '''
        #check that table is not none
        if to_plot is None:
            raise RuntimeError("No frequent terms")
        
        #handle no enriched table being provided
        if enriched is None:
            enriched = pd.DataFrame(columns=["Term"])

        #trim entity list and terms
        to_plot = self._prep_for_plot(to_plot, "Term", " ", no_term_to_entity=True)

        if not continuous:
            to_plot = to_plot.loc[
                (to_plot[f"n Matching {self.ENTITY_TYPE}s in List"] >= 2)
            ].copy()
            effect_size = "Effect Size"
        else:
            effect_size = "Correlation"
        

        if selected_terms is None:
            selected_terms = []
        selected_terms = to_plot.loc[to_plot["Term"].isin(selected_terms)]
        enriched_terms = to_plot.loc[
            to_plot["Term"].isin(enriched["Term"]) & ~to_plot["Term"].isin(selected_terms['Term'])
        ]
        other_terms = to_plot.loc[
            ~to_plot["Term"].isin(enriched["Term"]) & ~to_plot["Term"].isin(selected_terms['Term'])
        ]

        fig = go.Figure([
            go.Scattergl(
                x=other_terms[effect_size], 
                y=other_terms["-log10 FDR"],
                text=other_terms["Term"],
                name=f"Other Terms (n={other_terms['Term'].nunique()})",
                mode="markers",
                marker=dict(
                    color="#7e7e7e",
                    line=dict(width=2, color="#7e7e7e")
                ),
                customdata=self._customdata(other_terms, continuous=continuous),
                hovertemplate=self._hovertemplate(continuous=continuous),
            ),
            go.Scattergl(
                x=enriched_terms[effect_size], 
                y=enriched_terms["-log10 FDR"],
                text=enriched_terms["Term"],
                name=f"Enriched Terms (n={enriched_terms['Term'].nunique()})",
                mode="markers",
                marker=dict(
                    color="#03C04A",
                    line=dict(width=2, color="#03C04A")
                ),
                customdata=self._customdata(enriched_terms, continuous=continuous),
                hovertemplate=self._hovertemplate(continuous=continuous),
            ),
            go.Scattergl(
                x=selected_terms[effect_size], 
                y=selected_terms["-log10 FDR"],
                text=selected_terms["Term"],
                name=f"Selected Terms (n={selected_terms['Term'].nunique()})",
                mode="markers+text",
                textposition="top center",
                marker=dict(
                    color="#FF0000",
                    line=dict(width=2, color="#FF0000")
                ),
                customdata=self._customdata(selected_terms, continuous=continuous),
                hovertemplate=self._hovertemplate(continuous=continuous),
            )
        ])
        if enrich_kws is not None:
            if "max_fdr" in enrich_kws.keys():
                if enrich_kws["max_fdr"] is not None:
                    fig.add_hline(
                        y=-np.log10(enrich_kws["max_fdr"]), 
                        line_dash="dash", line_color="black"
                    )
        fig.update_layout(
            title=title,
            xaxis_title=effect_size,
            yaxis_title="-log10(FDR)"
        )
        if continuous:
            fig.update_layout(
                xaxis_showexponent='all',
                xaxis_exponentformat='e',
            )
        return fig
    
    def plot_frequent(self, *kws):
        '''
        Plot frequent terms as an interactive volcano.
        '''
        return self.plot_volc(continuous=False, *kws)
    
    def align_series(self, x, verbose=True, handle_duplicates=False):
        '''
        Check and align IDs the series provided by user.
        '''
        assert isinstance(x, pd.Series), "x must be a pandas Series"
        assert pd.api.types.is_float_dtype(x), "x must be a float dtype"
        #remap the input IDs, report if necessary
        remapped = self.corpus.id_mapper(x.index) if self.corpus.id_mapper is not None else x.index
        x_clean = x.copy()
        x_clean.index = remapped.values
        if verbose:
            invalid = sorted(remapped.index[remapped.isnull()].to_list())
            remapped = remapped[remapped.index != remapped.values].dropna().to_dict()
            if len(invalid) > 0:
                print(f"Invalid query entities: {invalid}")
            if len(remapped) > 0:
                print(f"Remapped query entities: {remapped}")
        #drop unmapped and non-present entities
        x_clean = x_clean.loc[lambda z: z.index.notnull() & z.index.isin(self.entities)]
        #check for duplications
        if x_clean.index.duplicated().any():
            if handle_duplicates == "drop":
                x_clean = x_clean.loc[~x_clean.index.duplicated(keep=False)]
            elif handle_duplicates == "mean":
                x_clean = x_clean.groupby(x_clean.index).mean()
            elif handle_duplicates == False:
                print(
                    "WARNING: duplicates were found after aligning IDs, will need to be corrected prior to correlations.",
                    "The handle_duplicates parameter can be used to address this."
                )
            else:
                raise Exception("invalid handle_duplicates value")
        return x_clean

    def _permutation_pool(self, tfidfs, x_clean, terms, rng, n_permutations=10, n_jobs=None, chunksize=100):
        '''
        Helper for permutation.
        '''
        observed = pd.Series(r_regression(tfidfs, x_clean), index=terms).rename("Observed")
        
        with Pool(n_jobs) as p:
            result = list(p.imap(
                functools_partial(_shuffled_correlation, tfidfs=tfidfs, x_clean=x_clean, observed=observed, rng=rng),
                np.arange(n_permutations), chunksize=chunksize
            ))
            
        return observed, result
    
    def get_correlated_terms(
            self, x, verbose=False, handle_duplicates=False, 
            method="f-test", n_permutations=10000, n_jobs=None, chunksize=500,
            max_fdr=None, corr_thresh=0.0001, plot=False, random_state=27, **plot_kws
        ):
        '''
        Preliminary attempt at continuous version.
        '''
        #remap and clean up input
        x_clean = self.align_series(x, verbose=verbose, handle_duplicates=handle_duplicates)

        assert x_clean.index.is_unique, "cannot get correlated terms since IDs are not unique"

        if len(x_clean) < 10000:
            raise Exception("Expected whole-genome list of continuous values (> 10,000)")
        
        #get tfidfs and reorder x_clean to match
        tfidfs, terms, entities = self.get_subset_tfidfs(entities=x_clean.index, terms=None)
        x_clean = x_clean[entities]

        if method == "f-test":
            corr_terms = pd.DataFrame(f_regression(tfidfs, x_clean), index=["F-stat", "p-val"], columns=terms).T
            corr_terms["FDR"] = scipy.stats.false_discovery_control(corr_terms["p-val"], method="bh")
            corr_terms["Correlation"] = r_regression(tfidfs, x_clean)
        elif method == "permutation":
            rng = np.random.default_rng(random_state)
            res = self._permutation_pool(
                tfidfs, x_clean, terms, rng, n_permutations=n_permutations, n_jobs=n_jobs, chunksize=chunksize
            )
            counts = pd.DataFrame(np.add.reduce(res[1]), index=res[0].index, columns=["Smaller", "Larger"])
            pvalues = (2*((counts+1) / (n_permutations+1)).min(axis=1)).clip(0,1)
            fdr = pd.Series(scipy.stats.false_discovery_control(pvalues, method="bh"), index=pvalues.index).rename("FDR")
            corr_terms = pd.DataFrame({"p-val":pvalues, "FDR":fdr, "Correlation":res[0]})
        else:
            raise Exception("invalid method")

        corr_terms.index.name = "Term"
        #if max FDR and corr thresh provided, filter in order to reduce overhead
        if max_fdr is not None:
            corr_terms = corr_terms.loc[
                (corr_terms["FDR"] < max_fdr)
            ]
        if corr_thresh is not None:
            corr_terms = corr_terms.loc[
                (corr_terms["Correlation"].abs() >= corr_thresh)
            ]
        if corr_terms.shape[0] == 0:
            return None
        corr_terms["Stopword"] = corr_terms.index.isin(self.stopwords)
        corr_terms["Synonyms"] = self.map_synonyms(corr_terms.index)
        corr_terms["Weighted Query Info"] = pd.DataFrame(tfidfs.T.multiply(x_clean).T.mean(axis=0), columns=terms).iloc[0]
        corr_terms["Total Info"] = self.get_total_info(corr_terms.index, entities=entities)
        corr_terms["IDF"] = self.idfs
        corr_terms[f"n Matching {self.ENTITY_TYPE}s Overall"] = self.n_matching_overall
        corr_terms["Directed Effect Size"] = corr_terms["Correlation"].abs()*corr_terms["Weighted Query Info"]
        corr_terms["Effect Size"] = corr_terms["Directed Effect Size"].abs()
        corr_terms.reset_index(inplace=True)

        #plot if desired
        if plot:
            fig = self.plot_correlated(corr_terms, **plot_kws)
            fig.show()
        return corr_terms 
    
    def get_enriched_continuous(
            self, x=None, corr_terms=None, max_fdr=0.05, corr_thresh=0.2, n_range=[None, "default"],   
            n=10, sort_by="Effect Size", plot=False, group_subterms=True,  
            handle_duplicates=False, verbose=False, **plot_kws
        ):
        '''
        Preliminary attempt at continuous version.
        '''

        #get terms
        if x is not None:
            assert corr_terms is None, "cannot provide x and corr_terms"
            corr_terms = self.get_correlated_terms(
                x, verbose=verbose, handle_duplicates=handle_duplicates,
                max_fdr=max_fdr, corr_thresh=corr_thresh
            )
        else:
            assert corr_terms is not None, "if x not provided, must provide corr_terms"

        #handle no terms surviving thresholds
        if corr_terms is None or corr_terms.shape[0] == 0:
            return None

        #set thresholds
        min_n = corr_terms[f"n Matching {self.ENTITY_TYPE}s Overall"].min() if n_range[0] is None else n_range[0]
        if n_range[1] is None:
            max_n = corr_terms["IDF"].min()
        elif n_range[1] == "default":
            max_n = self.max_n
        else:
            max_n = n_range[1]

        #filter based on thresholds
        enriched = corr_terms.loc[
            (~corr_terms["Stopword"]) &
            (corr_terms["FDR"] < max_fdr) & 
            (corr_terms[f"n Matching {self.ENTITY_TYPE}s Overall"] >= min_n) &
            (corr_terms[f"n Matching {self.ENTITY_TYPE}s Overall"] < max_n) &
            (corr_terms["Correlation"].abs() >= corr_thresh)
        ].copy()

        #handle no terms surviving thresholds
        if enriched.shape[0] == 0:
            return None

        enriched = self._sort_and_get_top_n(
            enriched, sort_by, n, group_subterms, continuous=True, filter=True
        )

        if plot:
            fig = self.plot_enriched_continuous(enriched, n=n, **plot_kws)
            fig.show()

        return enriched
    
    def plot_enriched_continuous(
        self, to_plot, filter=False, sort_by="Effect Size", n=None, group_subterms=True,
        title="Top Correlated TEA Terms", cluster_terms=True, split_on=" ", show_terms=None, **kws
    ):
        '''
        Interactive plot for continuous version.
        '''
        #check that table is not none
        if to_plot is None:
            raise RuntimeError("No enriched terms")

        groupby = "Term Group" if group_subterms else "Term"
        to_plot = self._prep_for_plot(to_plot, groupby, split_on, no_term_to_entity=True)
        #generate clusters
        term_cluster, _, term_group_cluster = self.gen_clusters(
            to_plot["Term"], None, term_groups=to_plot.set_index("Term")["Term Group"]
        )
        to_plot = to_plot.merge(
            term_group_cluster if group_subterms else term_cluster, how="left", on=groupby
        )

        #sort and trim
        if show_terms is not None:
            to_plot = to_plot.set_index("Term").loc[show_terms].reset_index()
            if len(to_plot) == 0:
                raise RuntimeError("Could not find terms listed in show_terms.")
        else:
            to_plot = self._sort_and_get_top_n(
                to_plot, sort_by, n, group_subterms, continuous=True,
                filter=filter, **kws
            )

        #create figure
        height = 500
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            row_heights=[250,height-250],
            vertical_spacing=0
        )
        fig.append_trace(
            go.Bar(
                y=to_plot["-log10 FDR"],
                x=to_plot[groupby], 
                marker=dict(color="#0aa18f"),
                orientation='v',
                customdata=self._customdata(to_plot, continuous=True),
                hovertemplate=self._hovertemplate(continuous=True),
            ), row=1, col=1
        )
        fig.append_trace(
            go.Scatter(
                y=to_plot['Correlation'],
                x=to_plot[groupby],
                mode="markers",
                marker=dict(
                    size=to_plot["Correlation"].abs()*50,
                    opacity=1,
                    line=dict(width=2, color='DarkSlateGrey'),
                    color=to_plot["Correlation"],
                    colorbar=dict(title="Correlation", len=0.5),
                    colorscale="RdBu",
                    cmax=1, cmid=0, cmin=-1
                ),
                customdata=to_plot[["Term", "Correlation"]].fillna('N/A'),
                hovertemplate="Term: %{customdata[0]}<br>" +\
                                "(correlation= %{customdata[1]:.4f})" +\
                                "<extra></extra>",
            ), row=2, col=1
        )
        fig.update_layout(
            barmode='overlay',
            autosize=True,
            height=height,
            showlegend=False,
            xaxis_type='category',
            yaxis_title="-log10(FDR)",
            yaxis2_title="Correlation",
            yaxis2_showexponent='all',
            yaxis2_exponentformat='e',
            title=title
        )

        #determine term and entity order
        if cluster_terms:
            term_order = to_plot.sort_values("Order").groupby(groupby).first().sort_values("Order")["Clipped "+groupby]
        else:
            term_order = to_plot.sort_values("FDR").groupby(groupby).first().sort_values("FDR")["Clipped "+groupby]
        fig.update_xaxes(
            categoryorder='array', 
            categoryarray=term_order.index,
            tickmode='array',
            tickvals=term_order.index,
            ticktext=term_order.values
        )

        return fig
    
    def plot_correlated(self, *kws):
        '''
        Interactive volcano for continuous version.
        '''
        return self.plot_volc(continuous=True, *kws)
    
    def plot_x_vs_term(self, x, term, title=None, **kws):
        '''
        Interactive plot of continuous versus tfidfs.
        '''
        x_clean = self.align_series(x, **kws)
        tfidfs = self.get_tfidf_series(term, entities=x_clean.index)
        trimmed_tfidfs = tfidfs.loc[lambda x: x > 0]
        x_clean, trimmed_tfidfs = x_clean.align(tfidfs, join="inner")
        height = 500
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            row_heights=[100,height-100],
            vertical_spacing=0
        )
        fig.add_trace(
            go.Histogram(
                x=x_clean,
            ), row=1, col=1
        )
        fig.append_trace(
            go.Scatter(
                x=x_clean,
                y=trimmed_tfidfs,
                text=tfidfs.index,
                textposition="bottom right",
                mode="markers",
                marker=dict(
                    opacity=0.75,
                    color="#2b2c2c"
                )
            ), row=2, col=1
        )
        fig.update_layout(
            height=500,
            autosize=True,
            yaxis_title=f"Input<br>Count",
            yaxis2_title=f"'{term}' TF-IDF",
            xaxis2_title="Input Values",
            title=title
        )
        return fig
    
def _shuffled_correlation(_, tfidfs, x_clean, observed, rng):
    shuffled_index = np.arange(len(x_clean))
    rng.shuffle(shuffled_index)
    shuffled_x = x_clean.iloc[shuffled_index]
    null = r_regression(tfidfs, shuffled_x)
    return np.array([null <= observed, null >= observed]).T

def save(tea, filename):
    with open(filename, 'wb') as f:
        dill.dump(tea, f)

def load(filename):
    with open(filename, 'rb') as f:
        tea = dill.load(f)
    return tea

class GeneTEA(xTEA):
    def __init__(self, corpus, sources=None, custom_stopwords=None, vec_kwargs=dict(max_df=0.5, min_df=3, binary=False)):
        super().__init__(corpus, sources, "Gene", custom_stopwords, vec_kwargs)


class PharmaTEA(xTEA):
    def __init__(self, corpus, sources=None, custom_stopwords=None, vec_kwargs=dict(max_df=0.5, min_df=3, binary=False)):
        super().__init__(corpus, sources, "Drug", custom_stopwords, vec_kwargs)