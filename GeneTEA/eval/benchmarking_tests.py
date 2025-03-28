import pandas as pd
import numpy as np
import time
import itertools
from ..utils import tril_stack
from .competitors import Competitors

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

from .benchmarking_sets import get_benchmarking_sets, subsample_and_combo

def find_coeffs(sets):
    #compute overlap coefficient
    def overlap_coeff(A, B):
        return len(A & B) / min(len(A), len(B))
    #for all combinations of sets without repeated combinations
    coeffs = [
        pd.Series({"Node1":id2, "Node2":id1, "Coeff":overlap_coeff(sets.loc[id1], sets.loc[id2])})
        for id1, id2 in itertools.combinations(sets.index, 2)
    ]
    return pd.concat(coeffs, axis=1).T

def get_overlaps(libraries, thresh=0.5, n=500, reps=10, random_seed=27):
    '''
    Find the number of high overlap pairs in random samples of gene sets from several libraries.
    '''
    overlap_coeffs, overlap_counts = {}, []
    #setup random state
    rng = np.random.default_rng(seed=random_seed)
    for i in range(reps):
        print(i)
        overlap_coeffs[i] = {}
        #for each library, find high overlaps
        for k in sorted(libraries.keys()):
            #get overlap coeffcient
            ov = find_coeffs(libraries[k].sample(n, random_state=rng.bit_generator))
            #count number pairs higher then threshold
            counts = pd.Series({"model":k, "high":(ov["Coeff"] >= thresh).sum(), "total":ov.shape[0]})
            #compute percent
            counts["percent"] = counts["high"] / counts["total"] * 100
            overlap_coeffs[i][k] = ov
            overlap_counts.append(counts)
    return overlap_coeffs, pd.concat(overlap_counts, axis=1).T


def benchmark_false_discovery(tea, competitors=["g:GOSt", "Enrichr"], max_fdr=0.05, n_samp=10, min_genes=2, random_seed=27):
    '''
    Benchmark false discovery by testing random gene sets of varying sizes.
    '''
    t0 = time.time()
    rng = np.random.default_rng(seed=random_seed)
    
    #get random sets of genes of various sizes
    test_sets = []
    for i in [3, 5, 10, 15, 20, 25, 50, 100, 250, 500, 1000]:
        test_sets.extend(rng.choice(tea.entities, (n_samp, i)))

    #setup competitor models
    if competitors is not None:
        competitors = Competitors(competitors)
    
    #get number enriched 
    false_discoveries ={}
    for i, gs in enumerate(test_sets):
        gs = list(gs)
        if i % 25 == 0: print(i)

        #get enrichment for the null gene set
        if competitors is not None:
            enriched_for_null = competitors.query(gs, max_fdr=max_fdr, min_genes=min_genes)
        #indicate error with -1
        else: 
            enriched_for_null = {c:-1 for c in competitors}
        try:
            enriched_for_null["GeneTEA"] = tea.get_enriched_terms(
                gs, n=None, max_fdr=max_fdr, min_genes=min_genes,
                group_subterms=False
            )
            #if no enriched terms, return empty df so shape is 0
            if enriched_for_null["GeneTEA"] is None:
                enriched_for_null["GeneTEA"] = pd.DataFrame()
        #indicate error with -1
        except:
            enriched_for_null["GeneTEA"] = -1

        #count number enriched terms
        false_discoveries[i] = pd.Series({
            (k, len(gs), ", ".join(gs)):result.shape[0] if result is not None else None
            for k, result in enriched_for_null.items()
        })
    false_discoveries = pd.concat(false_discoveries).rename_axis(["idx", "model", "len_query", "genes"]).to_frame("false_discoveries").reset_index()

    print(f"took {round((time.time() - t0) / 60, 3)} mins")

    return false_discoveries


class MedCPT():
    '''
    MedCPT model from NCBI.
    '''
    def __init__(self):
        self.cross_enc = "ncbi/MedCPT-Cross-Encoder"
        self.query_enc = "ncbi/MedCPT-Query-Encoder"
        self.article_enc = "ncbi/MedCPT-Article-Encoder"

    def rank_with_crossenc(self, query, articles, max_n=None):
        '''
        Rank articles against a query using the CrossEnc.
        '''
        tokenizer = AutoTokenizer.from_pretrained(self.cross_enc)
        model = AutoModelForSequenceClassification.from_pretrained(self.cross_enc)
        model.eval()
        pairs = [[query, to_rank[:max_n]] for to_rank in articles]
        with torch.no_grad():
            encoded = tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )
            logits = model(**encoded).logits.squeeze(dim=1)
        return logits.numpy()
    
    def embed_query(self, queries):
        '''
        Embed a query using the QEnc.
        '''
        tokenizer = AutoTokenizer.from_pretrained(self.query_enc)
        model = AutoModel.from_pretrained(self.query_enc)
        model.eval()
        with torch.no_grad():
            encoded = tokenizer(
                queries, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=512#64, 
            )
            # encode the queries (use the [CLS] last hidden states as the representations)
            embeds = model(**encoded).last_hidden_state[:, 0, :]
            return embeds.numpy()
        
    def embed_article(self, articles, max_n=None):
        '''
        Embed an article using the DEnc.
        '''
        tokenizer = AutoTokenizer.from_pretrained(self.article_enc)
        model = AutoModel.from_pretrained(self.article_enc)
        model.eval()
        with torch.no_grad():
            encoded = tokenizer(
                articles[:max_n], 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=512,
            )
            
            # encode the queries (use the [CLS] last hidden states as the representations)
            embeds = model(**encoded).last_hidden_state[:, 0, :]
            return embeds
        
    def article_redundancy(self, articles=None, embs=None, max_n=100):
        '''
        Measure redundancy of articles by computing cos sim over pairwise embeddings.
        '''
        article_emb = self.embed_article(articles)[:max_n] if embs is None else embs
        article_redundancy = tril_stack(pd.DataFrame(cosine_similarity(article_emb)))
        return article_redundancy.to_list()
    
    def score(self, query, articles, index=None, max_n=100):
        '''
        Score articles against a query by various metrics.
        '''
        #if query is list, find length
        if isinstance(query, list):
            len_query = len(query)
            query = ", ".join(query)
        else:
            len_query = None
        #get redundancy for articles
        redundancy = [self.article_redundancy(a, max_n=max_n) if len(a) > 0 else [] for a in articles]
        if index is None: index=list(range(len(articles)))
        #score each term in an article with the CrossEnc
        indiv = pd.concat([pd.DataFrame({
            "model":model,
            "query":query,
            "article":articles[i], 
            "rank":np.arange(len(articles[i]))+1, 
            "indiv_ranking":self.rank_with_crossenc(query, articles[i], max_n=max_n) if len(articles[i]) > 0 else None
        }) for i, model in enumerate(index)]).set_index(["model", "article"])
        #score the joined terms with the CrossEnC
        joined_articles = [". ".join(a) for a in articles]
        joined = pd.DataFrame({
            "query":query,
            "len_query":len_query,
            "articles":joined_articles,
            "n":[0 if a is None else len(a) for a in articles],
            "joined_ranking":self.rank_with_crossenc(query, joined_articles),
            "num_high_redundancy":[None if len(r) == 0 else len([x for x in r if x > 0.95]) for r in redundancy],
            "frac_high_redundancy":[None if len(r) == 0 else len([x for x in r if x > 0.95]) / len(r) for r in redundancy],
        }, index=index).rename_axis("model")
        return indiv, joined

def benchmark_w_medcpt(
    tea, gene_sets, competitors=["g:GOSt", "Enrichr"], TEA="GeneTEA", use_groups=True,
    n=None, max_fdr=0.05, min_genes=2, exclude_terms=[], max_n=100,
    with_random=False
):
    '''
    Benchmark gene sets using MedCPT.
    '''
    #setup MedCPT and competitors
    medcpt = MedCPT()
    comps = Competitors(competitors) if competitors is not None else None
    rng = np.random.default_rng(27)

    t0 = time.time()

    #loop through and score
    indiv, joined = {}, {}
    for gs, genes in gene_sets.items():
        #get competitors results
        if competitors is not None:
            results = comps.top_n(genes=genes, n=n, max_fdr=max_fdr, min_genes=min_genes)
        else:
            results = {}
        #get GeneTEA results
        try:
            results[TEA] = tea.get_enriched_terms(
                genes, n=n, max_fdr=max_fdr, min_genes=min_genes, exclude_terms=exclude_terms,
                group_subterms=False
            )
        except Exception as e:
            print("GeneTEA errored on ", gs, "Exception", e)
            results[TEA] = None
        #optionally use term groups rather then terms
        if results[TEA] is not None and use_groups:
            results[TEA+"-Grouped"] = results[TEA].rename(columns={"Term":"Sub-Term", "Term Group":"Term"})
        #score random terms
        if with_random:
            results["Random"] = tea.terms.to_series().to_frame("Term").sample(max_n, random_state=rng.bit_generator)
        #get the top terms for each model as a list
        top_terms = pd.Series({
            model:res["Term"].head(max_n).unique().tolist()
            if res is not None and len(res) > 0 else []
            for model, res in results.items() 
        })
        #score them
        if len(top_terms) > 0:
            indiv[gs], joined[gs] = medcpt.score(
                genes, top_terms, index=top_terms.index, max_n=max_n
            )
    indiv = pd.concat(indiv).rename_axis(["gene_set", "model", "term"]).reset_index()
    joined = pd.concat(joined).rename_axis(["gene_set", "model"]).reset_index()

    print(f"took {round((time.time() - t0) / 60, 3)} mins")

    return indiv, joined


def benchmark_all(tea, paths="taiga", competitors=["g:GOSt", "Enrichr"], run=["FDR", "Hallmark", "Experimental"], suffix=""):
    '''
    Run all benchmarking tests.
    ''' 
    if "FDR" in run:
        false_discoveries = benchmark_false_discovery(tea, competitors=competitors, n_samp=10)
        false_discoveries.to_csv(f"false_discoveries{suffix}.csv", index=None)

    benchmarking_sets = get_benchmarking_sets(paths)
    if "Hallmark" in run:
        hallmark_sets = subsample_and_combo(benchmarking_sets["Hallmark Collection"])

        hallmark_indiv, hallmark_scores = benchmark_w_medcpt(
            tea, hallmark_sets["Random Sub-samples"].set_index("gene_set")["genes"],
             competitors=competitors,
        )
        hallmark_scores = hallmark_scores.merge(
            hallmark_sets["Random Sub-samples"], how="left", on="gene_set"
        )
        hallmark_scores["n sampled"] = "n="+hallmark_scores["size"].astype(str)
        hallmark_indiv.to_csv(f"Hallmark_subsamples_indiv{suffix}.csv")
        hallmark_scores.to_csv(f"Hallmark_subsamples_scores{suffix}.csv")

        hallmark_indiv_combo, hallmark_scores_combo = benchmark_w_medcpt(
            tea, hallmark_sets["Random Combos"]["combined"],
            competitors=competitors,
        )
        hallmark_indiv_combo.to_csv(f"Hallmark_combo_indiv{suffix}.csv", index=None)
        hallmark_scores_combo.to_csv(f"Hallmark_combo_scores{suffix}.csv", index=None)

    if "Experimental" in run:
        for s in benchmarking_sets.index.get_level_values(0).unique():
            if "Hallmark" in s:
                continue
            print(s, len(benchmarking_sets[s]))
            indiv, scores = benchmark_w_medcpt(tea, benchmarking_sets[s],  competitors=competitors,)
            indiv.to_csv(s.replace(" ", "_")+f"_indiv{suffix}.csv", index=None)
            scores.to_csv(s.replace(" ", "_")+f"_scores{suffix}.csv", index=None)


from sklearn.neighbors import NearestNeighbors
import networkx as nx

def find_nearest_neighbors(x, y, xs, ys, mapping, ks=[1, 10, 20, 50, 100]):
    '''
    Find the nearest neighbors for y given a model trained on x at various ks.
    '''
    #train model with x
    nbrs = NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=-1).fit(x)
    k_matching_neighbors = {}
    print("testing ks", ks)
    for k in ks:
        print("k", k)
        #query with y
        _, indices = nbrs.kneighbors(y, k)
        neighbors = pd.DataFrame(indices, index=ys)
        index_to_entity = pd.Series(xs)
        #label neighbors
        matching_neighbors = {}
        for x_i, y_i in mapping.items():
            matching_neighbors[x_i] = neighbors.loc[x_i].replace(index_to_entity).isin(y_i).sum() / min(len(y_i), k)
        k_matching_neighbors[k] = pd.Series(matching_neighbors)
    k_matching_neighbors = pd.concat(k_matching_neighbors, axis=0).reset_index()
    #get fraction of y's targets that are in y's neighbors
    frac_knn = k_matching_neighbors.groupby("level_0")[0].apply(lambda z: (z > 0).sum() / len(mapping))
    return nbrs, k_matching_neighbors, frac_knn.to_frame("Fraction").rename_axis("k").reset_index()

def gen_knn_graph(nbrs, x, y, xs, ys, k=10):
    '''
    Get the graph representation of the nearest neighbors.
    '''
    dist, x_id = nbrs.kneighbors(y, k)
    G = nx.Graph()
    for yi, d, xi in zip(ys, dist, x_id):
        G.add_weighted_edges_from([(yi, np.array(xs)[xj], di) for di, xj in zip(d, xi)], weight="weight")
    return G


if __name__ == "__main__":
    from ..train import load_models
    name = "v2"
    tea = load_models(name=name)

    benchmark_all(tea, suffix="")